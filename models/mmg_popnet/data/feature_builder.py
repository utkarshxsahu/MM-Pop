import numpy as np
import pandas as pd
import torch
import polars as pl
import config.config as cfg
from transformers import AutoTokenizer

class FeatureBuilder:
    """
    Standard builder for Frozen models (MLP, GraphSAGE, GAT).
    Uses pre-computed embeddings.
    """
    def __init__(self, embedding_loader, preprocessor, posts_df, user_degrees_df=None, image_loader=None):
        self.embedding_loader = embedding_loader
        self.preprocessor = preprocessor
        self.image_loader = image_loader
        
        if isinstance(posts_df, pd.DataFrame):
            self.posts_pl = pl.from_pandas(posts_df)
        else:
            self.posts_pl = posts_df
            
        self.posts_dict = {
            k: v for k, v in self.posts_pl.partition_by(["tree_id"], as_dict=True).items()
        }
        
        self.user_degrees_lookup = {}
        if user_degrees_df is not None:
            self.user_degrees_lookup = user_degrees_df.set_index('user_id')['follower_count'].to_dict()
        
        self.embedding_projection = torch.nn.Linear(cfg.EMBEDDING_DIM, cfg.EMBEDDING_PROJECTION_DIM, bias=False)
        torch.nn.init.xavier_uniform_(self.embedding_projection.weight)
        self.embedding_projection.eval()

    def build_tree_features(self, tree_id, edges_df_slice, meta_row):
        key = (tree_id,)
        if key not in self.posts_dict:
            return None
        
        all_tree_posts = self.posts_dict[key].to_pandas()
        
        if len(edges_df_slice) > 0:
            valid_post_ids = set(edges_df_slice['post_id']).union(set(edges_df_slice['reply_to']))
        else:
            valid_post_ids = set(all_tree_posts[all_tree_posts['depth'] == 0]['post_id'])

        tree_posts = all_tree_posts[all_tree_posts['post_id'].isin(valid_post_ids)].copy()
        unique_posts = tree_posts['post_id'].unique()
        post_to_idx = {post_id: idx for idx, post_id in enumerate(unique_posts)}
        
        tree_posts['node_idx'] = tree_posts['post_id'].map(post_to_idx)
        tree_posts = tree_posts.sort_values('node_idx')
        
        embeddings = self.embedding_loader.get_embeddings(
            tree_posts['tree_id'].values,
            tree_posts['post_id'].values,
            tree_posts['user_id'].values
        )
        
        with torch.no_grad():
            emb_tensor = torch.FloatTensor(embeddings)
            embeddings_projected = self.embedding_projection(emb_tensor).numpy()
            
        node_features = self.preprocessor.transform_node_features(tree_posts, embeddings_projected)
        
        edge_list = []
        if len(edges_df_slice) > 0:
            src_indices = edges_df_slice['reply_to'].map(post_to_idx)
            dst_indices = edges_df_slice['post_id'].map(post_to_idx)
            valid_mask = src_indices.notna() & dst_indices.notna()
            
            if valid_mask.any():
                edge_list = np.stack([
                    src_indices[valid_mask].astype(int).values,
                    dst_indices[valid_mask].astype(int).values
                ], axis=1)
                
        if len(edge_list) == 0:
            edge_index = np.array([[], []], dtype=np.int64)
        else:
            edge_index = edge_list.T
            
        global_features = self.preprocessor.transform_global_features(meta_row)
        targets = self.preprocessor.transform_targets(meta_row)
        
        root_indices = tree_posts[tree_posts['depth'] == 0].index
        root_node_idx = tree_posts.loc[root_indices[0], 'node_idx'] if len(root_indices) > 0 else 0
        
        image_embedding = None
        image_pixels = None
        has_image = False
        if self.image_loader is not None:
            root_post_id = meta_row.get('root_post_id', None)
            if root_post_id is not None:
                if cfg.FREEZE_IMAGE_ENCODER:
                    img_emb = self.image_loader.get_embedding(root_post_id)
                    if img_emb is not None:
                        image_embedding = img_emb
                        has_image = True
                else:
                    img_pix = self.image_loader.get_image(root_post_id)
                    if img_pix is not None:
                        image_pixels = img_pix
                        has_image = True
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'global_features': global_features,
            'targets': targets,
            'root_node_idx': root_node_idx,
            'num_nodes': len(tree_posts),
            'tree_id': tree_id,
            'image_embedding': image_embedding,
            'image_pixels': image_pixels,
            'has_image': has_image
        }


class TokenizedFeatureBuilder:
    """
    Builder for Unfrozen (TextGraphSAGE) training.
    - Saves raw Token IDs for Transformer.
    - Saves structural features for MLP/GNN.
    - EXCLUDES frozen text embeddings.
    """
    def __init__(self, preprocessor, posts_df, user_degrees_df=None, image_loader=None,
                 model_name='sentence-transformers/all-MiniLM-L6-v2', max_len=128):
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.image_loader = image_loader
        
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Converting data to Polars for fast processing...")
        if isinstance(posts_df, pd.DataFrame):
            self.posts_pl = pl.from_pandas(posts_df)
        else:
            self.posts_pl = posts_df
            
        print("Partitioning posts by tree_id...")
        self.posts_dict = {
            k: v for k, v in self.posts_pl.partition_by(["tree_id"], as_dict=True).items()
        }
        
        self.user_degrees_lookup = {}
        if user_degrees_df is not None:
            self.user_degrees_lookup = user_degrees_df.set_index('user_id')['follower_count'].to_dict()

    def build_tree_features(self, tree_id, edges_df_slice, meta_row):
        key = (tree_id,)
        if key not in self.posts_dict:
            return None
        
        all_tree_posts = self.posts_dict[key].to_pandas()
        
        if len(edges_df_slice) > 0:
            valid_post_ids = set(edges_df_slice['post_id']).union(set(edges_df_slice['reply_to']))
        else:
            valid_post_ids = set(all_tree_posts[all_tree_posts['depth'] == 0]['post_id'])

        tree_posts = all_tree_posts[all_tree_posts['post_id'].isin(valid_post_ids)].copy()
        unique_posts = tree_posts['post_id'].unique()
        post_to_idx = {post_id: idx for idx, post_id in enumerate(unique_posts)}
        
        tree_posts['node_idx'] = tree_posts['post_id'].map(post_to_idx)
        tree_posts = tree_posts.sort_values('node_idx')
        
        text_col = 'text' if 'text' in tree_posts.columns else 'text'
        texts = tree_posts[text_col].fillna("[deleted]").astype(str).tolist()
        texts = [t if len(t.strip()) > 0 else "[deleted]" for t in texts]
        
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        structural_features = self.preprocessor.transform_node_features(
            tree_posts,
            embedding_array=None,
            exclude_features=['text_embedding']
        )
        
        edge_list = []
        if len(edges_df_slice) > 0:
            src_indices = edges_df_slice['reply_to'].map(post_to_idx)
            dst_indices = edges_df_slice['post_id'].map(post_to_idx)
            valid_mask = src_indices.notna() & dst_indices.notna()
            
            if valid_mask.any():
                edge_list = np.stack([
                    src_indices[valid_mask].astype(int).values,
                    dst_indices[valid_mask].astype(int).values
                ], axis=1)
        
        if len(edge_list) == 0:
            edge_index = np.array([[], []], dtype=np.int64)
        else:
            edge_index = edge_list.T
            
        global_features = self.preprocessor.transform_global_features(meta_row)
        targets = self.preprocessor.transform_targets(meta_row)
        
        root_indices = tree_posts[tree_posts['depth'] == 0].index
        root_node_idx = tree_posts.loc[root_indices[0], 'node_idx'] if len(root_indices) > 0 else 0
        
        image_embedding = None
        image_pixels = None
        has_image = False
        if self.image_loader is not None:
            root_post_id = meta_row.get('root_post_id', None)
            if root_post_id is not None:
                if cfg.FREEZE_IMAGE_ENCODER:
                    img_emb = self.image_loader.get_embedding(root_post_id)
                    if img_emb is not None:
                        image_embedding = img_emb
                        has_image = True
                else:
                    img_pix = self.image_loader.get_image(root_post_id)
                    if img_pix is not None:
                        image_pixels = img_pix
                        has_image = True
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'node_features': structural_features,
            'edge_index': edge_index,
            'global_features': global_features,
            'targets': targets,
            'root_node_idx': root_node_idx,
            'num_nodes': len(tree_posts),
            'tree_id': tree_id,
            'image_embedding': image_embedding,
            'image_pixels': image_pixels,
            'has_image': has_image
        }


# =============================================================================
# ROOT-ONLY FEATURE BUILDERS
# =============================================================================
# Used exclusively by foundational_runner.py when USE_ROOT_ONLY_CONDITION=True.
# Produce single-node graphs (root post only, no edges, no cascade).
# Output dict format is identical to the standard builders above —
# GraphDataset and the rest of the pipeline require zero changes.
# To disable: set USE_ROOT_ONLY_CONDITION=False in foundational_runner.py.
# =============================================================================

class RootOnlyFeatureBuilder:
    """
    Root-only builder for frozen-embedding models (GraphSAGE, GAT, MLP).

    Produces a single-node graph from the root post only.
    No edges, no cascade structure — just the root post's features.
    Compatible with FeatureBuilder output format.
    """
    def __init__(self, embedding_loader, preprocessor, posts_df,
                 user_degrees_df=None, image_loader=None):
        self.embedding_loader = embedding_loader
        self.preprocessor     = preprocessor
        self.image_loader     = image_loader

        if isinstance(posts_df, pd.DataFrame):
            self.posts_pl = pl.from_pandas(posts_df)
        else:
            self.posts_pl = posts_df

        self.posts_dict = {
            k: v for k, v in self.posts_pl.partition_by(["tree_id"], as_dict=True).items()
        }

        self.user_degrees_lookup = {}
        if user_degrees_df is not None:
            self.user_degrees_lookup = (
                user_degrees_df.set_index('user_id')['follower_count'].to_dict()
            )

        self.embedding_projection = torch.nn.Linear(
            cfg.EMBEDDING_DIM, cfg.EMBEDDING_PROJECTION_DIM, bias=False
        )
        torch.nn.init.xavier_uniform_(self.embedding_projection.weight)
        self.embedding_projection.eval()

    def build_tree_features(self, tree_id, edges_df_slice, meta_row):
        key = (tree_id,)
        if key not in self.posts_dict:
            return None

        all_tree_posts = self.posts_dict[key].to_pandas()

        # Root post only
        root_posts = all_tree_posts[all_tree_posts['depth'] == 0].copy()
        if len(root_posts) == 0:
            return None

        root_post = root_posts.iloc[[0]].copy()
        root_post['node_idx'] = 0

        embeddings = self.embedding_loader.get_embeddings(
            root_post['tree_id'].values,
            root_post['post_id'].values,
            root_post['user_id'].values
        )

        with torch.no_grad():
            emb_tensor           = torch.FloatTensor(embeddings)
            embeddings_projected = self.embedding_projection(emb_tensor).numpy()

        node_features = self.preprocessor.transform_node_features(
            root_post, embeddings_projected
        )

        # No edges
        edge_index      = np.array([[], []], dtype=np.int64)
        global_features = self.preprocessor.transform_global_features(meta_row)
        targets         = self.preprocessor.transform_targets(meta_row)

        image_embedding = None
        image_pixels    = None
        has_image       = False
        if self.image_loader is not None:
            root_post_id = meta_row.get('root_post_id', None)
            if root_post_id is not None:
                if cfg.FREEZE_IMAGE_ENCODER:
                    img_emb = self.image_loader.get_embedding(root_post_id)
                    if img_emb is not None:
                        image_embedding = img_emb
                        has_image       = True
                else:
                    img_pix = self.image_loader.get_image(root_post_id)
                    if img_pix is not None:
                        image_pixels = img_pix
                        has_image    = True

        return {
            'node_features':   node_features,
            'edge_index':      edge_index,
            'global_features': global_features,
            'targets':         targets,
            'root_node_idx':   0,
            'num_nodes':       1,
            'tree_id':         tree_id,
            'image_embedding': image_embedding,
            'image_pixels':    image_pixels,
            'has_image':       has_image
        }


class RootOnlyTokenizedFeatureBuilder:
    """
    Root-only builder for TextGraphSAGE (unfrozen transformer).

    Produces a single-node graph from the root post only.
    Compatible with TokenizedFeatureBuilder output format.
    """
    def __init__(self, preprocessor, posts_df, user_degrees_df=None,
                 image_loader=None,
                 model_name='sentence-transformers/all-MiniLM-L6-v2',
                 max_len=128):
        self.preprocessor = preprocessor
        self.max_len      = max_len
        self.image_loader = image_loader

        print(f"[RootOnly] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if isinstance(posts_df, pd.DataFrame):
            self.posts_pl = pl.from_pandas(posts_df)
        else:
            self.posts_pl = posts_df

        self.posts_dict = {
            k: v for k, v in self.posts_pl.partition_by(["tree_id"], as_dict=True).items()
        }

        self.user_degrees_lookup = {}
        if user_degrees_df is not None:
            self.user_degrees_lookup = (
                user_degrees_df.set_index('user_id')['follower_count'].to_dict()
            )

    def build_tree_features(self, tree_id, edges_df_slice, meta_row):
        key = (tree_id,)
        if key not in self.posts_dict:
            return None

        all_tree_posts = self.posts_dict[key].to_pandas()

        # Root post only
        root_posts = all_tree_posts[all_tree_posts['depth'] == 0].copy()
        if len(root_posts) == 0:
            return None

        root_post = root_posts.iloc[[0]].copy()
        root_post['node_idx'] = 0

        text_col = 'text' if 'text' in root_post.columns else 'text'
        texts    = root_post[text_col].fillna("[deleted]").astype(str).tolist()
        texts    = [t if len(t.strip()) > 0 else "[deleted]" for t in texts]

        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        structural_features = self.preprocessor.transform_node_features(
            root_post,
            embedding_array=None,
            exclude_features=['text_embedding']
        )

        # No edges
        edge_index      = np.array([[], []], dtype=np.int64)
        global_features = self.preprocessor.transform_global_features(meta_row)
        targets         = self.preprocessor.transform_targets(meta_row)

        image_embedding = None
        image_pixels    = None
        has_image       = False
        if self.image_loader is not None:
            root_post_id = meta_row.get('root_post_id', None)
            if root_post_id is not None:
                if cfg.FREEZE_IMAGE_ENCODER:
                    img_emb = self.image_loader.get_embedding(root_post_id)
                    if img_emb is not None:
                        image_embedding = img_emb
                        has_image       = True
                else:
                    img_pix = self.image_loader.get_image(root_post_id)
                    if img_pix is not None:
                        image_pixels = img_pix
                        has_image    = True

        return {
            'input_ids':       encoding['input_ids'],
            'attention_mask':  encoding['attention_mask'],
            'node_features':   structural_features,
            'edge_index':      edge_index,
            'global_features': global_features,
            'targets':         targets,
            'root_node_idx':   0,
            'num_nodes':       1,
            'tree_id':         tree_id,
            'image_embedding': image_embedding,
            'image_pixels':    image_pixels,
            'has_image':       has_image
        }