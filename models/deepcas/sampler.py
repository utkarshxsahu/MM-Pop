"""
Random walk sampling for conversation trees (USER IDS).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import config

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


class RandomWalkSampler:
    """Sample random walk paths from conversation trees."""
    
    def __init__(self, K=None, T=None, p_jump=None, seed=None):
        self.K = K or config.K
        self.T = T or config.T
        self.p_jump = p_jump or config.P_JUMP
        self.rng = np.random.RandomState(seed or config.RANDOM_SEED)
    
    def precompute_all_paths(self, tree_ids, edges_df, posts_df, desc="Sampling"):
        """Precompute paths for all trees (OPTIMIZED with Polars)."""
        print(f"Pre-partitioning data with Polars for {desc}...")

        edges_filtered = edges_df[edges_df["tree_id"].isin(tree_ids)]
        posts_filtered = posts_df[posts_df["tree_id"].isin(tree_ids)]

        edges_dict = {}
        posts_dict = {}

        if pl is not None:
            edges_pl = pl.from_pandas(edges_filtered)
            posts_pl = pl.from_pandas(posts_filtered)

            if edges_pl.height > 0:
                for tree_id, group in edges_pl.partition_by("tree_id", as_dict=True).items():
                    edges_dict[tree_id[0]] = group.to_pandas()

            if posts_pl.height > 0:
                for tree_id, group in posts_pl.partition_by("tree_id", as_dict=True).items():
                    posts_dict[tree_id[0]] = group.to_pandas()
        else:
            if len(edges_filtered) > 0:
                for tree_id, group in edges_filtered.groupby("tree_id"):
                    edges_dict[tree_id] = group.copy()

            if len(posts_filtered) > 0:
                for tree_id, group in posts_filtered.groupby("tree_id"):
                    posts_dict[tree_id] = group.copy()
        
        print(f"Starting path sampling for {len(tree_ids)} trees...")
        path_cache = {}
        
        for tree_id in tqdm(tree_ids, desc=desc):
            tree_edges = edges_dict.get(tree_id, pd.DataFrame())
            tree_posts = posts_dict.get(tree_id, pd.DataFrame())
            
            if len(tree_posts) == 0:
                continue
            
            paths, post_to_user = self._sample_paths_from_tree(tree_id, tree_edges, tree_posts)
            
            if paths is not None:
                path_cache[tree_id] = {
                    'paths': paths,  # Paths contain user_ids
                    'post_to_user': post_to_user,
                    'num_nodes': len(post_to_user)
                }
        
        return path_cache
    
    def _sample_paths_from_tree(self, tree_id, tree_edges, tree_posts):
        """
        Sample paths from tree.
        Returns paths with USER IDs (as strings).
        """
        if len(tree_edges) == 0:
            # No edges - might just be root
            tree_posts = tree_posts[tree_posts['depth'] == 0]
            if len(tree_posts) == 0:
                return None, None
        else:
            # Get post_ids from early window edges
            early_post_ids = set(tree_edges['post_id'].unique()) | set(tree_edges['reply_to'].unique())
            tree_posts = tree_posts[tree_posts['post_id'].isin(early_post_ids)].copy()
            
            if len(tree_posts) == 0:
                return None, None
        
        # Create mappings: post_id -> local_idx and post_id -> user_id
        post_ids = tree_posts['post_id'].values
        user_ids = tree_posts['user_id'].astype(str).values  # Ensure strings
        
        post_to_idx = {pid: idx for idx, pid in enumerate(post_ids)}
        post_to_user = {pid: uid for pid, uid in zip(post_ids, user_ids)}
        
        # Find root
        root_posts = tree_posts[tree_posts['depth'] == 0]
        if len(root_posts) == 0:
            return None, None
        root_post_id = root_posts.iloc[0]['post_id']
        root_idx = post_to_idx[root_post_id]
        
        # Build adjacency using LOCAL indices (for sampling)
        children_map = defaultdict(list)
        for _, edge in tree_edges.iterrows():
            parent_id = edge['reply_to']
            child_id = edge['post_id']
            if parent_id in post_to_idx and child_id in post_to_idx:
                children_map[post_to_idx[parent_id]].append(post_to_idx[child_id])
        
        # Sample paths (in local indices)
        all_indices = list(range(len(post_ids)))
        local_paths = [self._sample_one_path(root_idx, all_indices, children_map) 
                       for _ in range(self.K)]
        
        # Convert paths from local indices to USER IDs (strings)
        idx_to_post = {idx: pid for pid, idx in post_to_idx.items()}
        user_paths = []
        for local_path in local_paths:
            user_path = [post_to_user[idx_to_post[local_idx]] for local_idx in local_path]
            user_paths.append(user_path)
        
        return user_paths, post_to_user
    
    def _sample_one_path(self, root_idx, all_indices, children_map):
        """
        Sample one random walk path with degree-based weighting.
        Degrees computed from early window only (no data leakage).
        """
        path = []
        current = root_idx
        
        # Compute out-degrees from early window children_map
        # children_map is built from tree_edges (early window), so this is safe!
        degrees = {idx: len(children_map.get(idx, [])) for idx in all_indices}
        
        for step in range(self.T):
            path.append(current)
            if step == self.T - 1:
                break
            
            if self.rng.random() < self.p_jump:
                # Jump to random node, weighted by degree
                weights = np.array([degrees.get(idx, 0) + 1 for idx in all_indices], dtype=float)
                weights = weights / weights.sum()
                current = self.rng.choice(all_indices, p=weights)
            else:
                children = children_map.get(current, [])
                if len(children) > 0:
                    # Select child weighted by their degree (prefer viral branches)
                    weights = np.array([degrees.get(child, 0) + 1 for child in children], dtype=float)
                    weights = weights / weights.sum()
                    current = self.rng.choice(children, p=weights)
                else:
                    # Leaf node, jump to random node
                    weights = np.array([degrees.get(idx, 0) + 1 for idx in all_indices], dtype=float)
                    weights = weights / weights.sum()
                    current = self.rng.choice(all_indices, p=weights)
        
        return path
