"""
Efficient embedding loader supporting Shards (Bluesky) and Single File (Reddit)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import config.config as cfg
import config.dataset_config as ds_cfg

class EmbeddingLoader:
    def __init__(self, embedding_dir, preload_all=True):
        self.embedding_dir = embedding_dir
        # Use .get() to avoid key errors if config isn't fully set yet, defaults to 'sharded'
        self.mode = cfg.DATASET_CONF.get('embedding_mode', 'sharded') 
        self.index = self._load_index()
        
        # Cache storage
        self.shard_cache = {}  # For sharded mode
        self.full_embedding_cache = None # For single file mode
        
        if preload_all:
            print(f"Preloading embeddings ({self.mode} mode)...")
            self._preload_embeddings()
            
    def _load_index(self):
        """Load index mapping"""
        print(f"Loading embedding index from {self.embedding_dir}...")
        
        if self.mode == 'sharded':
            # Bluesky: Multiple index shards
            index_files = sorted(Path(self.embedding_dir).glob("index_shard_*.parquet"))
            if not index_files:
                raise FileNotFoundError(f"No index files found in {self.embedding_dir}")
            
            indices = []
            for idx_file in index_files:
                df = pd.read_parquet(idx_file)
                indices.append(df)
            combined_index = pd.concat(indices, ignore_index=True)
            
            # Lookup key: (tree_id, post_id, user_id) -> (shard, row)
            combined_index['key'] = list(zip(
                combined_index['tree_id'], 
                combined_index['post_id'], 
                combined_index['user_id']
            ))
            return combined_index.set_index('key')[['shard', 'row']].to_dict('index')

        elif self.mode == 'single_file':
            # Reddit: Single index.parquet
            index_path = os.path.join(self.embedding_dir, "index.parquet")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            df = pd.read_parquet(index_path)
            
            # If the parquet just lists post_ids in order matching the npy file:
            if 'row_idx' not in df.columns:
                df['row_idx'] = np.arange(len(df))
                
            # Create lookup: post_id -> row_idx
            return df.set_index('post_id')['row_idx'].to_dict()

    def _preload_embeddings(self):
        """Load actual vectors into RAM"""
        if self.mode == 'sharded':
            # Get unique shard IDs from index
            shard_ids = set(info['shard'] for info in self.index.values())
            for shard_id in sorted(shard_ids):
                shard_file = os.path.join(self.embedding_dir, f"embeddings384_fp16_shard_{shard_id:05d}.npy")
                if os.path.exists(shard_file):
                    self.shard_cache[shard_id] = np.load(shard_file).astype(np.float32)
                    
        elif self.mode == 'single_file':
            npy_path = os.path.join(self.embedding_dir, "embeddings384_fp16.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"Embedding file not found: {npy_path}")
            
            print(f"Loading large embedding file: {npy_path}")
            self.full_embedding_cache = np.load(npy_path).astype(np.float32)

    def get_embeddings(self, tree_ids, post_ids, user_ids):
        """
        Get embeddings for a batch
        """
        embeddings = []
        
        for tree_id, post_id, user_id in zip(tree_ids, post_ids, user_ids):
            
            if self.mode == 'sharded':
                key = (tree_id, post_id, user_id)
                if key not in self.index:
                    embeddings.append(np.zeros(cfg.EMBEDDING_DIM, dtype=np.float32))
                    continue
                
                shard_info = self.index[key]
                shard_id = shard_info['shard']
                row = shard_info['row']
                
                if shard_id in self.shard_cache:
                    embeddings.append(self.shard_cache[shard_id][row])
                else:
                    embeddings.append(np.zeros(cfg.EMBEDDING_DIM, dtype=np.float32))

            elif self.mode == 'single_file':
                # Reddit lookup uses post_id
                if post_id not in self.index:
                    embeddings.append(np.zeros(cfg.EMBEDDING_DIM, dtype=np.float32))
                    continue
                
                row_idx = self.index[post_id]
                embeddings.append(self.full_embedding_cache[row_idx])
        
        return np.array(embeddings, dtype=np.float32)

    def clear_cache(self):
        """Clear cache to free memory"""
        print("Clearing embedding cache...")
        self.shard_cache.clear()
        self.full_embedding_cache = None
        # Force garbage collection to reclaim memory immediately
        gc.collect()