"""
Precompute and cache all features to disk (Single Process - Polars Optimized)
"""

import os
import pickle
import pandas as pd
import polars as pl
from tqdm import tqdm

# Added force_recompute parameter
def precompute_features(tree_ids, edges_df, meta_df, posts_df, feature_builder, cache_path, desc="Precomputing", force_recompute=False):
    """
    Unified precomputer using Polars for fast lookup.
    """
    # Check if cache exists AND we are not forcing recompute
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached features from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache (will recompute): {e}")

    print(f"{desc} features for {len(tree_ids)} trees...")
    
    # 1. Optimize Edges: Partition by tree_id using Polars
    print("Partitioning edges with Polars...")
    edges_pl = pl.from_pandas(edges_df)
    
    if edges_pl.height > 0:
        edges_dict = {
            k: v for k, v in edges_pl.partition_by(["tree_id"], as_dict=True).items()
        }
    else:
        edges_dict = {}
    
    feature_cache = {}
    
    # 2. Sequential Loop
    for tree_id in tqdm(tree_ids, desc=desc):
        if tree_id not in meta_df.index:
            continue
            
        key = (tree_id,)
        if key in edges_dict:
            tree_edges = edges_dict[key].to_pandas()
        else:
            tree_edges = pd.DataFrame(columns=edges_df.columns)
            
        meta_row = meta_df.loc[tree_id]
        
        data = feature_builder.build_tree_features(tree_id, tree_edges, meta_row)
        
        if data is not None:
            feature_cache[tree_id] = data
            
    print(f"Saving feature cache to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(feature_cache, f)
        
    return feature_cache