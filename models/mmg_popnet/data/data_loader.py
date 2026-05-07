import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config.config as cfg
import config.feature_config as feat_cfg

def load_early_window_data(window_minutes):
    """Load edges and meta for window"""
    edges_path = os.path.join(cfg.EARLY_WINDOW_DIR, f"edges_{window_minutes}min.parquet")
    meta_path = os.path.join(cfg.EARLY_WINDOW_DIR, f"meta_{window_minutes}min.parquet")
    
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Edges file not found: {edges_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    edges_df = pd.read_parquet(edges_path)
    meta_df = pd.read_parquet(meta_path)
    
    return edges_df, meta_df

def get_persistent_splits(tree_ids, window_minutes):
    """
    Load splits from disk if they exist, otherwise compute and save them.
    Location: {EARLY_WINDOW_DIR}/splits/
    Files: train_ids_{win}min.npy, val_ids_{win}min.npy, test_ids_{win}min.npy
    """
    split_dir = os.path.join(cfg.EARLY_WINDOW_DIR, "splits")
    os.makedirs(split_dir, exist_ok=True)
    
    train_path = os.path.join(split_dir, f"train_ids_{window_minutes}min.npy")
    val_path = os.path.join(split_dir, f"val_ids_{window_minutes}min.npy")
    test_path = os.path.join(split_dir, f"test_ids_{window_minutes}min.npy")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"\nLoading existing data splits from: {split_dir}")
        train_ids = np.load(train_path, allow_pickle=True)
        val_ids = np.load(val_path, allow_pickle=True)
        test_ids = np.load(test_path, allow_pickle=True)
        
        current_set = set(tree_ids)
        loaded_set = set(train_ids) | set(val_ids) | set(test_ids)
        intersect = len(current_set.intersection(loaded_set))
        
        print(f"  Loaded {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test trees")
        print(f"  {intersect} of these trees are present in the current filtered metadata.")
        
        return train_ids, val_ids, test_ids
    
    else:
        print(f"\nNo existing splits found. Creating new splits...")
        train_ids, val_ids, test_ids = split_tree_ids(tree_ids)
        
        print(f"Saving splits to: {split_dir}")
        np.save(train_path, train_ids)
        np.save(val_path, val_ids)
        np.save(test_path, test_ids)
        
        return train_ids, val_ids, test_ids

def split_tree_ids(tree_ids, random_seed=cfg.RANDOM_SEED):
    """Compute random splits"""
    train_ids, temp_ids = train_test_split(
        tree_ids,
        test_size=(cfg.VAL_RATIO + cfg.TEST_RATIO),
        random_state=random_seed,
        shuffle=True
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Data split calculation:")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/len(tree_ids)*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} ({len(val_ids)/len(tree_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} ({len(test_ids)/len(tree_ids)*100:.1f}%)")
    
    return train_ids, val_ids, test_ids


def _extract_root_score(full_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the score of each thread's root post and return a DataFrame
    with columns [tree_id, root_score].

    root_score definition:
      - Bluesky : like_count  of the post where post_type == 'root'
      - Reddit  : score       of the post where post_type == 'root'

    This is intentionally NOT a thread-level aggregate — it captures
    intrinsic content appeal before thread growth, which is semantically
    different from structural / participation targets.
    """
    dataset_type = cfg.DATASET_CONF.get('type', 'reddit')
    score_col = 'like_count' if dataset_type == 'bluesky' else 'score'

    if score_col not in full_posts.columns:
        print(f"  Warning: '{score_col}' column not found in posts; root_score will be 0.")
        return pd.DataFrame(columns=['tree_id', 'root_score'])

    if 'post_type' not in full_posts.columns:
        print("  Warning: 'post_type' column not found; root_score will be 0.")
        return pd.DataFrame(columns=['tree_id', 'root_score'])

    root_posts = (
        full_posts[full_posts['post_type'] == 'root']
        [['tree_id', score_col]]
        .rename(columns={score_col: 'root_score'})
        .drop_duplicates(subset='tree_id')
    )

    print(f"  Extracted root_score ({score_col}) for {len(root_posts):,} trees "
          f"(median={root_posts['root_score'].median():.0f}, "
          f"max={root_posts['root_score'].max():.0f})")
    return root_posts


def load_all_data(window_minutes):
    print(f"\n{'='*80}")
    print(f"LOADING DATA: {cfg.CURRENT_DATASET_NAME.upper()} - {window_minutes}MIN")
    print(f"{'='*80}")
    
    # 1. Load Early Window Data
    edges_df, meta_df = load_early_window_data(window_minutes)
    valid_tree_ids = set(meta_df['tree_id'].unique())
    print(f"Early window trees: {len(valid_tree_ids)}")

    # 2. Load Full Thread Metadata (Targets)
    print(f"Loading full metadata from {cfg.THREAD_METADATA}")
    full_metadata = pd.read_parquet(cfg.THREAD_METADATA)
    full_metadata = full_metadata[full_metadata['tree_id'].isin(valid_tree_ids)].copy()
    print(f"Filtered full metadata to {len(full_metadata)} matching trees")

    # 3. Load Full Posts (Node Features)
    print(f"Loading full posts from {cfg.THREAD_POSTS}")
    full_posts = pd.read_parquet(cfg.THREAD_POSTS)
    full_posts = full_posts[full_posts['tree_id'].isin(valid_tree_ids)].copy()
    print(f"Filtered posts to {len(full_posts)} rows for valid trees")

    # 4. Extract root_score from root posts and map into metadata.
    #    root_score = like_count (Bluesky) or score (Reddit) of the root post only.
    #    We use a direct map() instead of merge() to avoid silent failures when
    #    tree_id dtypes differ between DataFrames (e.g. str vs int).
    print(f"Extracting root_score from root posts...")
    root_scores = _extract_root_score(full_posts)
    if len(root_scores) > 0:
        # Cast both sides to the same type before building the lookup dict
        meta_id_dtype = full_metadata['tree_id'].dtype
        root_scores['tree_id'] = root_scores['tree_id'].astype(meta_id_dtype)
        score_map = root_scores.set_index('tree_id')['root_score'].to_dict()
        full_metadata['root_score'] = (
            full_metadata['tree_id'].map(score_map).fillna(0).clip(lower=0)
        )
        n_with_score = (full_metadata['root_score'] > 0).sum()
        print(f"  Mapped root_score to {n_with_score:,} / {len(full_metadata):,} trees")
    else:
        full_metadata['root_score'] = 0
        print("  Warning: no root_score extracted, defaulting to 0")

    # 5. Handle User Degrees (Optional)
    user_degrees = None
    if cfg.DATASET_CONF.get('has_user_degrees', True) and cfg.USER_DEGREES:
        print(f"Loading user degrees from {cfg.USER_DEGREES}")
        user_degrees = pd.read_csv(cfg.USER_DEGREES)
    
    # 6. Merge Features
    available_targets = [t for t in feat_cfg.OUTPUT_TARGETS if t in full_metadata.columns]
    
    missing_targets = set(feat_cfg.OUTPUT_TARGETS) - set(available_targets)
    if missing_targets:
        print(f"Warning: Missing targets in metadata: {missing_targets}")
        feat_cfg.OUTPUT_TARGETS = available_targets
        # Rebuild TARGET_GROUPS to reflect missing targets
        _rebuild_target_groups_from_available(available_targets)

    cols_to_merge = ['tree_id'] + available_targets
    if 'time_sin' in full_metadata.columns and 'time_sin' not in meta_df.columns:
        cols_to_merge.extend(['time_sin', 'time_cos'])
        
    combined_meta = meta_df.merge(
        full_metadata[cols_to_merge],
        on='tree_id',
        how='inner'
    )
    
    # 7. Merge User Degrees
    if user_degrees is not None:
        combined_meta = combined_meta.merge(
            user_degrees[['user_id', 'follower_count']],
            left_on='root_author',
            right_on='user_id',
            how='left'
        )
        combined_meta['log_follower_count'] = np.log1p(combined_meta['follower_count'].fillna(0))
        combined_meta.drop(columns=['user_id', 'follower_count'], errors='ignore', inplace=True)
    
    # 8. Clean up
    combined_meta = combined_meta.dropna(subset=available_targets)
    if 'time_sin' in combined_meta.columns:
        combined_meta = combined_meta.dropna(subset=['time_sin', 'time_cos'])
        
    tree_ids = combined_meta['tree_id'].values
    print(f"Final dataset size after merging and cleaning: {len(tree_ids)} trees")
    
    train_ids, val_ids, test_ids = get_persistent_splits(tree_ids, window_minutes)
    
    return {
        'edges': edges_df,
        'meta': combined_meta,
        'posts': full_posts,
        'user_degrees': user_degrees,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids
    }


def load_future_horizon_data(early_window_dir: str):
    """
    Load future horizon ground truth files for trajectory prediction.

    Expects files generated by generate_future_horizons.py at:
        {early_window_dir}/future_horizons/gt_{H}h.parquet   (H in 4, 8, 16, 24)
        {early_window_dir}/future_horizons/valid_mask.parquet

    Returns:
        horizon_dfs  : dict {'4h': df, '8h': df, '16h': df, '24h': df}
                       Each df is indexed by tree_id; metric columns are log1p-transformed.
        valid_mask_df: DataFrame indexed by tree_id with bool columns
                       valid_4h, valid_8h, valid_16h, valid_24h.

    Raises FileNotFoundError if the future_horizons directory is missing.
    """
    import config.feature_config as feat_cfg

    fh_dir = os.path.join(early_window_dir, 'future_horizons')
    if not os.path.isdir(fh_dir):
        raise FileNotFoundError(
            f"future_horizons directory not found at '{fh_dir}'. "
            f"Run generate_future_horizons.py first."
        )

    # Columns produced by generate_future_horizons.py
    metric_cols = list(feat_cfg.HORIZON_GT_COL_MAP.keys())
    horizon_dfs = {}

    for h in ['4h', '8h', '16h', '24h']:
        path = os.path.join(fh_dir, f'gt_{h}.parquet')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing future horizon file: {path}. "
                f"Run generate_future_horizons.py --dataset <name> first."
            )
        df = pd.read_parquet(path)
        # Apply log1p transform (same as final-state targets)
        for col in metric_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        df = df.set_index('tree_id')
        horizon_dfs[h] = df
        print(f"  [future_horizons] Loaded gt_{h}.parquet: {len(df):,} trees")

    # Validity mask
    mask_path = os.path.join(fh_dir, 'valid_mask.parquet')
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing valid_mask.parquet at {mask_path}")
    valid_mask_df = pd.read_parquet(mask_path).set_index('tree_id')
    print(f"  [future_horizons] Loaded valid_mask: {len(valid_mask_df):,} trees")

    return horizon_dfs, valid_mask_df


def _rebuild_target_groups_from_available(available_targets: list):
    """
    If some targets are missing from the metadata, prune TARGET_GROUPS
    to only include available targets, maintaining group structure.
    Called only as a fallback — in normal operation all targets are present.
    """
    import config.feature_config as feat_cfg
    available_set = set(available_targets)
    new_groups = {}
    for group_name, targets in feat_cfg.TARGET_GROUPS.items():
        pruned = [t for t in targets if t in available_set]
        if pruned:
            new_groups[group_name] = pruned
    feat_cfg.TARGET_GROUPS = new_groups
    feat_cfg.OUTPUT_TARGETS = [t for targets in new_groups.values() for t in targets]
    print(f"  Rebuilt TARGET_GROUPS: {new_groups}")