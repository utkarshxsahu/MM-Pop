#!/usr/bin/env python3
"""
Generate future horizon ground truth snapshots for trajectory prediction.

These are NOT early windows (model inputs). These are ground truth label files
only — structural metrics computed at fixed absolute time horizons from the
root post creation time. Used as supervision signal for trajectory prediction.

Output per dataset, per horizon:
    {early_window_dir}/future_horizons/gt_{H}h.parquet
    Columns: tree_id, gt_num_posts, gt_max_depth, gt_max_width,
             gt_structural_virality, gt_num_unique_users

Also outputs a validity summary:
    {early_window_dir}/future_horizons/valid_mask.parquet
    Columns: tree_id, valid_4h, valid_8h, valid_16h, valid_24h

Usage:
    python generate_future_horizons.py --dataset bluesky
    python generate_future_horizons.py --dataset gaming
    python generate_future_horizons.py --dataset futurology
    python generate_future_horizons.py --dataset ama
    python generate_future_horizons.py --dataset all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

# ============================================================
# DATASET CONFIGURATION
# Mirrors dataset_config.py — update paths here if they change
# ============================================================

BLUESKY_ROOT  = "/scratch/dgl/Social_Network/bluesky/processed"
REDDIT_ROOT   = "/scratch/dgl/Social_Network/reddit/filtered/tree_full"

DATASETS = {
    'bluesky': {
        'type': 'bluesky',
        'posts_file': os.path.join(
            BLUESKY_ROOT,
            "discussion_trees/samples/tree_65k/thread_posts_with_all_labels2.parquet"
        ),
        'early_window_dir': os.path.join(
            BLUESKY_ROOT,
            "discussion_trees/samples/tree_65k/gnn_snapshots"
        ),
        # Windows that have existing splits on disk
        'split_windows': [2, 10, 30],
        # Columns to load from posts file
        'time_col': 'timestamp',        # datetime column
        'time_mode': 'datetime',        # compute time_since_root from timestamps
        'post_id_col': 'post_id',
        'user_id_col': 'user_id',
        'reply_to_col': 'reply_to',
        'depth_col': 'depth',
        'post_type_col': 'post_type',
    },
    'gaming': {
        'type': 'reddit',
        'posts_file': os.path.join(REDDIT_ROOT, "reddit_gaming_posts.parquet"),
        'early_window_dir': os.path.join(REDDIT_ROOT, "snapshots/gaming"),
        'split_windows': [20, 50, 90],
        'time_col': 'time_since_root',  # already in minutes
        'time_mode': 'minutes',
        'post_id_col': 'post_id',
        'user_id_col': 'user_id',
        'reply_to_col': 'reply_to',
        'depth_col': 'depth',
        'post_type_col': 'post_type',
    },
    'futurology': {
        'type': 'reddit',
        'posts_file': os.path.join(REDDIT_ROOT, "reddit_futurology_posts.parquet"),
        'early_window_dir': os.path.join(REDDIT_ROOT, "snapshots/futurology"),
        'split_windows': [30, 90, 180],
        'time_col': 'time_since_root',
        'time_mode': 'minutes',
        'post_id_col': 'post_id',
        'user_id_col': 'user_id',
        'reply_to_col': 'reply_to',
        'depth_col': 'depth',
        'post_type_col': 'post_type',
    },
    'ama': {
        'type': 'reddit',
        'posts_file': os.path.join(REDDIT_ROOT, "reddit_ama_posts.parquet"),
        'early_window_dir': os.path.join(REDDIT_ROOT, "snapshots/ama"),
        'split_windows': [15, 30, 60],
        'time_col': 'time_since_root',
        'time_mode': 'minutes',
        'post_id_col': 'post_id',
        'user_id_col': 'user_id',
        'reply_to_col': 'reply_to',
        'depth_col': 'depth',
        'post_type_col': 'post_type',
    },
}

# Future horizons in minutes
HORIZONS = {
    '4h':  4  * 60,   # 240 min
    '8h':  8  * 60,   # 480 min
    '16h': 16 * 60,   # 960 min
    '24h': 24 * 60,   # 1440 min
}

# Number of parallel workers for structural virality BFS
N_WORKERS = 9


# ============================================================
# SPLIT LOADING — union across all windows
# ============================================================

def load_split_union(early_window_dir: str, split_windows: list) -> set:
    """
    Load all split .npy files for all windows and return the union of tree IDs.
    This ensures we compute horizon labels for every tree that appeared in
    any training/val/test split, regardless of which early window it came from.
    """
    split_dir = os.path.join(early_window_dir, "splits")
    all_ids = set()
    found_any = False

    for win in split_windows:
        for split_name in ['train', 'val', 'test']:
            path = os.path.join(split_dir, f"{split_name}_ids_{win}min.npy")
            if os.path.exists(path):
                ids = np.load(path, allow_pickle=True)
                all_ids.update(ids.tolist())
                found_any = True
            else:
                print(f"  [WARN] Split file not found: {path}")

    if not found_any:
        raise FileNotFoundError(
            f"No split files found in {split_dir}. "
            f"Run the main experiment at least once to generate splits."
        )

    print(f"  Split union: {len(all_ids):,} unique tree IDs across windows {split_windows}")
    return all_ids


# ============================================================
# TIME NORMALISATION
# For Bluesky: compute time_since_root_min from datetime timestamps.
# For Reddit:  time_since_root column is already in minutes — use directly.
# ============================================================

def add_time_since_root_minutes(posts_df: pd.DataFrame, ds_cfg: dict) -> pd.DataFrame:
    """
    Returns posts_df with a new column 'time_since_root_min' (float, minutes).
    Works for both datetime-based (Bluesky) and precomputed (Reddit) datasets.
    """
    if ds_cfg['time_mode'] == 'minutes':
        # Reddit: column already exists in minutes
        posts_df = posts_df.copy()
        posts_df['time_since_root_min'] = posts_df[ds_cfg['time_col']].astype(float)
        return posts_df

    # Bluesky: derive from timestamps
    posts_df = posts_df.copy()
    posts_df[ds_cfg['time_col']] = pd.to_datetime(posts_df[ds_cfg['time_col']])

    # Find root post timestamp per tree (post_type == 'root')
    root_times = (
        posts_df[posts_df[ds_cfg['post_type_col']] == 'root']
        [['tree_id', ds_cfg['time_col']]]
        .rename(columns={ds_cfg['time_col']: 'root_time'})
        .drop_duplicates(subset='tree_id')
    )

    posts_df = posts_df.merge(root_times, on='tree_id', how='left')
    posts_df['time_since_root_min'] = (
        (posts_df[ds_cfg['time_col']] - posts_df['root_time'])
        .dt.total_seconds() / 60.0
    )
    posts_df = posts_df.drop(columns=['root_time'])
    return posts_df


# ============================================================
# STRUCTURAL VIRALITY (BFS Wiener Index) — per tree
# ============================================================

def _structural_virality_bfs(posts_subset: pd.DataFrame) -> float:
    """
    Compute normalised structural virality (Wiener Index) for a set of posts.
    O(n^2) BFS — parallelised at the tree level by the caller.
    """
    n = len(posts_subset)
    if n <= 1:
        return 0.0

    # Build undirected adjacency from reply graph
    adjacency = defaultdict(list)
    valid_ids = set(posts_subset['post_id'])

    for _, row in posts_subset.iterrows():
        pid   = row['post_id']
        r_to  = row['reply_to']
        if pd.notna(r_to) and r_to in valid_ids:
            adjacency[pid].append(r_to)
            adjacency[r_to].append(pid)

    post_ids = posts_subset['post_id'].tolist()
    total_distance = 0

    for start in post_ids:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            d    = distances[curr]
            for nb in adjacency[curr]:
                if nb not in distances:
                    distances[nb] = d + 1
                    queue.append(nb)
        for other in post_ids:
            if other != start and other in distances:
                total_distance += distances[other]

    wiener_index = total_distance // 2
    return float((2.0 * wiener_index) / (n * (n - 1)))


# ============================================================
# PER-TREE PROCESSING — one horizon at a time
# Called inside worker processes.
# ============================================================

def _process_tree(args):
    """
    Worker function. Receives a tuple to avoid pickle issues with lambdas.
    Returns a dict row or None if the tree is not valid for this horizon.
    """
    tree_id, group_data, horizon_min = args

    # Validity check: tree must have lived longer than the horizon
    tree_lifetime = group_data['time_since_root_min'].max()
    if tree_lifetime <= horizon_min:
        return None  # tree was already dead at this horizon — no label

    # Filter to posts within the horizon window
    snapshot = group_data[group_data['time_since_root_min'] <= horizon_min]

    if len(snapshot) == 0:
        return None

    # Structural metrics
    n               = len(snapshot)
    max_depth       = int(snapshot['depth'].max())
    depth_counts    = snapshot['depth'].value_counts()
    max_width       = int(depth_counts.max()) if len(depth_counts) > 0 else 1
    num_unique_users = int(snapshot['user_id'].nunique())
    virality        = _structural_virality_bfs(snapshot)

    return {
        'tree_id':                tree_id,
        'gt_num_posts':           n,
        'gt_max_depth':           max_depth,
        'gt_max_width':           max_width,
        'gt_num_unique_users':    num_unique_users,
        'gt_structural_virality': virality,
    }


# ============================================================
# HORIZON PROCESSING — one full horizon for a dataset
# ============================================================

def process_horizon(
    posts_df:    pd.DataFrame,
    tree_ids:    set,
    horizon_min: int,
    horizon_tag: str,
    n_workers:   int,
) -> pd.DataFrame:
    """
    Process all trees for a single future horizon.
    Parallelises BFS across trees using ProcessPoolExecutor.

    Returns a DataFrame with gt_* columns for all valid trees.
    """
    # Pre-filter posts to only the trees we care about
    subset = posts_df[posts_df['tree_id'].isin(tree_ids)]

    # Build list of (tree_id, group_df, horizon_min) tasks
    tasks = [
        (tid, grp.reset_index(drop=True), horizon_min)
        for tid, grp in subset.groupby('tree_id')
    ]

    total   = len(tasks)
    results = []

    print(f"    Processing {total:,} trees with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_tree, task): task[0] for task in tasks}

        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 5000 == 0 or done == total:
                print(f"      {done:,}/{total:,} trees done", flush=True)
            row = future.result()
            if row is not None:
                results.append(row)

    df = pd.DataFrame(results)
    valid_count   = len(df)
    invalid_count = total - valid_count
    survival_rate = (valid_count / total * 100) if total > 0 else 0.0

    print(f"    ✓ {horizon_tag}: {valid_count:,} valid trees "
          f"({survival_rate:.1f}% survival, "
          f"{invalid_count:,} trees dead before this horizon)")

    return df


# ============================================================
# VALID MASK SUMMARY
# ============================================================

def build_valid_mask(horizon_dfs: dict, all_tree_ids: set) -> pd.DataFrame:
    """
    Build a boolean mask table:
        tree_id | valid_4h | valid_8h | valid_16h | valid_24h

    A tree is valid for horizon H if it appears in that horizon's output df
    (i.e. it survived the lifetime check and had at least one post in window).
    """
    mask_df = pd.DataFrame({'tree_id': sorted(all_tree_ids)})

    for tag, df in horizon_dfs.items():
        col = f"valid_{tag}"
        valid_set = set(df['tree_id'].tolist()) if len(df) > 0 else set()
        mask_df[col] = mask_df['tree_id'].isin(valid_set)

    return mask_df


# ============================================================
# MAIN DATASET RUNNER
# ============================================================

def run_dataset(dataset_name: str, n_workers: int = N_WORKERS):
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASETS.keys())}"
        )

    ds_cfg = DATASETS[dataset_name]

    # ------------------------------------------------------------------
    # 1. Load split union — these are the trees we need labels for
    # ------------------------------------------------------------------
    print("\n[1/4] Loading split tree IDs...")
    tree_ids = load_split_union(ds_cfg['early_window_dir'], ds_cfg['split_windows'])

    # ------------------------------------------------------------------
    # 2. Load posts — only columns needed for metric computation
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading posts from:\n      {ds_cfg['posts_file']}")

    needed_cols = [
        'tree_id',
        ds_cfg['post_id_col'],
        ds_cfg['user_id_col'],
        ds_cfg['reply_to_col'],
        ds_cfg['depth_col'],
        ds_cfg['time_col'],
    ]
    # Bluesky also needs post_type to find root timestamps
    if ds_cfg['time_mode'] == 'datetime':
        needed_cols.append(ds_cfg['post_type_col'])

    posts_df = pd.read_parquet(ds_cfg['posts_file'], columns=needed_cols)

    # Standardise column names so downstream code is uniform
    posts_df = posts_df.rename(columns={
        ds_cfg['post_id_col']:  'post_id',
        ds_cfg['user_id_col']:  'user_id',
        ds_cfg['reply_to_col']: 'reply_to',
        ds_cfg['depth_col']:    'depth',
    })
    if ds_cfg['time_mode'] == 'datetime':
        posts_df = posts_df.rename(columns={ds_cfg['post_type_col']: 'post_type'})

    print(f"  Loaded {len(posts_df):,} posts from "
          f"{posts_df['tree_id'].nunique():,} trees total")

    # Add time_since_root_min column (works for both bluesky and reddit)
    posts_df = add_time_since_root_minutes(posts_df, ds_cfg)

    # Filter to only split trees
    posts_df = posts_df[posts_df['tree_id'].isin(tree_ids)].copy()
    print(f"  After filtering to split trees: "
          f"{len(posts_df):,} posts, {posts_df['tree_id'].nunique():,} trees")

    # ------------------------------------------------------------------
    # 3. Process each horizon
    # ------------------------------------------------------------------
    print(f"\n[3/4] Processing {len(HORIZONS)} horizons with {n_workers} workers...")

    output_dir = os.path.join(ds_cfg['early_window_dir'], "future_horizons")
    os.makedirs(output_dir, exist_ok=True)

    horizon_dfs = {}

    for tag, horizon_min in HORIZONS.items():
        print(f"\n  -- Horizon {tag} ({horizon_min} min) --")

        df = process_horizon(
            posts_df    = posts_df,
            tree_ids    = tree_ids,
            horizon_min = horizon_min,
            horizon_tag = tag,
            n_workers   = n_workers,
        )

        # Save per-horizon file
        out_path = os.path.join(output_dir, f"gt_{tag}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"    Saved → {out_path}")

        horizon_dfs[tag] = df

    # ------------------------------------------------------------------
    # 4. Save valid mask summary
    # ------------------------------------------------------------------
    print(f"\n[4/4] Building validity mask...")

    mask_df   = build_valid_mask(horizon_dfs, tree_ids)
    mask_path = os.path.join(output_dir, "valid_mask.parquet")
    mask_df.to_parquet(mask_path, index=False)

    # Print summary table
    print(f"\n  Validity summary for {dataset_name.upper()}:")
    print(f"  {'Horizon':<10} {'Valid Trees':>12} {'Coverage':>10}")
    print(f"  {'-'*35}")
    for tag in HORIZONS:
        col   = f"valid_{tag}"
        n_val = int(mask_df[col].sum())
        pct   = n_val / len(mask_df) * 100
        print(f"  {tag:<10} {n_val:>12,} {pct:>9.1f}%")

    print(f"\n  Valid mask saved → {mask_path}")
    print(f"\n  All outputs in: {output_dir}")
    print(f"  Files:")
    for tag in HORIZONS:
        print(f"    gt_{tag}.parquet")
    print(f"    valid_mask.parquet")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate future horizon ground truth snapshots."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASETS.keys()) + ['all'],
        help="Dataset to process. Use 'all' to run all datasets sequentially."
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=N_WORKERS,
        help=f"Number of parallel workers for BFS (default: {N_WORKERS})"
    )
    args = parser.parse_args()

    if args.dataset == 'all':
        for name in DATASETS:
            run_dataset(name, n_workers=args.workers)
    else:
        run_dataset(args.dataset, n_workers=args.workers)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()