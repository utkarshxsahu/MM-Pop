"""
Load data for CasSeqGCN experiments.
"""

import os
import numpy as np
import pandas as pd

import config


def load_early_window_data(dataset_name, window_label, posts_df=None):
    """Load graph inputs for a standard window or construct root-only inputs."""
    spec = config.get_dataset_spec(dataset_name)

    if window_label == config.ROOT_ONLY_WINDOW:
        if posts_df is None:
            raise ValueError("posts_df is required for root-only mode")
        root_posts = _extract_root_posts(posts_df)
        edges_df = root_posts[["tree_id", "post_id"]].copy()
        edges_df["user_id"] = root_posts["user_id"] if "user_id" in root_posts.columns else pd.NA
        edges_df["reply_to"] = pd.NA
        meta_df = root_posts[["tree_id"]].drop_duplicates().reset_index(drop=True)
        return edges_df, meta_df

    window_minutes = int(window_label)
    edges_path = os.path.join(spec["early_window_dir"], f"edges_{window_minutes}min.parquet")
    meta_path = os.path.join(spec["early_window_dir"], f"meta_{window_minutes}min.parquet")
    return pd.read_parquet(edges_path), pd.read_parquet(meta_path)


def load_full_data(dataset_name):
    """Load full thread metadata and full post tables for a dataset."""
    spec = config.get_dataset_spec(dataset_name)
    metadata = pd.read_parquet(spec["metadata_path"])
    posts = pd.read_parquet(spec["posts_path"])
    return metadata, posts


def load_saved_splits(dataset_name, window_label):
    """Load saved train/val/test split ids for a dataset/window."""
    split_dir = config.get_split_dir(dataset_name)
    split_window = config.get_split_window(dataset_name, window_label)

    train_path = os.path.join(split_dir, f"train_ids_{split_window}min.npy")
    val_path = os.path.join(split_dir, f"val_ids_{split_window}min.npy")
    test_path = os.path.join(split_dir, f"test_ids_{split_window}min.npy")

    missing = [path for path in (train_path, val_path, test_path) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing}")

    return (
        np.load(train_path, allow_pickle=True),
        np.load(val_path, allow_pickle=True),
        np.load(test_path, allow_pickle=True),
    )


def load_future_horizon_labels(dataset_name):
    """Load future-horizon parquet files."""
    horizon_dir = config.get_future_horizon_dir(dataset_name)
    valid_mask = pd.read_parquet(os.path.join(horizon_dir, "valid_mask.parquet"))
    gt_frames = {}
    for horizon in config.INTERMEDIATE_HORIZONS:
        gt_frames[horizon] = pd.read_parquet(
            os.path.join(horizon_dir, f"gt_{horizon}.parquet")
        )
    return gt_frames, valid_mask


def load_dataset(dataset_name, window_label, future_horizons=False):
    """
    Load all data needed for one dataset/window experiment.
    """
    metadata_df, posts_df = load_full_data(dataset_name)
    edges_df, early_meta_df = load_early_window_data(dataset_name, window_label, posts_df=posts_df)

    valid_tree_ids = set(early_meta_df["tree_id"].unique())
    metadata_df = metadata_df[metadata_df["tree_id"].isin(valid_tree_ids)].copy()
    posts_df = posts_df[posts_df["tree_id"].isin(valid_tree_ids)].copy()
    edges_df = edges_df[edges_df["tree_id"].isin(valid_tree_ids)].copy()

    final_meta = build_final_target_table(dataset_name, metadata_df, posts_df)
    final_meta = final_meta[final_meta["tree_id"].isin(valid_tree_ids)].copy()
    final_meta = final_meta.dropna(subset=config.TARGET_NAMES).reset_index(drop=True)

    train_ids, val_ids, test_ids = load_saved_splits(dataset_name, window_label)
    available_tree_ids = set(final_meta["tree_id"].unique()) & set(edges_df["tree_id"].unique())
    train_ids = _filter_split_ids(train_ids, available_tree_ids)
    val_ids = _filter_split_ids(val_ids, available_tree_ids)
    test_ids = _filter_split_ids(test_ids, available_tree_ids)

    gt_frames = None
    valid_mask = None
    if future_horizons:
        gt_frames, valid_mask = load_future_horizon_labels(dataset_name)

    return {
        "dataset_name": dataset_name,
        "window_label": window_label,
        "edges": edges_df,
        "meta": final_meta,
        "posts": posts_df,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "target_names": list(config.TARGET_NAMES),
        "future_horizon_gt": gt_frames,
        "future_horizon_valid_mask": valid_mask,
    }


def build_final_target_table(dataset_name, metadata_df, posts_df):
    """Assemble the 6 final targets in raw space."""
    required = [
        "tree_id",
        "max_width",
        "max_depth",
        "structural_virality",
        "num_posts",
        "num_unique_users",
    ]
    final_meta = metadata_df[required].copy()

    root_scores = build_root_score_table(dataset_name, posts_df)
    final_meta = final_meta.merge(root_scores, on="tree_id", how="left")
    final_meta["root_score"] = final_meta["root_score"].fillna(0)
    return final_meta


def build_root_score_table(dataset_name, posts_df):
    """Extract root engagement per tree in raw space."""
    spec = config.get_dataset_spec(dataset_name)
    score_column = spec["root_score_column"]

    root_posts = _extract_root_posts(posts_df)
    root_scores = (
        root_posts[["tree_id", score_column]]
        .drop_duplicates(subset=["tree_id"])
        .rename(columns={score_column: "root_score"})
    )
    root_scores["root_score"] = root_scores["root_score"].clip(lower=0).fillna(0)
    return root_scores


def _extract_root_posts(posts_df):
    if "post_type" not in posts_df.columns:
        raise KeyError("Posts parquet is missing required column: post_type")
    root_posts = posts_df[posts_df["post_type"] == "root"].copy()
    if root_posts.empty:
        raise ValueError("No root posts found in posts table")
    return root_posts


def _filter_split_ids(tree_ids, available_tree_ids):
    return np.array([tree_id for tree_id in tree_ids if tree_id in available_tree_ids], dtype=object)
