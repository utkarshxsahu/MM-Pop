"""
Load data from parquet files.
"""

import os
import numpy as np
import pandas as pd

import config


FINAL_TARGET_MASK = np.array([True, True, True, True, True, True], dtype=bool)
INTERMEDIATE_TARGET_MASK = np.array([True, True, True, True, True, False], dtype=bool)


def _get_reference_window(window_minutes):
    if window_minutes != 0:
        return window_minutes
    return config.ROOT_ONLY_REFERENCE_WINDOW


def _split_paths(window_minutes):
    split_window = _get_reference_window(window_minutes)
    if split_window is None:
        raise ValueError(
            f"Dataset '{config.CURRENT_DATASET}' uses root-only window 0 but has no "
            "reference window configured."
        )

    train_path = os.path.join(config.SPLIT_DIR, f"train_ids_{split_window}min.npy")
    val_path = os.path.join(config.SPLIT_DIR, f"val_ids_{split_window}min.npy")
    test_path = os.path.join(config.SPLIT_DIR, f"test_ids_{split_window}min.npy")
    return train_path, val_path, test_path


def load_early_window_data(window_minutes):
    """Load early-window graph inputs for a specific time window."""
    if window_minutes == 0:
        reference_window = _get_reference_window(window_minutes)
        edges_path = os.path.join(config.EARLY_WINDOW_DIR, f"edges_{reference_window}min.parquet")
        meta_path = os.path.join(config.EARLY_WINDOW_DIR, f"meta_{reference_window}min.parquet")

        reference_edges = pd.read_parquet(edges_path)
        reference_meta = pd.read_parquet(meta_path)

        root_only_edges = reference_edges.iloc[0:0].copy()
        return root_only_edges, reference_meta

    edges_path = os.path.join(config.EARLY_WINDOW_DIR, f"edges_{window_minutes}min.parquet")
    meta_path = os.path.join(config.EARLY_WINDOW_DIR, f"meta_{window_minutes}min.parquet")

    edges_df = pd.read_parquet(edges_path)
    meta_df = pd.read_parquet(meta_path)
    return edges_df, meta_df


def load_full_data():
    """Load full thread metadata and posts."""
    metadata = pd.read_parquet(config.THREAD_METADATA)
    posts = pd.read_parquet(config.THREAD_POSTS)

    user_degrees = None
    if config.HAS_USER_DEGREES and config.USER_DEGREES:
        user_degrees = pd.read_csv(config.USER_DEGREES)

    return metadata, posts, user_degrees


def _build_root_scores(posts_df):
    score_col = config.ROOT_SCORE_COLUMN
    root_posts = posts_df[posts_df["post_type"] == "root"].copy()
    if root_posts.empty and "depth" in posts_df.columns:
        root_posts = posts_df[posts_df["depth"] == 0].copy()

    if root_posts.empty:
        raise ValueError(
            f"Could not find root posts for dataset '{config.CURRENT_DATASET}'."
        )

    root_posts[score_col] = root_posts[score_col].clip(lower=0).fillna(0)
    root_scores = (
        root_posts.groupby("tree_id", as_index=False)[score_col]
        .first()
        .rename(columns={score_col: "root_score"})
    )
    return root_scores


def _merge_targets(meta_df, full_metadata, full_posts, user_degrees):
    target_columns = [
        "max_width",
        "max_depth",
        "structural_virality",
        "num_posts",
        "num_unique_users",
    ]
    missing_targets = [col for col in target_columns if col not in full_metadata.columns]
    if missing_targets:
        raise ValueError(
            f"Missing target columns in metadata for {config.CURRENT_DATASET}: {missing_targets}"
        )

    root_scores = _build_root_scores(full_posts)

    cols_to_merge = ["tree_id"] + target_columns
    optional_cols = [col for col in ["time_sin", "time_cos", "root_author"] if col in full_metadata.columns]
    cols_to_merge.extend(optional_cols)

    overlap_cols = [col for col in cols_to_merge if col != "tree_id" and col in meta_df.columns]
    if overlap_cols:
        meta_df = meta_df.drop(columns=overlap_cols)

    combined_meta = meta_df.merge(full_metadata[cols_to_merge], on="tree_id", how="inner")
    combined_meta = combined_meta.merge(root_scores, on="tree_id", how="left")

    if user_degrees is not None and "root_author" in combined_meta.columns:
        combined_meta = combined_meta.merge(
            user_degrees[["user_id", "follower_count"]],
            left_on="root_author",
            right_on="user_id",
            how="left",
        )
        combined_meta["log_follower_count"] = np.log1p(combined_meta["follower_count"].fillna(0))

    combined_meta["root_score"] = combined_meta["root_score"].fillna(0).clip(lower=0)
    combined_meta = combined_meta.dropna(subset=config.OUTPUT_TARGETS)
    if "time_sin" in combined_meta.columns:
        combined_meta = combined_meta.dropna(subset=["time_sin", "time_cos"])

    return combined_meta


def get_train_val_test_splits(window_minutes):
    """Load precomputed train/val/test splits."""
    train_path, val_path, test_path = _split_paths(window_minutes)
    missing = [path for path in [train_path, val_path, test_path] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing split files for dataset '{config.CURRENT_DATASET}', window {window_minutes}: {missing}"
        )

    train_ids = np.load(train_path, allow_pickle=True)
    val_ids = np.load(val_path, allow_pickle=True)
    test_ids = np.load(test_path, allow_pickle=True)
    return train_ids, val_ids, test_ids


def _load_horizon_tables():
    horizon_tables = {}
    valid_mask_path = os.path.join(config.FUTURE_HORIZON_DIR, "valid_mask.parquet")
    valid_mask = pd.read_parquet(valid_mask_path)

    for tag in config.HORIZON_TAGS:
        if tag == "final":
            continue
        horizon_tables[tag] = pd.read_parquet(
            os.path.join(config.FUTURE_HORIZON_DIR, f"gt_{tag}.parquet")
        )

    return horizon_tables, valid_mask


def _build_final_target_map(meta_df):
    final_targets = {}
    for row in meta_df[["tree_id"] + config.OUTPUT_TARGETS].itertuples(index=False):
        row_dict = row._asdict()
        final_targets[row_dict["tree_id"]] = np.array(
            [
                np.log1p(max(row_dict[target], 0))
                for target in config.OUTPUT_TARGETS
            ],
            dtype=np.float32,
        )
    return final_targets


def _build_sample_specs(tree_ids, final_target_map):
    sample_specs = {}
    for tree_id in tree_ids:
        sample_specs[tree_id] = [
            {
                "tree_id": tree_id,
                "horizon_tag": "final",
                "horizon_index": config.HORIZON_TO_INDEX["final"],
                "targets": final_target_map[tree_id],
                "target_mask": FINAL_TARGET_MASK.copy(),
            }
        ]
    return sample_specs


def _augment_with_future_horizons(sample_specs):
    horizon_tables, valid_mask = _load_horizon_tables()
    valid_mask = valid_mask.set_index("tree_id")

    for tag, horizon_df in horizon_tables.items():
        horizon_df = horizon_df.set_index("tree_id")
        valid_col = f"valid_{tag}"

        for tree_id, specs in sample_specs.items():
            if tree_id not in valid_mask.index:
                continue
            if not bool(valid_mask.at[tree_id, valid_col]):
                continue
            if tree_id not in horizon_df.index:
                continue

            row = horizon_df.loc[tree_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            targets = np.array(
                [
                    np.log1p(max(row["gt_max_width"], 0)),
                    np.log1p(max(row["gt_max_depth"], 0)),
                    np.log1p(max(row["gt_structural_virality"], 0)),
                    np.log1p(max(row["gt_num_posts"], 0)),
                    np.log1p(max(row["gt_num_unique_users"], 0)),
                    0.0,
                ],
                dtype=np.float32,
            )
            specs.insert(
                -1,
                {
                    "tree_id": tree_id,
                    "horizon_tag": tag,
                    "horizon_index": config.HORIZON_TO_INDEX[tag],
                    "targets": targets,
                    "target_mask": INTERMEDIATE_TARGET_MASK.copy(),
                },
            )

    return sample_specs


def load_dataset(window_minutes):
    """
    Load complete dataset for a specific input window.
    """
    print(f"\n{'=' * 80}")
    label = "0MIN (ROOT-ONLY)" if window_minutes == 0 else f"{window_minutes}MIN"
    print(f"LOADING {config.CURRENT_DATASET.upper()} - {label}")
    print(f"{'=' * 80}")

    edges_df, meta_df = load_early_window_data(window_minutes)
    print(f"Early window trees: {len(meta_df)}")

    full_metadata, full_posts, user_degrees = load_full_data()

    valid_tree_ids = set(meta_df["tree_id"].unique())
    full_metadata = full_metadata[full_metadata["tree_id"].isin(valid_tree_ids)].copy()
    full_posts = full_posts[full_posts["tree_id"].isin(valid_tree_ids)].copy()
    print(f"Full metadata trees: {len(full_metadata)}")

    combined_meta = _merge_targets(meta_df, full_metadata, full_posts, user_degrees)
    tree_ids = combined_meta["tree_id"].values
    print(f"Final dataset size: {len(tree_ids)} trees")

    train_ids, val_ids, test_ids = get_train_val_test_splits(window_minutes)

    dataset_tree_ids = set(tree_ids.tolist())
    train_ids = np.array([tree_id for tree_id in train_ids if tree_id in dataset_tree_ids], dtype=object)
    val_ids = np.array([tree_id for tree_id in val_ids if tree_id in dataset_tree_ids], dtype=object)
    test_ids = np.array([tree_id for tree_id in test_ids if tree_id in dataset_tree_ids], dtype=object)

    if window_minutes == 0:
        root_posts = full_posts[full_posts["tree_id"].isin(dataset_tree_ids)].copy()
        if "post_type" in root_posts.columns:
            root_posts = root_posts[root_posts["post_type"] == "root"].copy()
        elif "depth" in root_posts.columns:
            root_posts = root_posts[root_posts["depth"] == 0].copy()
        posts_for_inputs = root_posts
    else:
        posts_for_inputs = full_posts

    final_target_map = _build_final_target_map(combined_meta)
    sample_specs = _build_sample_specs(tree_ids, final_target_map)
    if config.USE_FUTURE_HORIZONS:
        sample_specs = _augment_with_future_horizons(sample_specs)

    return {
        "edges": edges_df,
        "meta": combined_meta,
        "posts": posts_for_inputs,
        "user_degrees": user_degrees,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "available_targets": config.OUTPUT_TARGETS.copy(),
        "sample_specs": sample_specs,
        "window_minutes": window_minutes,
    }
