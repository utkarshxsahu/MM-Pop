"""
Load dataset inputs, targets, and split-aware sample tables.
"""

import os
import numpy as np
import pandas as pd

import config


FINAL_TARGET_COLUMNS = {
    "max_width": ["final_max_width", "max_width"],
    "max_depth": ["final_max_depth", "max_depth"],
    "structural_virality": ["final_structural_virality", "structural_virality"],
    "num_posts": ["final_num_posts", "num_posts"],
    "num_unique_users": ["final_num_unique_users", "num_unique_users"],
}

HORIZON_TARGET_COLUMNS = {
    "max_width": "gt_max_width",
    "max_depth": "gt_max_depth",
    "structural_virality": "gt_structural_virality",
    "num_posts": "gt_num_posts",
    "num_unique_users": "gt_num_unique_users",
}


def _log1p_clip(series):
    return np.log1p(series.fillna(0).clip(lower=0).astype(float))


def _resolve_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing expected columns {candidates}")
    return None


def load_early_window_data(window_tag, dataset_name=None):
    dataset_cfg = config.get_dataset_config(dataset_name)
    if window_tag == config.WINDOW_ROOT_ONLY:
        ref_window = config.get_reference_split_window(dataset_name)
        edges_path = os.path.join(dataset_cfg["early_window_dir"], f"edges_{ref_window}min.parquet")
        meta_path = os.path.join(dataset_cfg["early_window_dir"], f"meta_{ref_window}min.parquet")
    else:
        edges_path = os.path.join(dataset_cfg["early_window_dir"], f"edges_{int(window_tag)}min.parquet")
        meta_path = os.path.join(dataset_cfg["early_window_dir"], f"meta_{int(window_tag)}min.parquet")
    return pd.read_parquet(edges_path), pd.read_parquet(meta_path)


def load_full_data(dataset_name=None):
    dataset_cfg = config.get_dataset_config(dataset_name)
    metadata = pd.read_parquet(dataset_cfg["thread_metadata"])
    posts = pd.read_parquet(dataset_cfg["thread_posts"])
    user_degrees = None
    if dataset_cfg["has_user_degrees"] and dataset_cfg["user_degrees"]:
        user_degrees = pd.read_csv(dataset_cfg["user_degrees"])
    return metadata, posts, user_degrees


def extract_root_scores(posts_df, dataset_name=None):
    score_col = config.get_dataset_config(dataset_name)["root_score_column"]
    if "post_type" not in posts_df.columns:
        raise KeyError("Posts parquet is missing required column 'post_type'")
    if score_col not in posts_df.columns:
        raise KeyError(f"Posts parquet is missing root score column '{score_col}'")

    root_posts = posts_df[posts_df["post_type"] == "root"].copy()
    root_scores = (
        root_posts.groupby("tree_id")[score_col]
        .first()
        .fillna(0)
        .clip(lower=0)
        .rename("root_score")
        .reset_index()
    )
    return root_scores


def get_train_val_test_splits(window_tag, dataset_name=None):
    dataset_cfg = config.get_dataset_config(dataset_name)
    split_dir = os.path.join(dataset_cfg["early_window_dir"], "splits")
    split_window = config.get_split_window(window_tag, dataset_name)
    train_path = os.path.join(split_dir, f"train_ids_{split_window}min.npy")
    val_path = os.path.join(split_dir, f"val_ids_{split_window}min.npy")
    test_path = os.path.join(split_dir, f"test_ids_{split_window}min.npy")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Saved split files were not found for dataset={dataset_name or config.ACTIVE_DATASET}, "
            f"window={window_tag}, reference_window={split_window}"
        )

    return (
        np.load(train_path, allow_pickle=True),
        np.load(val_path, allow_pickle=True),
        np.load(test_path, allow_pickle=True),
    )


def build_final_targets(full_metadata, full_posts, dataset_name=None):
    target_df = full_metadata.copy()
    for col in ["root_score", "root_score_x", "root_score_y"]:
        if col in target_df.columns:
            target_df = target_df.drop(columns=[col])
    root_scores = extract_root_scores(full_posts, dataset_name)
    target_df = target_df.merge(root_scores, on="tree_id", how="left")

    final_targets = pd.DataFrame({"tree_id": target_df["tree_id"]})
    for target_name, candidate_cols in FINAL_TARGET_COLUMNS.items():
        source_col = _resolve_column(target_df, candidate_cols)
        final_targets[target_name] = _log1p_clip(target_df[source_col])
    root_score_col = _resolve_column(target_df, ["root_score", "root_score_y", "root_score_x"])
    final_targets["root_score"] = _log1p_clip(target_df[root_score_col])
    return final_targets.drop_duplicates(subset=["tree_id"])


def build_base_meta(window_tag, early_meta, full_metadata, user_degrees=None):
    merge_cols = ["tree_id"]
    if "time_sin" in full_metadata.columns and "time_cos" in full_metadata.columns:
        merge_cols.extend(["time_sin", "time_cos"])
    if "root_author" in full_metadata.columns:
        merge_cols.append("root_author")

    full_meta_subset = full_metadata[merge_cols].drop_duplicates(subset=["tree_id"])
    base_meta = early_meta.merge(full_meta_subset, on="tree_id", how="left", suffixes=("", "_full"))

    if user_degrees is not None and "root_author" in base_meta.columns:
        base_meta = base_meta.merge(
            user_degrees[["user_id", "follower_count"]],
            left_on="root_author",
            right_on="user_id",
            how="left",
        )
        base_meta["log_follower_count"] = np.log1p(base_meta["follower_count"].fillna(0))

    base_meta["window_tag"] = window_tag
    return base_meta.drop_duplicates(subset=["tree_id"])


def load_horizon_data(dataset_name=None):
    dataset_cfg = config.get_dataset_config(dataset_name)
    horizon_dir = os.path.join(dataset_cfg["early_window_dir"], "future_horizons")
    horizon_tables = {}
    for horizon_tag in config.HORIZON_TAGS:
        if horizon_tag == "final":
            continue
        path = os.path.join(horizon_dir, f"gt_{horizon_tag}.parquet")
        horizon_tables[horizon_tag] = pd.read_parquet(path)
    valid_mask = pd.read_parquet(os.path.join(horizon_dir, "valid_mask.parquet"))
    return horizon_tables, valid_mask


def _build_final_samples(base_meta, final_targets):
    samples = base_meta[["tree_id", "window_tag"]].copy()
    samples = samples.merge(final_targets, on="tree_id", how="inner")
    samples["horizon_tag"] = "final"
    samples["loss_mask"] = [[True, True, True, True, True, True]] * len(samples)
    samples["horizon_vec"] = [config.HORIZON_ONE_HOTS["final"]] * len(samples)
    return samples


def _build_horizon_samples(base_meta, final_targets, horizon_tables, valid_mask):
    all_samples = []
    final_samples = _build_final_samples(base_meta, final_targets)
    all_samples.append(final_samples)

    valid_mask = valid_mask.drop_duplicates(subset=["tree_id"])
    for horizon_tag, table in horizon_tables.items():
        valid_col = f"valid_{horizon_tag}"
        if valid_col not in valid_mask.columns:
            raise KeyError(f"Missing validity column '{valid_col}' in valid_mask.parquet")

        horizon_df = table.drop_duplicates(subset=["tree_id"]).copy()
        horizon_df = horizon_df.merge(valid_mask[["tree_id", valid_col]], on="tree_id", how="left")
        horizon_df = horizon_df[horizon_df[valid_col].fillna(False)]
        if horizon_df.empty:
            continue

        sample_df = base_meta[["tree_id", "window_tag"]].merge(horizon_df[["tree_id"]], on="tree_id", how="inner")
        for target_name, source_col in HORIZON_TARGET_COLUMNS.items():
            if source_col not in horizon_df.columns:
                raise KeyError(f"Missing horizon column '{source_col}' in {horizon_tag}")
            values = horizon_df.set_index("tree_id")[source_col]
            sample_df[target_name] = _log1p_clip(sample_df["tree_id"].map(values))

        sample_df["root_score"] = 0.0
        sample_df["horizon_tag"] = horizon_tag
        sample_df["loss_mask"] = [[True, True, True, True, True, False]] * len(sample_df)
        sample_df["horizon_vec"] = [config.HORIZON_ONE_HOTS[horizon_tag]] * len(sample_df)
        all_samples.append(sample_df)

    combined = pd.concat(all_samples, ignore_index=True)
    combined["horizon_index"] = combined["horizon_tag"].map(config.HORIZON_TO_INDEX)
    return combined.sort_values(["tree_id", "horizon_index"]).drop(columns=["horizon_index"])


def _restrict_samples(samples_df, split_ids):
    split_tree_ids = set(split_ids.tolist() if hasattr(split_ids, "tolist") else list(split_ids))
    return samples_df[samples_df["tree_id"].isin(split_tree_ids)].reset_index(drop=True)


def load_dataset(window_tag, dataset_name=None, mode=None):
    dataset_name = dataset_name or config.ACTIVE_DATASET
    mode = mode or config.MODE
    dataset_cfg = config.get_dataset_config(dataset_name)

    print(f"\n{'=' * 80}")
    print(f"LOADING {dataset_name.upper()} - {config.get_window_label(window_tag)} - {mode.upper()}")
    print(f"{'=' * 80}")

    edges_df, early_meta = load_early_window_data(window_tag, dataset_name)
    full_metadata, full_posts, user_degrees = load_full_data(dataset_name)

    valid_tree_ids = set(early_meta["tree_id"].unique())
    full_metadata = full_metadata[full_metadata["tree_id"].isin(valid_tree_ids)].copy()
    full_posts = full_posts[full_posts["tree_id"].isin(valid_tree_ids)].copy()

    final_targets = build_final_targets(full_metadata, full_posts, dataset_name)
    base_meta = build_base_meta(window_tag, early_meta, full_metadata, user_degrees)

    samples_df = _build_final_samples(base_meta, final_targets)
    if mode == config.MODE_FUTURE_HORIZON:
        horizon_tables, valid_mask = load_horizon_data(dataset_name)
        samples_df = _build_horizon_samples(base_meta, final_targets, horizon_tables, valid_mask)

    train_ids, val_ids, test_ids = get_train_val_test_splits(window_tag, dataset_name)
    train_samples = _restrict_samples(samples_df, train_ids)
    val_samples = _restrict_samples(samples_df, val_ids)
    test_samples = _restrict_samples(samples_df, test_ids)

    print(f"Base trees: {len(valid_tree_ids)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    return {
        "dataset_name": dataset_name,
        "dataset_config": dataset_cfg,
        "mode": mode,
        "window_tag": window_tag,
        "edges": edges_df,
        "early_meta": early_meta,
        "base_meta": base_meta,
        "targets": final_targets,
        "posts": full_posts,
        "user_degrees": user_degrees,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "target_names": list(config.TARGET_NAMES),
    }
