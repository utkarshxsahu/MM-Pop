from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


FINAL_TARGET_COLUMNS = [
    "max_width",
    "max_depth",
    "structural_virality",
    "num_posts",
    "num_unique_users",
]
HORIZON_GT_COLUMNS = {
    "4h": [
        "gt_max_width",
        "gt_max_depth",
        "gt_structural_virality",
        "gt_num_posts",
        "gt_num_unique_users",
    ],
    "8h": [
        "gt_max_width",
        "gt_max_depth",
        "gt_structural_virality",
        "gt_num_posts",
        "gt_num_unique_users",
    ],
    "16h": [
        "gt_max_width",
        "gt_max_depth",
        "gt_structural_virality",
        "gt_num_posts",
        "gt_num_unique_users",
    ],
    "24h": [
        "gt_max_width",
        "gt_max_depth",
        "gt_structural_virality",
        "gt_num_posts",
        "gt_num_unique_users",
    ],
}


@dataclass
class PathsExample:
    tree_id: int
    paths: List[List[int]]
    bin_ids: List[int]
    y: np.ndarray
    horizon_mask: np.ndarray
    horizon_idx: int
    horizon_label: str


def _safe_id_convert(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        return series.apply(lambda x: hash(str(x)) % (2**31)).astype(np.int64)
    return series.astype(np.int64)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _normalize_split_ids(array: np.ndarray) -> np.ndarray:
    series = pd.Series(array)
    return _safe_id_convert(series).to_numpy(dtype=np.int64)


def load_npy_splits(early_window_dir: str, split_window_minutes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_dir = Path(early_window_dir) / "splits"
    train_path = split_dir / f"train_ids_{split_window_minutes}min.npy"
    val_path = split_dir / f"val_ids_{split_window_minutes}min.npy"
    test_path = split_dir / f"test_ids_{split_window_minutes}min.npy"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Split files not found in {split_dir} for {split_window_minutes}min."
        )

    train_ids = _normalize_split_ids(np.load(train_path, allow_pickle=True))
    val_ids = _normalize_split_ids(np.load(val_path, allow_pickle=True))
    test_ids = _normalize_split_ids(np.load(test_path, allow_pickle=True))
    return train_ids, val_ids, test_ids


def build_user_vocab(edges_df: pd.DataFrame, meta_df: pd.DataFrame) -> Tuple[Dict[int, int], int]:
    users = set(edges_df["user_id"].dropna().astype(np.int64).tolist())
    users.update(meta_df["root_author"].dropna().astype(np.int64).tolist())

    user2idx: Dict[int, int] = {}
    next_idx = 1
    for user_id in sorted(users):
        user2idx[int(user_id)] = next_idx
        next_idx += 1
    return user2idx, 0


def compute_bin_id(t_minutes: float, T: float, num_bins: int) -> int:
    if T <= 0:
        return 0
    recency = max(0.0, min(T, T - max(0.0, t_minutes)))
    bin_width = T / float(num_bins)
    b = int(recency / bin_width) if bin_width > 0 else 0
    return int(min(num_bins - 1, max(0, b)))


def build_paths_for_tree(
    tree_edges: pd.DataFrame,
    tree_posts_ts: pd.DataFrame,
    root_post_id: int,
    root_author: int,
    root_ts: pd.Timestamp,
    user2idx: Dict[int, int],
    unk_idx: int,
    observation_window: int,
    num_bins: int,
    include_root_path: bool = True,
) -> Tuple[List[List[int]], List[int]]:
    post_to_parent: Dict[int, Optional[int]] = {}
    post_to_user: Dict[int, int] = {}
    for row in tree_edges.itertuples(index=False):
        pid = int(row.post_id)
        post_to_user[pid] = int(row.user_id)
        post_to_parent[pid] = None if pd.isna(row.reply_to) else int(row.reply_to)

    ts_map: Dict[int, pd.Timestamp] = {}
    for row in tree_posts_ts.itertuples(index=False):
        ts_map[int(row.post_id)] = row.timestamp

    root_user_idx = user2idx.get(int(root_author), unk_idx)
    post_ids = list(post_to_user.keys())
    if include_root_path and int(root_post_id) not in post_to_user:
        post_ids.append(int(root_post_id))

    paths: List[List[int]] = []
    bin_ids: List[int] = []
    for pid in post_ids:
        if pid == int(root_post_id) and include_root_path:
            t_minutes = 0.0
            leaf_user = int(root_author)
        else:
            leaf_user = post_to_user.get(pid, int(root_author))
            leaf_ts = ts_map.get(pid, pd.NaT)
            if pd.isna(leaf_ts) or pd.isna(root_ts):
                t_minutes = 0.0
            else:
                t_minutes = float((leaf_ts - root_ts).total_seconds() / 60.0)
                t_minutes = max(0.0, min(float(observation_window), t_minutes))

        bin_ids.append(compute_bin_id(t_minutes=t_minutes, T=float(observation_window), num_bins=num_bins))

        if pid == int(root_post_id) and include_root_path:
            chain_posts = [int(root_post_id)]
        else:
            chain_posts = []
            cur = pid
            visited = set()
            while True:
                if cur in visited:
                    break
                visited.add(cur)
                chain_posts.append(cur)
                parent = post_to_parent.get(cur, None)
                if parent is None:
                    break
                if parent == int(root_post_id):
                    chain_posts.append(int(root_post_id))
                    break
                cur = parent
            chain_posts = list(reversed(chain_posts))

        user_seq = [root_user_idx]
        for chain_post in chain_posts:
            if chain_post == int(root_post_id):
                continue
            uid = post_to_user.get(chain_post, leaf_user)
            user_seq.append(user2idx.get(int(uid), unk_idx))

        paths.append(user_seq)
    return paths, bin_ids


def build_root_only_paths(root_author: int, user2idx: Dict[int, int], unk_idx: int) -> Tuple[List[List[int]], List[int]]:
    root_user_idx = user2idx.get(int(root_author), unk_idx)
    return [[root_user_idx]], [0]


def _final_target_vector(row: pd.Series, root_score_lookup: Dict[int, float]) -> np.ndarray:
    values = [float(row[col]) for col in FINAL_TARGET_COLUMNS]
    values.append(float(root_score_lookup.get(int(row["tree_id"]), 0.0)))
    arr = np.array(values, dtype=np.float32)
    return np.log1p(np.maximum(arr, 0.0)).astype(np.float32)


def _load_root_scores(posts_path: str, score_column: str) -> Dict[int, float]:
    cols = ["tree_id", "post_type", score_column]
    posts_df = pd.read_parquet(posts_path, columns=cols)
    posts_df = posts_df.dropna(subset=["tree_id", "post_type", score_column])
    posts_df["tree_id"] = _safe_id_convert(posts_df["tree_id"])
    root_df = posts_df[posts_df["post_type"] == "root"].copy()
    root_df[score_column] = np.maximum(root_df[score_column].astype(float), 0.0)
    grouped = root_df.groupby("tree_id", sort=False)[score_column].first()
    return {int(tree_id): float(value) for tree_id, value in grouped.items()}


def _load_future_horizon_tables(future_horizons_dir: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    gt_tables: Dict[str, pd.DataFrame] = {}
    for horizon in ("4h", "8h", "16h", "24h"):
        path = Path(future_horizons_dir) / f"gt_{horizon}.parquet"
        cols = ["tree_id"] + HORIZON_GT_COLUMNS[horizon]
        df = pd.read_parquet(path, columns=cols)
        df["tree_id"] = _safe_id_convert(df["tree_id"])
        for col in HORIZON_GT_COLUMNS[horizon]:
            df[col] = np.log1p(np.maximum(df[col].astype(np.float32), 0.0)).astype(np.float32)
        gt_tables[horizon] = df.set_index("tree_id")

    mask_path = Path(future_horizons_dir) / "valid_mask.parquet"
    mask_df = pd.read_parquet(mask_path, columns=["tree_id", "valid_4h", "valid_8h", "valid_16h", "valid_24h"])
    mask_df["tree_id"] = _safe_id_convert(mask_df["tree_id"])
    return gt_tables, mask_df.set_index("tree_id")


class DeepHawkesPathsDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        posts_path: str,
        edges_path: Optional[str],
        window_minutes: int,
        targets: List[str],
        num_bins: int,
        split: str,
        early_window_dir: str,
        split_window_minutes: Optional[int] = None,
        include_root_path: bool = True,
        use_future_horizons: bool = False,
        future_horizons_dir: Optional[str] = None,
        score_column: str = "score",
        limit_trees: Optional[int] = None,
    ):
        super().__init__()
        assert split in {"train", "val", "test"}

        self.window_minutes = int(window_minutes)
        self.num_bins = int(num_bins)
        self.include_root_path = bool(include_root_path)
        self.targets = targets
        self.use_future_horizons = bool(use_future_horizons)

        meta_cols = ["tree_id", "root_author", "root_post_id", "root_timestamp"] + FINAL_TARGET_COLUMNS
        meta_df = pd.read_parquet(meta_path, columns=meta_cols)
        meta_df = meta_df.dropna(subset=["tree_id", "root_author", "root_post_id", "root_timestamp"] + FINAL_TARGET_COLUMNS)
        meta_df["tree_id"] = _safe_id_convert(meta_df["tree_id"])
        meta_df["root_author"] = _safe_id_convert(meta_df["root_author"])
        meta_df["root_post_id"] = _safe_id_convert(meta_df["root_post_id"])
        meta_df["root_timestamp"] = _safe_to_datetime(meta_df["root_timestamp"])

        train_ids, val_ids, test_ids = load_npy_splits(early_window_dir, split_window_minutes or window_minutes)
        if split == "train":
            split_ids = set(train_ids.tolist())
        elif split == "val":
            split_ids = set(val_ids.tolist())
        else:
            split_ids = set(test_ids.tolist())

        meta_df = meta_df[meta_df["tree_id"].isin(split_ids)].copy()
        if limit_trees is not None:
            keep = sorted(meta_df["tree_id"].unique().tolist())[: int(limit_trees)]
            meta_df = meta_df[meta_df["tree_id"].isin(keep)].copy()

        root_score_lookup = _load_root_scores(posts_path, score_column)

        if self.window_minutes == 0:
            edges_df = pd.DataFrame(columns=["tree_id", "post_id", "user_id", "reply_to"])
        else:
            if edges_path is None:
                raise ValueError("edges_path is required for non-root-only windows")
            edges_df = pd.read_parquet(edges_path, columns=["tree_id", "post_id", "user_id", "reply_to"])
            edges_df = edges_df.dropna(subset=["tree_id", "post_id", "user_id"])
            edges_df["tree_id"] = _safe_id_convert(edges_df["tree_id"])
            edges_df["post_id"] = _safe_id_convert(edges_df["post_id"])
            edges_df["user_id"] = _safe_id_convert(edges_df["user_id"])
            edges_df["reply_to"] = edges_df["reply_to"].apply(
                lambda x: hash(str(x)) % (2**31) if pd.notna(x) and isinstance(x, str) else (int(x) if pd.notna(x) else np.nan)
            )
            edges_df = edges_df[edges_df["tree_id"].isin(meta_df["tree_id"].unique())]

        self.user2idx, self.unk_idx = build_user_vocab(edges_df, meta_df)
        self.num_users = len(self.user2idx) + 1

        posts_df = pd.read_parquet(posts_path, columns=["tree_id", "post_id", "timestamp"])
        posts_df = posts_df.dropna(subset=["tree_id", "post_id", "timestamp"])
        posts_df["tree_id"] = _safe_id_convert(posts_df["tree_id"])
        posts_df["post_id"] = _safe_id_convert(posts_df["post_id"])
        posts_df["timestamp"] = _safe_to_datetime(posts_df["timestamp"])
        posts_df = posts_df[posts_df["tree_id"].isin(meta_df["tree_id"].unique())]

        edges_by_tree = {tree_id: df for tree_id, df in edges_df.groupby("tree_id", sort=False)}
        posts_by_tree = {tree_id: df for tree_id, df in posts_df.groupby("tree_id", sort=False)}

        future_gt_tables: Dict[str, pd.DataFrame] = {}
        future_mask_df: Optional[pd.DataFrame] = None
        if self.use_future_horizons:
            if not future_horizons_dir:
                raise ValueError("future_horizons_dir must be provided when USE_FUTURE_HORIZONS is enabled")
            future_gt_tables, future_mask_df = _load_future_horizon_tables(future_horizons_dir)

        final_mask = np.ones(len(self.targets), dtype=bool)
        intermediate_mask = np.ones(len(self.targets), dtype=bool)
        intermediate_mask[config.ROOT_SCORE_IDX] = False

        self.examples: List[PathsExample] = []
        for row in meta_df.itertuples(index=False):
            tree_id = int(row.tree_id)
            root_ts = row.root_timestamp
            if pd.isna(root_ts):
                continue

            if self.window_minutes == 0:
                paths, bin_ids = build_root_only_paths(row.root_author, self.user2idx, self.unk_idx)
            else:
                tree_edges = edges_by_tree.get(tree_id)
                if tree_edges is None or len(tree_edges) == 0:
                    continue
                tree_posts_ts = posts_by_tree.get(tree_id, pd.DataFrame(columns=["post_id", "timestamp"]))
                paths, bin_ids = build_paths_for_tree(
                    tree_edges=tree_edges,
                    tree_posts_ts=tree_posts_ts,
                    root_post_id=int(row.root_post_id),
                    root_author=int(row.root_author),
                    root_ts=root_ts,
                    user2idx=self.user2idx,
                    unk_idx=self.unk_idx,
                    observation_window=self.window_minutes,
                    num_bins=self.num_bins,
                    include_root_path=self.include_root_path,
                )
                if len(paths) == 0:
                    continue

            final_y = _final_target_vector(pd.Series(row._asdict()), root_score_lookup)
            if not self.use_future_horizons:
                self.examples.append(
                    PathsExample(
                        tree_id=tree_id,
                        paths=paths,
                        bin_ids=bin_ids,
                        y=final_y,
                        horizon_mask=final_mask.copy(),
                        horizon_idx=config.HORIZON_INDEX["final"],
                        horizon_label="final",
                    )
                )
                continue

            assert future_mask_df is not None
            for horizon in ("4h", "8h", "16h", "24h"):
                valid_col = f"valid_{horizon}"
                if tree_id not in future_mask_df.index or not bool(future_mask_df.at[tree_id, valid_col]):
                    continue
                if tree_id not in future_gt_tables[horizon].index:
                    continue
                gt_row = future_gt_tables[horizon].loc[tree_id]
                y = np.zeros(len(self.targets), dtype=np.float32)
                y[: config.ROOT_SCORE_IDX] = gt_row[HORIZON_GT_COLUMNS[horizon]].to_numpy(dtype=np.float32)
                self.examples.append(
                    PathsExample(
                        tree_id=tree_id,
                        paths=paths,
                        bin_ids=bin_ids,
                        y=y,
                        horizon_mask=intermediate_mask.copy(),
                        horizon_idx=config.HORIZON_INDEX[horizon],
                        horizon_label=horizon,
                    )
                )

            self.examples.append(
                PathsExample(
                    tree_id=tree_id,
                    paths=paths,
                    bin_ids=bin_ids,
                    y=final_y,
                    horizon_mask=final_mask.copy(),
                    horizon_idx=config.HORIZON_INDEX["final"],
                    horizon_label="final",
                )
            )

        if not self.examples:
            raise RuntimeError("No examples built. Check data paths, splits, and filters.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PathsExample:
        return self.examples[idx]


def collate_paths_batch(batch: List[PathsExample]) -> Dict[str, torch.Tensor | List[str]]:
    flat_paths: List[List[int]] = []
    flat_bins: List[int] = []
    flat_tree_ptr: List[int] = []
    ys = []
    tree_ids = []
    horizon_masks = []
    horizon_indices = []
    horizon_labels = []

    for batch_idx, example in enumerate(batch):
        ys.append(example.y)
        tree_ids.append(example.tree_id)
        horizon_masks.append(example.horizon_mask)
        horizon_indices.append(example.horizon_idx)
        horizon_labels.append(example.horizon_label)
        for path, bin_id in zip(example.paths, example.bin_ids):
            flat_paths.append(path)
            flat_bins.append(int(bin_id))
            flat_tree_ptr.append(int(batch_idx))

    lengths = torch.tensor([len(path) for path in flat_paths], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) > 0 else 1
    paths_padded = torch.zeros((len(flat_paths), max_len), dtype=torch.long)
    for idx, path in enumerate(flat_paths):
        paths_padded[idx, : len(path)] = torch.tensor(path, dtype=torch.long)

    return {
        "paths_padded": paths_padded,
        "lengths": lengths,
        "bin_ids": torch.tensor(flat_bins, dtype=torch.long),
        "tree_ptr": torch.tensor(flat_tree_ptr, dtype=torch.long),
        "y": torch.tensor(np.stack(ys, axis=0), dtype=torch.float32),
        "horizon_mask": torch.tensor(np.stack(horizon_masks, axis=0), dtype=torch.bool),
        "horizon_idx": torch.tensor(horizon_indices, dtype=torch.long),
        "horizon_one_hot": torch.nn.functional.one_hot(
            torch.tensor(horizon_indices, dtype=torch.long),
            num_classes=config.N_HORIZONS,
        ).to(torch.float32),
        "tree_id": torch.tensor(tree_ids, dtype=torch.long),
        "horizon_label": horizon_labels,
    }
