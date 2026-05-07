"""
Dataset for CasSeqGCN with optional future-horizon expansion.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


class CasSeqGCNDataset(Dataset):
    def __init__(
        self,
        tree_ids,
        edges_df,
        meta_df,
        posts_df,
        preprocessor,
        future_horizons=False,
        future_horizon_gt=None,
        future_horizon_valid_mask=None,
    ):
        self.preprocessor = preprocessor
        self.future_horizons = future_horizons

        self.meta_df = meta_df.set_index("tree_id") if "tree_id" in meta_df.columns else meta_df
        self.edges_by_tree = {
            tid: grp.reset_index(drop=True)
            for tid, grp in edges_df.groupby("tree_id")
        }

        self.ts_by_tree = {}
        if posts_df is not None and "timestamp" in posts_df.columns:
            for tid, grp in posts_df.groupby("tree_id"):
                self.ts_by_tree[tid] = dict(zip(grp["post_id"], grp["timestamp"]))

        self.valid_mask = None
        if future_horizon_valid_mask is not None:
            self.valid_mask = future_horizon_valid_mask.set_index("tree_id")

        self.horizon_gt = {}
        if future_horizon_gt is not None:
            for horizon, frame in future_horizon_gt.items():
                self.horizon_gt[horizon] = frame.set_index("tree_id")

        self.samples = self._build_samples(tree_ids)
        self._graph_cache = {}

    def _build_samples(self, tree_ids):
        samples = []
        for tree_id in tree_ids:
            if tree_id not in self.edges_by_tree or tree_id not in self.meta_df.index:
                continue

            if self.future_horizons and self.valid_mask is not None:
                for horizon in config.INTERMEDIATE_HORIZONS:
                    if self._is_valid_horizon(tree_id, horizon):
                        samples.append(self._make_sample(tree_id, horizon))

            samples.append(self._make_sample(tree_id, "final"))
        return samples

    def _is_valid_horizon(self, tree_id, horizon):
        if tree_id not in self.valid_mask.index:
            return False
        if tree_id not in self.horizon_gt.get(horizon, pd.DataFrame()).index:
            return False
        valid_col = f"valid_{horizon}"
        if valid_col not in self.valid_mask.columns:
            return False
        return bool(self.valid_mask.loc[tree_id, valid_col])

    def _make_sample(self, tree_id, horizon_tag):
        if horizon_tag == "final":
            meta_row = self.meta_df.loc[tree_id, config.TARGET_NAMES]
            if isinstance(meta_row, pd.DataFrame):
                meta_row = meta_row.iloc[0]
            raw_targets = meta_row.to_numpy(dtype=np.float32)
            target_mask = np.ones(len(config.TARGET_NAMES), dtype=bool)
        else:
            raw_targets = self._get_intermediate_targets(tree_id, horizon_tag)
            target_mask = np.array([True, True, True, True, True, False], dtype=bool)

        targets = self.preprocessor.transform_targets(raw_targets)
        return {
            "tree_id": tree_id,
            "horizon_tag": horizon_tag,
            "horizon_onehot": np.asarray(config.HORIZON_ONEHOT[horizon_tag], dtype=np.float32),
            "targets": targets,
            "target_mask": target_mask,
        }

    def _get_intermediate_targets(self, tree_id, horizon_tag):
        frame = self.horizon_gt[horizon_tag]
        row = frame.loc[tree_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return np.array(
            [
                row["gt_max_width"],
                row["gt_max_depth"],
                row["gt_structural_virality"],
                row["gt_num_posts"],
                row["gt_num_unique_users"],
                0.0,
            ],
            dtype=np.float32,
        )

    def _build_graph(self, edges):
        post_ids = edges["post_id"].unique().tolist()
        node_idx = {pid: i for i, pid in enumerate(post_ids)}
        n_nodes = len(post_ids)

        in_deg = np.zeros(n_nodes, dtype=np.float32)
        out_deg = np.zeros(n_nodes, dtype=np.float32)
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        for _, row in edges.iterrows():
            src = row["post_id"]
            dst = row["reply_to"]
            if pd.isna(dst) or src not in node_idx or dst not in node_idx:
                continue
            i, j = node_idx[src], node_idx[dst]
            out_deg[i] += 1
            in_deg[j] += 1
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        return node_idx, adj, in_deg, out_deg, n_nodes

    @staticmethod
    def _compute_laplacian(adj):
        n_nodes = adj.shape[0]
        deg = adj.sum(1)
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-8)), 0.0)
        d_mat = np.diag(inv_sqrt)
        return np.eye(n_nodes, dtype=np.float32) - d_mat @ adj @ d_mat

    def _get_post_order(self, tree_id, edges, node_idx):
        post_ids = list(node_idx.keys())
        if tree_id in self.ts_by_tree:
            ts_map = self.ts_by_tree[tree_id]
            return sorted(post_ids, key=lambda post_id: ts_map.get(post_id, float("inf")))

        known = set(node_idx.keys())
        root_candidates = edges[
            edges["reply_to"].isna() | ~edges["reply_to"].isin(known)
        ]["post_id"].tolist()
        root = root_candidates[0] if root_candidates else post_ids[0]

        children = {}
        for _, row in edges.iterrows():
            dst = row["reply_to"]
            if not pd.isna(dst) and dst in known:
                children.setdefault(dst, []).append(row["post_id"])

        order = []
        visited = set()
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            queue.extend(children.get(node, []))

        order.extend([post_id for post_id in post_ids if post_id not in visited])
        return order

    def _build_snapshots(self, post_order, node_idx, in_deg, out_deg, n_nodes):
        n_posts = len(post_order)
        snap_ends = list(range(0, n_posts, config.Q))
        if snap_ends[-1] != n_posts - 1:
            snap_ends.append(n_posts - 1)
        snap_ends = snap_ends[: config.K_MAX]

        n_snapshots = len(snap_ends)
        snapshots = np.zeros((n_snapshots, n_nodes, config.NODE_FEATURE_DIM), dtype=np.float32)

        for snap_idx, end in enumerate(snap_ends):
            active = set(post_order[: end + 1])
            for post_id, node_idx_value in node_idx.items():
                snapshots[snap_idx, node_idx_value, 0] = 1.0 if post_id in active else 0.0
                snapshots[snap_idx, node_idx_value, 1] = in_deg[node_idx_value]
                snapshots[snap_idx, node_idx_value, 2] = out_deg[node_idx_value]

        return snapshots, n_snapshots

    def _process_tree_graph(self, tree_id):
        edges = self.edges_by_tree[tree_id]
        node_idx, adj, in_deg, out_deg, n_nodes = self._build_graph(edges)
        laplacian = self._compute_laplacian(adj)
        post_order = self._get_post_order(tree_id, edges, node_idx)
        snapshots, n_snapshots = self._build_snapshots(post_order, node_idx, in_deg, out_deg, n_nodes)
        return {
            "snapshots": torch.FloatTensor(snapshots),
            "L_sn": torch.FloatTensor(laplacian),
            "n_nodes": n_nodes,
            "n_snapshots": n_snapshots,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tree_id = sample["tree_id"]
        if tree_id not in self._graph_cache:
            self._graph_cache[tree_id] = self._process_tree_graph(tree_id)

        graph = self._graph_cache[tree_id]
        return {
            **graph,
            "targets": torch.FloatTensor(sample["targets"]),
            "target_mask": torch.BoolTensor(sample["target_mask"]),
            "horizon_onehot": torch.FloatTensor(sample["horizon_onehot"]),
            "horizon_tag": sample["horizon_tag"],
            "tree_id": tree_id,
        }


def collate_fn(batch):
    batch_size = len(batch)
    max_nodes = max(item["n_nodes"] for item in batch)
    max_snapshots = max(item["n_snapshots"] for item in batch)
    feat_dim = batch[0]["snapshots"].shape[-1]

    snapshots = torch.zeros(batch_size, max_snapshots, max_nodes, feat_dim)
    laplacian = torch.zeros(batch_size, max_nodes, max_nodes)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    snap_mask = torch.zeros(batch_size, max_snapshots, dtype=torch.bool)
    targets = torch.stack([item["targets"] for item in batch])
    target_mask = torch.stack([item["target_mask"] for item in batch])
    horizon_onehot = torch.stack([item["horizon_onehot"] for item in batch])

    for idx, item in enumerate(batch):
        n_nodes = item["n_nodes"]
        n_snapshots = item["n_snapshots"]
        snapshots[idx, :n_snapshots, :n_nodes] = item["snapshots"]
        laplacian[idx, :n_nodes, :n_nodes] = item["L_sn"]
        node_mask[idx, :n_nodes] = True
        snap_mask[idx, :n_snapshots] = True

    return {
        "snapshots": snapshots,
        "L_sn": laplacian,
        "node_mask": node_mask,
        "snap_mask": snap_mask,
        "targets": targets,
        "target_mask": target_mask,
        "horizon_onehot": horizon_onehot,
        "horizon_tags": [item["horizon_tag"] for item in batch],
        "tree_ids": [item["tree_id"] for item in batch],
    }
