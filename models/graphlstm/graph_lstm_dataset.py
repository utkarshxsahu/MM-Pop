"""
PyTorch Dataset for Graph-LSTM samples.
"""

import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from data_loader import load_early_window_data
from precompute_features import get_cache_dir


STRUCT_FEAT_COLS = [
    "time_since_root",
    "log_time_since_root",
    "time_since_parent",
    "log_time_since_parent",
    "num_previous",
    "num_later",
    "is_op",
    "author_count",
    "depth",
    "num_siblings",
    "num_children",
    "subtree_height",
    "subtree_size",
    "norm_children_mean",
    "norm_children_rank",
    "norm_subtree_mean",
    "norm_subtree_rank",
]


class GraphLSTMDataset(Dataset):
    def __init__(self, samples_df, window_tag, dataset_name=None):
        self.samples_df = samples_df.reset_index(drop=True).copy()
        self.window_tag = window_tag
        self.dataset_name = dataset_name or config.ACTIVE_DATASET

        cache_dir = get_cache_dir(window_tag, self.dataset_name)
        feat_df = pd.read_parquet(os.path.join(cache_dir, "node_features.parquet"))
        with open(os.path.join(cache_dir, "token_ids.pkl"), "rb") as handle:
            self.token_ids = pickle.load(handle)
        with open(os.path.join(get_cache_dir(None, self.dataset_name), "vocab.json")) as handle:
            self.vocab_size = len(json.load(handle))

        edges_df, _ = load_early_window_data(window_tag, self.dataset_name)
        if window_tag == config.WINDOW_ROOT_ONLY:
            edges_df = edges_df.iloc[0:0].copy()

        valid_tree_ids = set(feat_df["tree_id"].unique())
        self.samples_df = self.samples_df[self.samples_df["tree_id"].isin(valid_tree_ids)].reset_index(drop=True)

        self.feats_by_tree = {
            tree_id: group.reset_index(drop=True)
            for tree_id, group in feat_df.groupby("tree_id")
        }
        self.edges_by_tree = {tree_id: group for tree_id, group in edges_df.groupby("tree_id")}

        sample_feat = next(iter(self.feats_by_tree.values()))
        self.feat_cols = [col for col in STRUCT_FEAT_COLS if col in sample_feat.columns]
        self.target_keys = list(config.TARGET_NAMES)

    @property
    def struct_dim(self):
        return len(self.feat_cols)

    @property
    def n_targets(self):
        return len(self.target_keys)

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        return self._prepare_sample(self.samples_df.iloc[idx])

    def _prepare_sample(self, sample_row):
        tree_id = sample_row["tree_id"]
        feats_df = self.feats_by_tree.get(tree_id)
        if feats_df is None or feats_df.empty:
            return None

        edges_df = self.edges_by_tree.get(tree_id, pd.DataFrame(columns=["post_id", "reply_to"]))
        nodes = feats_df["post_id"].tolist()
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        node_set = set(nodes)
        node_count = len(nodes)

        time_map = feats_df.set_index("post_id")["time_since_root"].to_dict()

        children_of = defaultdict(list)
        parent_of = {}
        for _, edge in edges_df.iterrows():
            post_id, reply_to = edge["post_id"], edge["reply_to"]
            if pd.notna(reply_to) and post_id in node_set and reply_to in node_set:
                children_of[reply_to].append(post_id)
                parent_of[post_id] = reply_to
        for parent in children_of:
            children_of[parent].sort(key=lambda x: time_map.get(x, 0))

        parent_idx = np.full(node_count, -1, dtype=np.int64)
        sib_pred_idx = np.full(node_count, -1, dtype=np.int64)
        first_child_idx = np.full(node_count, -1, dtype=np.int64)
        sib_succ_idx = np.full(node_count, -1, dtype=np.int64)

        for node in nodes:
            node_idx = node_to_idx[node]
            parent = parent_of.get(node)
            if parent and parent in node_to_idx:
                parent_idx[node_idx] = node_to_idx[parent]
                siblings = children_of.get(parent, [])
                sibling_pos = siblings.index(node) if node in siblings else -1
                if sibling_pos > 0:
                    sib_pred_idx[node_idx] = node_to_idx[siblings[sibling_pos - 1]]
                if 0 <= sibling_pos < len(siblings) - 1:
                    sib_succ_idx[node_idx] = node_to_idx[siblings[sibling_pos + 1]]
            children = children_of.get(node, [])
            if children:
                first_child_idx[node_idx] = node_to_idx[children[0]]

        struct_feats = feats_df[self.feat_cols].fillna(0).values.astype(np.float32)
        depth_arr = feats_df["depth"].fillna(0).values.astype(np.int64)

        sib_pos_map = {}
        for parent, child_list in children_of.items():
            for pos, child in enumerate(child_list):
                sib_pos_map[child] = pos
        sib_pos_arr = np.array([sib_pos_map.get(node, 0) for node in nodes], dtype=np.int64)

        token_seqs = [self.token_ids.get(node, [0]) for node in nodes]
        max_len = max((len(seq) for seq in token_seqs), default=1)
        token_arr = np.zeros((node_count, max_len), dtype=np.int64)
        token_lens = np.ones(node_count, dtype=np.int64)
        for i, seq in enumerate(token_seqs):
            seq_len = len(seq)
            if seq_len > 0:
                token_arr[i, :seq_len] = seq
                token_lens[i] = seq_len

        targets = np.array([float(sample_row[name]) for name in self.target_keys], dtype=np.float32)
        loss_mask = np.array(sample_row["loss_mask"], dtype=bool)
        horizon_vec = np.array(sample_row["horizon_vec"], dtype=np.float32)

        return {
            "tree_id": tree_id,
            "window_tag": sample_row["window_tag"],
            "horizon_tag": sample_row["horizon_tag"],
            "struct_feats": torch.FloatTensor(struct_feats),
            "token_ids": torch.LongTensor(token_arr),
            "token_lens": torch.LongTensor(token_lens),
            "parent_idx": torch.LongTensor(parent_idx),
            "sib_pred_idx": torch.LongTensor(sib_pred_idx),
            "first_child_idx": torch.LongTensor(first_child_idx),
            "sib_succ_idx": torch.LongTensor(sib_succ_idx),
            "depth_arr": torch.LongTensor(depth_arr),
            "sib_pos_arr": torch.LongTensor(sib_pos_arr),
            "targets": torch.FloatTensor(targets),
            "loss_mask": torch.BoolTensor(loss_mask),
            "horizon_vec": torch.FloatTensor(horizon_vec),
        }


def collate_fn(batch):
    return [item for item in batch if item is not None]
