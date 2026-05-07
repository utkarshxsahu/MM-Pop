"""
PyTorch dataset for DeepCas.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import config


class DeepCasDataset(Dataset):
    """Dataset for DeepCas model with optional future-horizon expansion."""

    def __init__(self, tree_ids, path_cache, meta_df, preprocessor, user_to_idx, sample_specs):
        self.data = []

        meta_indexed = meta_df.set_index("tree_id")

        print(f"Creating dataset with {len(tree_ids)} trees...")
        for tree_id in tqdm(tree_ids, desc="Preparing"):
            if tree_id not in path_cache or tree_id not in meta_indexed.index:
                continue
            if tree_id not in sample_specs:
                continue

            path_data = path_cache[tree_id]
            meta_row = meta_indexed.loc[tree_id]
            if isinstance(meta_row, pd.DataFrame):
                meta_row = meta_row.iloc[0]

            paths_with_indices = []
            for path in path_data["paths"]:
                idx_path = [user_to_idx.get(str(uid), 0) for uid in path]
                paths_with_indices.append(idx_path)

            paths = np.array(paths_with_indices, dtype=np.int64)
            tree_size = path_data["num_nodes"]

            for spec in sample_specs[tree_id]:
                horizon_one_hot = np.zeros(len(config.HORIZON_TAGS), dtype=np.float32)
                horizon_one_hot[spec["horizon_index"]] = 1.0
                global_features = preprocessor.transform_global_features(
                    meta_row, horizon_one_hot=horizon_one_hot
                )

                self.data.append(
                    {
                        "paths": torch.LongTensor(paths),
                        "global_features": torch.FloatTensor(global_features),
                        "targets": torch.FloatTensor(spec["targets"]),
                        "target_mask": torch.BoolTensor(spec["target_mask"]),
                        "tree_size": tree_size,
                        "tree_id": tree_id,
                        "horizon_tag": spec["horizon_tag"],
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Custom collate function."""
    return {
        "paths": torch.stack([item["paths"] for item in batch]),
        "global_features": torch.stack([item["global_features"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
        "target_mask": torch.stack([item["target_mask"] for item in batch]),
        "tree_sizes": torch.LongTensor([item["tree_size"] for item in batch]),
        "tree_ids": [item["tree_id"] for item in batch],
        "horizon_tags": [item["horizon_tag"] for item in batch],
    }
