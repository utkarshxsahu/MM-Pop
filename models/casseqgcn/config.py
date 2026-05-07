"""
Configuration for CasSeqGCN experiments.
"""

import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "casseqgcn")

TARGET_NAMES = [
    "max_width",
    "max_depth",
    "structural_virality",
    "num_posts",
    "num_unique_users",
    "root_score",
]
ROOT_SCORE_INDEX = 5

HORIZON_ORDER = ["4h", "8h", "16h", "24h", "final"]
HORIZON_TO_INDEX = {name: idx for idx, name in enumerate(HORIZON_ORDER)}
INTERMEDIATE_HORIZONS = HORIZON_ORDER[:-1]
HORIZON_ONEHOT = {
    name: [1.0 if i == idx else 0.0 for i in range(len(HORIZON_ORDER))]
    for name, idx in HORIZON_TO_INDEX.items()
}

ROOT_ONLY_WINDOW = "root_only"
ROOT_ONLY_REFERENCE_WINDOW = {
    "bluesky": 2,
    "gaming": 20,
    "futurology": 90,
    "ama": 15,
}

DATASET_SPECS = {
    "bluesky": {
        "metadata_path": os.path.join(DATASETS_ROOT, "bluesky", "metadata", "thread_metadata_updated_added.parquet"),
        "posts_path": os.path.join(DATASETS_ROOT, "bluesky", "metadata", "thread_posts_with_all_labels2.parquet"),
        "early_window_dir": os.path.join(DATASETS_ROOT, "bluesky", "snapshots"),
        "embedding_dir": os.path.join(DATASETS_ROOT, "bluesky", "embeddings"),
        "output_dir": RESULTS_ROOT,
        "windows": [2, 10],
        "root_score_column": "like_count",
    },
    "gaming": {
        "metadata_path": os.path.join(DATASETS_ROOT, "reddit", "gaming", "metadata", "reddit_gaming_metadata.parquet"),
        "posts_path": os.path.join(DATASETS_ROOT, "reddit", "gaming", "metadata", "reddit_gaming_posts.parquet"),
        "early_window_dir": os.path.join(DATASETS_ROOT, "reddit", "gaming", "snapshots"),
        "embedding_dir": os.path.join(DATASETS_ROOT, "reddit", "gaming", "embeddings"),
        "image_dir": os.path.join(DATASETS_ROOT, "reddit", "gaming", "images"),
        "output_dir": RESULTS_ROOT,
        "windows": [20, 50],
        "root_score_column": "score",
    },
    "futurology": {
        "metadata_path": os.path.join(DATASETS_ROOT, "reddit", "futurology", "metadata", "reddit_futurology_metadata.parquet"),
        "posts_path": os.path.join(DATASETS_ROOT, "reddit", "futurology", "metadata", "reddit_futurology_posts.parquet"),
        "early_window_dir": os.path.join(DATASETS_ROOT, "reddit", "futurology", "snapshots"),
        "embedding_dir": os.path.join(DATASETS_ROOT, "reddit", "futurology", "embeddings"),
        "image_dir": os.path.join(DATASETS_ROOT, "reddit", "futurology", "images"),
        "output_dir": RESULTS_ROOT,
        "windows": [90],
        "root_score_column": "score",
    },
    "ama": {
        "metadata_path": os.path.join(DATASETS_ROOT, "reddit", "ama", "metadata", "reddit_ama_metadata.parquet"),
        "posts_path": os.path.join(DATASETS_ROOT, "reddit", "ama", "metadata", "reddit_ama_posts.parquet"),
        "early_window_dir": os.path.join(DATASETS_ROOT, "reddit", "ama", "snapshots"),
        "embedding_dir": os.path.join(DATASETS_ROOT, "reddit", "ama", "embeddings"),
        "image_dir": os.path.join(DATASETS_ROOT, "reddit", "ama", "images"),
        "output_dir": RESULTS_ROOT,
        "windows": [30],
        "root_score_column": "score",
    },
}

CANDIDATE_CONFIGS = [
    {
        "name": "lr_0.005",
        "gcn_hidden": 32,
        "lstm_hidden": 32,
        "gcn_layers": 2,
        "lstm_layers": 2,
        "dropout": 0.5,
        "weight_decay": 1e-5,
        "lr": 0.005,
    },
    {
        "name": "lr_0.01",
        "gcn_hidden": 32,
        "lstm_hidden": 32,
        "gcn_layers": 2,
        "lstm_layers": 2,
        "dropout": 0.5,
        "weight_decay": 1e-5,
        "lr": 0.01,
    },
    {
        "name": "lr_0.03",
        "gcn_hidden": 32,
        "lstm_hidden": 32,
        "gcn_layers": 2,
        "lstm_layers": 2,
        "dropout": 0.5,
        "weight_decay": 1e-5,
        "lr": 0.03,
    },
    {
        "name": "lr_0.05",
        "gcn_hidden": 32,
        "lstm_hidden": 32,
        "gcn_layers": 2,
        "lstm_layers": 2,
        "dropout": 0.5,
        "weight_decay": 1e-5,
        "lr": 0.05,
    },
]

NODE_FEATURE_DIM = 3
NODE_EMBED_DIM = 32
SNAPSHOT_EMBED_DIM = 32
N_ROUTING_ITER = 3
Q = 5
K_MAX = 15

BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 10
TRIAL_EPOCHS = 20
TRIAL_PATIENCE = 5
NUM_WORKERS = 8
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset_spec(dataset_name):
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_SPECS[dataset_name]


def get_window_labels(dataset_name, include_root_only=False, requested_windows=None):
    spec = get_dataset_spec(dataset_name)
    if requested_windows in (None, [], "all"):
        windows = list(spec["windows"])
    else:
        windows = []
        for window in requested_windows:
            if window == ROOT_ONLY_WINDOW:
                windows.append(ROOT_ONLY_WINDOW)
            else:
                windows.append(int(window))

    if include_root_only and ROOT_ONLY_WINDOW not in windows:
        windows = [ROOT_ONLY_WINDOW] + windows
    return windows


def get_split_window(dataset_name, window_label):
    if window_label == ROOT_ONLY_WINDOW:
        return ROOT_ONLY_REFERENCE_WINDOW[dataset_name]
    return int(window_label)


def get_split_dir(dataset_name):
    return os.path.join(get_dataset_spec(dataset_name)["early_window_dir"], "splits")


def get_future_horizon_dir(dataset_name):
    return os.path.join(get_dataset_spec(dataset_name)["early_window_dir"], "future_horizons")
