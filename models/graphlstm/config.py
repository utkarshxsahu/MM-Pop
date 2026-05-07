"""
Configuration for Graph-LSTM experiments.
"""

import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "graphlstm")
RUN_OUTPUT_ROOT = None

TARGET_NAMES = [
    "max_width",
    "max_depth",
    "structural_virality",
    "num_posts",
    "num_unique_users",
    "root_score",
]

# Future-horizon tags used when MODE is "future_horizon".
# These are the exact labels used for expanded supervision and reporting.
HORIZON_TAGS = ["4h", "8h", "16h", "24h", "final"]
HORIZON_TO_INDEX = {tag: idx for idx, tag in enumerate(HORIZON_TAGS)}
HORIZON_ONE_HOTS = {
    tag: [1.0 if i == idx else 0.0 for i in range(len(HORIZON_TAGS))]
    for tag, idx in HORIZON_TO_INDEX.items()
}
HORIZON_MINUTES = {"4h": 240, "8h": 480, "16h": 960, "24h": 1440}

# Run in the original setup:
# one sample per tree, predicting the final thread state only.
MODE_STANDARD = "standard"

# Run in future-horizon mode:
# each tree expands into up to 5 samples (4h, 8h, 16h, 24h, final).
MODE_FUTURE_HORIZON = "future_horizon"

# Special window tag meaning "use only the root post as model input".
# This is not a real 0-minute snapshot file; it reuses the dataset's
# reference split files while constructing a single-node input graph.
WINDOW_ROOT_ONLY = "root_only"


DATASET_REGISTRY = {
    "bluesky": {
        "base_dir": os.path.join(DATASETS_ROOT, "bluesky"),
        "thread_metadata": os.path.join(DATASETS_ROOT, "bluesky", "metadata", "thread_metadata_updated_added.parquet"),
        "thread_posts": os.path.join(DATASETS_ROOT, "bluesky", "metadata", "thread_posts_with_all_labels2.parquet"),
        "user_degrees": os.path.join(DATASETS_ROOT, "bluesky", "metadata", "user_degrees.csv"),
        "has_user_degrees": True,
        "early_window_dir": os.path.join(DATASETS_ROOT, "bluesky", "snapshots"),
        "output_dir": RESULTS_ROOT,
        "windows": [2, 10],
        "root_only_split_window": 2,
        "root_score_column": "like_count",
    },
    "gaming": {
        "base_dir": os.path.join(DATASETS_ROOT, "reddit", "gaming"),
        "thread_metadata": os.path.join(DATASETS_ROOT, "reddit", "gaming", "metadata", "reddit_gaming_metadata.parquet"),
        "thread_posts": os.path.join(DATASETS_ROOT, "reddit", "gaming", "metadata", "reddit_gaming_posts.parquet"),
        "user_degrees": None,
        "has_user_degrees": False,
        "early_window_dir": os.path.join(DATASETS_ROOT, "reddit", "gaming", "snapshots"),
        "output_dir": RESULTS_ROOT,
        "windows": [20, 50],
        "root_only_split_window": 20,
        "root_score_column": "score",
    },
    "futurology": {
        "base_dir": os.path.join(DATASETS_ROOT, "reddit", "futurology"),
        "thread_metadata": os.path.join(DATASETS_ROOT, "reddit", "futurology", "metadata", "reddit_futurology_metadata.parquet"),
        "thread_posts": os.path.join(DATASETS_ROOT, "reddit", "futurology", "metadata", "reddit_futurology_posts.parquet"),
        "user_degrees": None,
        "has_user_degrees": False,
        "early_window_dir": os.path.join(DATASETS_ROOT, "reddit", "futurology", "snapshots"),
        "output_dir": RESULTS_ROOT,
        "windows": [90],
        "root_only_split_window": 90,
        "root_score_column": "score",
    },
    "ama": {
        "base_dir": os.path.join(DATASETS_ROOT, "reddit", "ama"),
        "thread_metadata": os.path.join(DATASETS_ROOT, "reddit", "ama", "metadata", "reddit_ama_metadata.parquet"),
        "thread_posts": os.path.join(DATASETS_ROOT, "reddit", "ama", "metadata", "reddit_ama_posts.parquet"),
        "user_degrees": None,
        "has_user_degrees": False,
        "early_window_dir": os.path.join(DATASETS_ROOT, "reddit", "ama", "snapshots"),
        "output_dir": RESULTS_ROOT,
        "windows": [30],
        "root_only_split_window": 15,
        "root_score_column": "score",
    },
}


# Which dataset(s) to run.
# Examples:
#   DATASETS = ["gaming"]
#   DATASETS = ["bluesky", "ama"]
#   DATASETS = "all"
DATASETS = "all"

# Which time windows to run for every selected dataset.
# Use None to take the dataset defaults from DATASET_REGISTRY.
# Examples:
#   WINDOWS = None
#   WINDOWS = [20, 50, 90]
#   WINDOWS = [WINDOW_ROOT_ONLY]
#   WINDOWS = [WINDOW_ROOT_ONLY, 20, 50, 90]
WINDOWS = None

# If True, prepend root_only in addition to WINDOWS/default windows.
# If False, root_only is used only when explicitly listed in WINDOWS.
INCLUDE_ROOT_ONLY = True

# Main experiment mode.
# Set to MODE_STANDARD or MODE_FUTURE_HORIZON.
MODE = MODE_FUTURE_HORIZON

PRETRAINED_EMBEDDINGS_PATH = None
PRETRAINED_EMBEDDINGS_VOCAB_PATH = None

# Embedding training behavior:
#   "fine_tune" -> load pretrained embeddings and keep training them
#   "freeze"    -> load pretrained embeddings and keep them fixed
EMBEDDING_TRAINING_MODE = "fine_tune"

# Smaller learning rate for embeddings when EMBEDDING_TRAINING_MODE="fine_tune".
EMBEDDING_LR = 1e-4

# Optimizer for the model body/head: "adam" or "adamw".
OPTIMIZER_NAME = "adamw"

# Final Graph-LSTM baseline hyperparameters.
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
DROPOUT = 0.3
FINAL_MLP_HIDDEN = 128
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3

# Training loop settings.
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 10
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tracks the current dataset while looping over DATASETS.
ACTIVE_DATASET = DATASETS[0]


def set_active_dataset(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    global ACTIVE_DATASET
    ACTIVE_DATASET = dataset_name


def get_dataset_config(dataset_name=None):
    dataset_name = dataset_name or ACTIVE_DATASET
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_REGISTRY[dataset_name]


def get_requested_datasets():
    if DATASETS == "all":
        return list(DATASET_REGISTRY.keys())
    if isinstance(DATASETS, str):
        return [DATASETS]
    return list(DATASETS)


def get_requested_windows(dataset_name=None):
    dataset_cfg = get_dataset_config(dataset_name)
    windows = list(dataset_cfg["windows"] if WINDOWS is None else WINDOWS)
    if INCLUDE_ROOT_ONLY and WINDOW_ROOT_ONLY not in windows:
        windows = [WINDOW_ROOT_ONLY] + windows
    return windows


def is_future_horizon_mode():
    return MODE == MODE_FUTURE_HORIZON


def get_reference_split_window(dataset_name=None):
    return get_dataset_config(dataset_name)["root_only_split_window"]


def get_split_window(window_tag, dataset_name=None):
    if window_tag == WINDOW_ROOT_ONLY:
        return get_reference_split_window(dataset_name)
    return int(window_tag)


def get_window_label(window_tag):
    if window_tag == WINDOW_ROOT_ONLY:
        return WINDOW_ROOT_ONLY
    return f"{int(window_tag)}min"


def get_output_root(dataset_name=None, mode=None):
    dataset_cfg = get_dataset_config(dataset_name)
    mode = mode or MODE
    base_dir = RUN_OUTPUT_ROOT or dataset_cfg["output_dir"]
    return os.path.join(base_dir, dataset_name or ACTIVE_DATASET, mode)
