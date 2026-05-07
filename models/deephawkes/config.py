"""
DeepHawkes configuration.
"""
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "deephawkes")
BLUESKY_DATA_ROOT = os.path.join(DATASETS_ROOT, "bluesky")
BLUESKY_SNAPSHOT_ROOT = os.path.join(BLUESKY_DATA_ROOT, "snapshots")
REDDIT_ROOT = os.path.join(DATASETS_ROOT, "reddit")

TARGETS = [
    "max_width",
    "max_depth",
    "structural_virality",
    "num_posts",
    "num_unique_users",
    "root_score",
]
ROOT_SCORE_IDX = 5

USE_FUTURE_HORIZONS = True
HORIZON_HOURS = ["4h", "8h", "16h", "24h", "final"]
HORIZON_INDEX = {"4h": 0, "8h": 1, "16h": 2, "24h": 3, "final": 4}
N_HORIZONS = 5

DATASETS = {
    "bluesky": {
        "type": "bluesky",
        "paths": {
            "thread_metadata": os.path.join(BLUESKY_DATA_ROOT, "metadata", "thread_metadata_updated_added.parquet"),
            "thread_posts": os.path.join(BLUESKY_DATA_ROOT, "metadata", "thread_posts_with_all_labels2.parquet"),
            "early_window_dir": BLUESKY_SNAPSHOT_ROOT,
            "future_horizons_dir": os.path.join(BLUESKY_SNAPSHOT_ROOT, "future_horizons"),
        },
        "windows": [0, 2, 10],
        "root_only_split_window": 2,
        "root_score_column": "like_count",
    },
    "gaming": {
        "type": "reddit",
        "paths": {
            "thread_metadata": os.path.join(REDDIT_ROOT, "gaming", "metadata", "reddit_gaming_metadata.parquet"),
            "thread_posts": os.path.join(REDDIT_ROOT, "gaming", "metadata", "reddit_gaming_posts.parquet"),
            "early_window_dir": os.path.join(REDDIT_ROOT, "gaming", "snapshots"),
            "future_horizons_dir": os.path.join(REDDIT_ROOT, "gaming", "snapshots", "future_horizons"),
        },
        "windows": [0, 20, 50],
        "root_only_split_window": 20,
        "root_score_column": "score",
    },
    "futurology": {
        "type": "reddit",
        "paths": {
            "thread_metadata": os.path.join(REDDIT_ROOT, "futurology", "metadata", "reddit_futurology_metadata.parquet"),
            "thread_posts": os.path.join(REDDIT_ROOT, "futurology", "metadata", "reddit_futurology_posts.parquet"),
            "early_window_dir": os.path.join(REDDIT_ROOT, "futurology", "snapshots"),
            "future_horizons_dir": os.path.join(REDDIT_ROOT, "futurology", "snapshots", "future_horizons"),
        },
        "windows": [0, 90],
        "root_only_split_window": 90,
        "root_score_column": "score",
    },
    "ama": {
        "type": "reddit",
        "paths": {
            "thread_metadata": os.path.join(REDDIT_ROOT, "ama", "metadata", "reddit_ama_metadata.parquet"),
            "thread_posts": os.path.join(REDDIT_ROOT, "ama", "metadata", "reddit_ama_posts.parquet"),
            "early_window_dir": os.path.join(REDDIT_ROOT, "ama", "snapshots"),
            "future_horizons_dir": os.path.join(REDDIT_ROOT, "ama", "snapshots", "future_horizons"),
        },
        "windows": [0, 30],
        "root_only_split_window": 15,
        "root_score_column": "score",
    },
}

HYPERPARAMETER_CONFIG = {
    "name": "original_based_setting",
    "embedding_dim": 50,
    "hidden_dim": 32,
    "num_bins": 10,
    "dropout": 0.5,
    "lr_emb": 5e-4,
    "lr": 5e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
}

EPOCHS = 200
PATIENCE = 10
NUM_WORKERS = 15
SEED = 42

DATASETS_TO_RUN = ["bluesky", "gaming", "futurology", "ama"]
SAVE_CHECKPOINTS = False
DEBUG_MODE = False
DEBUG_LIMIT_TREES = 100
USE_GPU = True

RESULTS_BASE_DIR = RESULTS_ROOT
