"""
Configuration for standalone DeepCas implementation.
"""

import os
import torch


BLUESKY_ROOT = "/scratch/dgl/Social_Network/bluesky/processed"
REDDIT_ROOT = "/scratch/dgl/Social_Network/reddit/filtered/tree_full"

TARGET_NAMES = [
    "max_width",
    "max_depth",
    "structural_virality",
    "num_posts",
    "num_unique_users",
    "root_score",
]

HORIZON_TAGS = ["4h", "8h", "16h", "24h", "final"]
HORIZON_TO_INDEX = {tag: idx for idx, tag in enumerate(HORIZON_TAGS)}
HORIZON_TO_MINUTES = {
    "4h": 240,
    "8h": 480,
    "16h": 960,
    "24h": 1440,
}

DATASETS = {
    "bluesky": {
        "paths": {
            "thread_metadata": os.path.join(
                BLUESKY_ROOT,
                "discussion_trees/samples/tree_65k/thread_metadata_updated_added.parquet",
            ),
            "thread_posts": os.path.join(
                BLUESKY_ROOT,
                "discussion_trees/samples/tree_65k/thread_posts_with_all_labels2.parquet",
            ),
            "user_degrees": os.path.join(BLUESKY_ROOT, "followers/user_degrees.csv"),
            "early_window_dir": os.path.join(
                BLUESKY_ROOT, "discussion_trees/samples/tree_65k/gnn_snapshots"
            ),
            "split_dir": os.path.join(
                BLUESKY_ROOT,
                "discussion_trees/samples/tree_65k/gnn_snapshots/splits",
            ),
            "future_horizon_dir": os.path.join(
                BLUESKY_ROOT,
                "discussion_trees/samples/tree_65k/gnn_snapshots/future_horizons",
            ),
            "output_dir": os.path.join(
                BLUESKY_ROOT,
                "discussion_trees/samples/tree_65k/deepcas_results/bluesky",
            ),
            "embedding_dir": os.path.join(
                BLUESKY_ROOT,
                "discussion_trees/samples/tree_65k/deepcas_embeddings",
            ),
        },
        "windows": [0, 2, 10, 20],
        "root_only_reference_window": 2,
        "has_user_degrees": True,
        "root_score_column": "like_count",
    },
    "gaming": {
        "paths": {
            "thread_metadata": os.path.join(REDDIT_ROOT, "reddit_gaming_metadata.parquet"),
            "thread_posts": os.path.join(REDDIT_ROOT, "reddit_gaming_posts.parquet"),
            "user_degrees": None,
            "early_window_dir": os.path.join(REDDIT_ROOT, "snapshots/gaming"),
            "split_dir": os.path.join(REDDIT_ROOT, "snapshots/gaming/splits"),
            "future_horizon_dir": os.path.join(REDDIT_ROOT, "snapshots/gaming/future_horizons"),
            "output_dir": os.path.join(REDDIT_ROOT, "deepcas_results/gaming"),
            "embedding_dir": os.path.join(REDDIT_ROOT, "deepcas_embeddings/gaming"),
        },
        "windows": [0, 20, 50, 90],
        "root_only_reference_window": 20,
        "has_user_degrees": False,
        "root_score_column": "score",
    },
    "futurology": {
        "paths": {
            "thread_metadata": os.path.join(REDDIT_ROOT, "reddit_futurology_metadata.parquet"),
            "thread_posts": os.path.join(REDDIT_ROOT, "reddit_futurology_posts.parquet"),
            "user_degrees": None,
            "early_window_dir": os.path.join(REDDIT_ROOT, "snapshots/futurology"),
            "split_dir": os.path.join(REDDIT_ROOT, "snapshots/futurology/splits"),
            "future_horizon_dir": os.path.join(
                REDDIT_ROOT, "snapshots/futurology/future_horizons"
            ),
            "output_dir": os.path.join(REDDIT_ROOT, "deepcas_results/futurology"),
            "embedding_dir": os.path.join(REDDIT_ROOT, "deepcas_embeddings/futurology"),
        },
        "windows": [0, 30, 90, 180],
        "root_only_reference_window": 30,
        "has_user_degrees": False,
        "root_score_column": "score",
    },
    "ama": {
        "paths": {
            "thread_metadata": os.path.join(REDDIT_ROOT, "reddit_ama_metadata.parquet"),
            "thread_posts": os.path.join(REDDIT_ROOT, "reddit_ama_posts.parquet"),
            "user_degrees": None,
            "early_window_dir": os.path.join(REDDIT_ROOT, "snapshots/ama"),
            "split_dir": os.path.join(REDDIT_ROOT, "snapshots/ama/splits"),
            "future_horizon_dir": os.path.join(REDDIT_ROOT, "snapshots/ama/future_horizons"),
            "output_dir": os.path.join(REDDIT_ROOT, "deepcas_results/ama"),
            "embedding_dir": os.path.join(REDDIT_ROOT, "deepcas_embeddings/ama"),
        },
        "windows": [0, 15, 30, 60],
        "root_only_reference_window": 15,
        "has_user_degrees": False,
        "root_score_column": "score",
    },
}

# === RUN SELECTION ===
DATASETS_TO_RUN = ["bluesky"]
USE_FUTURE_HORIZONS = True

# === ACTIVE DATASET STATE ===
CURRENT_DATASET = None
BASE_DIR = None
THREAD_METADATA = None
THREAD_POSTS = None
USER_DEGREES = None
EARLY_WINDOW_DIR = None
SPLIT_DIR = None
FUTURE_HORIZON_DIR = None
OUTPUT_DIR = None
EMBEDDING_DIR = None
WINDOWS = []
ROOT_ONLY_REFERENCE_WINDOW = None
HAS_USER_DEGREES = False
ROOT_SCORE_COLUMN = None

# === OUTPUT TARGETS ===
OUTPUT_TARGETS = TARGET_NAMES.copy()

PRETRAINED_EMBEDDINGS = "user_embeddings.npy"
USER_VOCAB_MAP = "user_vocab.json"


def set_dataset(dataset_name):
    """Activate one dataset configuration."""
    global CURRENT_DATASET
    global THREAD_METADATA
    global THREAD_POSTS
    global USER_DEGREES
    global EARLY_WINDOW_DIR
    global SPLIT_DIR
    global FUTURE_HORIZON_DIR
    global OUTPUT_DIR
    global EMBEDDING_DIR
    global WINDOWS
    global ROOT_ONLY_REFERENCE_WINDOW
    global HAS_USER_DEGREES
    global ROOT_SCORE_COLUMN
    global BASE_DIR

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_cfg = DATASETS[dataset_name]
    paths = dataset_cfg["paths"]

    CURRENT_DATASET = dataset_name
    THREAD_METADATA = paths["thread_metadata"]
    THREAD_POSTS = paths["thread_posts"]
    USER_DEGREES = paths["user_degrees"]
    EARLY_WINDOW_DIR = paths["early_window_dir"]
    SPLIT_DIR = paths["split_dir"]
    FUTURE_HORIZON_DIR = paths["future_horizon_dir"]
    OUTPUT_DIR = paths["output_dir"]
    EMBEDDING_DIR = paths["embedding_dir"]
    WINDOWS = dataset_cfg["windows"]
    ROOT_ONLY_REFERENCE_WINDOW = dataset_cfg["root_only_reference_window"]
    HAS_USER_DEGREES = dataset_cfg["has_user_degrees"]
    ROOT_SCORE_COLUMN = dataset_cfg["root_score_column"]
    BASE_DIR = os.path.dirname(os.path.dirname(EARLY_WINDOW_DIR))


def get_embedding_paths():
    """Return dataset-specific embedding and vocab paths."""
    return {
        "embeddings": os.path.join(EMBEDDING_DIR, PRETRAINED_EMBEDDINGS),
        "vocab": os.path.join(EMBEDDING_DIR, USER_VOCAB_MAP),
    }


def get_embedding_paths_for_window(split_window):
    """Return embedding and vocab paths for a specific train split window."""
    window_dir = os.path.join(EMBEDDING_DIR, f"{split_window}min")
    return {
        "dir": window_dir,
        "embeddings": os.path.join(window_dir, PRETRAINED_EMBEDDINGS),
        "vocab": os.path.join(window_dir, USER_VOCAB_MAP),
        "metadata": os.path.join(window_dir, "user_embeddings_metadata.json"),
    }


# Set a default dataset for direct imports.
set_dataset(DATASETS_TO_RUN[0])


# === DEEPCAS HYPERPARAMETERS ===
K = 200
T = 10
P_JUMP = 0.05
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.4
FINAL_MLP_HIDDEN = 128

# === TRAINING ===
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EMBEDDING_LEARNING_RATE = 5e-4
MAX_EPOCHS = 200
PATIENCE = 10
RANDOM_SEED = 42

# === DEVICE ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
