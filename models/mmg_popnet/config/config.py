"""
Main configuration file (Updated for Multi-Dataset Support + Multi-GPU)
"""

import os
import torch
from config.dataset_config import DATASETS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# --- DATASET SELECTION ---
#CURRENT_DATASET_NAME = "gaming"
DATASETS_TO_RUN =  ["bluesky", "gaming", "futurology", "ama"]
CURRENT_DATASET_NAME = DATASETS_TO_RUN[0]  # used for initial config load

DATASET_CONF = DATASETS[CURRENT_DATASET_NAME]
PATHS = DATASET_CONF['paths']

THREAD_METADATA = PATHS['thread_metadata']
THREAD_POSTS = PATHS['thread_posts']
USER_DEGREES = PATHS['user_degrees']
EARLY_WINDOW_DIR = PATHS['early_window_dir']
EMBEDDING_DIR = PATHS['embedding_dir']
OUTPUT_DIR = PATHS['output_dir']
IMAGE_DIR = PATHS.get('image_dir', None)

# --- CACHE SETTINGS ---
CACHE_DIR = None
MAIN_CACHE_BASE = os.path.join(PROJECT_ROOT, "results", "mmg_popnet", "feature_cache")
FORCE_RECOMPUTE = False

# --- IMAGE FEATURES ---
USE_IMAGE_FEATURES = True
FREEZE_IMAGE_ENCODER = False
UNFREEZE_LAST_N_LAYERS = 1
IMAGE_LR = 1e-6
IMAGE_PROJECTION_LR = 1e-3
IMAGE_DROPOUT = 0.3
IMAGE_WEIGHT_DECAY = 0.01

RAW_IMAGE_DIRS = {
    'gaming': DATASETS['gaming']['paths']['image_dir'],
    'futurology': DATASETS['futurology']['paths']['image_dir'],
    'ama': DATASETS['ama']['paths']['image_dir'],
}

EARLY_WINDOWS = DATASET_CONF['windows']

MODELS_TO_RUN =  ['text_graphsage']
#MODELS_TO_RUN =  ['mlp']

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# --- TRAINING REPRODUCIBILITY ---
# RANDOM_SEED above controls persistent train/val/test split creation.
# These seeds control model initialization, dropout, and train-loader shuffling.
USE_TRAINING_SEED = True
TRAINING_SEED = 42
MODALITY_ABLATION_SEED = 42
TRAINING_SEEDS = [42]
MODALITY_ABLATION_SEEDS = [42, 1042, 2042]

# --- NORMALIZED METRICS SUMMARIES ---
# One machine-readable JSON per runner invocation. These are additive and do
# not replace the detailed per-run/per-window results files.
WRITE_METRICS_SUMMARY_JSON = True
METRICS_SUMMARY_DIR = os.path.join(PROJECT_ROOT, "results", "mmg_popnet", "metrics_summaries")

MAX_EPOCHS = 200
PATIENCE = 10
BATCH_SIZE = 256

TUNING_EPOCHS = 10
N_TRIALS = 25
OPTUNA_TIMEOUT = 3600

# -----------------------------------------------------------------------
# DEVICE CONFIGURATION
# -----------------------------------------------------------------------
# Primary device — used for all models and as default for data loading.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Detect available GPUs
_N_GPUS = torch.cuda.device_count()
MULTI_GPU = _N_GPUS >= 2

if MULTI_GPU:
    # Model parallelism: transformer lives on GPU 0, GNN on GPU 1.
    # Only used by text_graphsage. All other models run entirely on DEVICE.
    DEVICE_TRANSFORMER = torch.device('cuda:0')
    DEVICE_GNN = torch.device('cuda:1')
    print(f"[config] Multi-GPU mode: {_N_GPUS} GPUs detected. "
          f"text_graphsage will use model parallelism "
          f"(transformer={DEVICE_TRANSFORMER}, GNN={DEVICE_GNN})")
else:
    DEVICE_TRANSFORMER = DEVICE
    DEVICE_GNN = DEVICE
    if torch.cuda.is_available():
        print(f"[config] Single-GPU mode: {torch.cuda.get_device_name(0)}")
    else:
        print("[config] No GPU detected — running on CPU")
# -----------------------------------------------------------------------

EMBEDDING_DIM = 384
EMBEDDING_PROJECTION_DIM = 32

# End-to-end projection sizes used by text_graphsage / trajectory models.
# TEXT_PROJECTION_DIM controls transformer output -> node-text features.
# IMAGE_PROJECTION_DIM controls CLIP/image embedding -> fused image features.
TEXT_PROJECTION_DIM = 32
IMAGE_PROJECTION_DIM = 64

# --- FUTURE HORIZONS ---
# Toggle: when True, model is trained/evaluated at 4h, 8h, 16h, 24h, and final horizons.
# When False (default), behaviour is identical to the original single-horizon code.
USE_FUTURE_HORIZONS = False
HORIZON_HOURS       = ['4h', '8h', '16h', '24h', 'final']
HORIZON_INDEX       = {'4h': 0, '8h': 1, '16h': 2, '24h': 3, 'final': 4}
N_HORIZONS          = 5
