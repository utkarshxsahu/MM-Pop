"""
Example Configuration - Copy this and modify for your needs

This shows common configuration patterns for different use cases.
"""

# ==============================================================================
# EXAMPLE 1: Quick Test on Bluesky with Debug Mode
# ==============================================================================

# In config.py, set:
DATASETS_TO_RUN = ['bluesky']
GRID_SEARCH = False
SAVE_CHECKPOINTS = False
DEBUG_MODE = True
DEBUG_LIMIT_TREES = 100

HYPERPARAMETER_CONFIGS = [
    {
        'name': 'quick_test',
        'embedding_dim': 32,
        'hidden_dim': 32,
        'mlp_dim': 64,
        'num_bins': 5,
        'dropout': 0.5,
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'batch_size': 64,
    }
]

# ==============================================================================
# EXAMPLE 2: Full Training on Multiple Datasets
# ==============================================================================

# In config.py, set:
DATASETS_TO_RUN = ['bluesky', 'gaming', 'futurology', 'ama']
GRID_SEARCH = False
SAVE_CHECKPOINTS = True  # Save best models
DEBUG_MODE = False

HYPERPARAMETER_CONFIGS = [
    {
        'name': 'small_model',
        'embedding_dim': 32,
        'hidden_dim': 32,
        'mlp_dim': 64,
        'num_bins': 5,
        'dropout': 0.7,
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'batch_size': 64,
    },
    {
        'name': 'medium_model',
        'embedding_dim': 64,
        'hidden_dim': 64,
        'mlp_dim': 128,
        'num_bins': 5,
        'dropout': 0.5,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
    },
    {
        'name': 'large_model',
        'embedding_dim': 128,
        'hidden_dim': 128,
        'mlp_dim': 256,
        'num_bins': 5,
        'dropout': 0.3,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'batch_size': 32,
    },
]

# ==============================================================================
# EXAMPLE 3: Grid Search for Hyperparameter Tuning
# ==============================================================================

# In config.py, set:
DATASETS_TO_RUN = ['bluesky']
GRID_SEARCH = True
SAVE_CHECKPOINTS = False
DEBUG_MODE = False

HYPERPARAMETER_GRID = {
    'embedding_dim': [32, 64],
    'hidden_dim': [32, 64],
    'mlp_dim': [64, 128],
    'num_bins': [5],
    'dropout': [0.3, 0.5, 0.7],
    'lr': [5e-4, 1e-3],
    'weight_decay': [1e-4, 1e-3],
    'batch_size': [64],
}
# This will try 2×2×2×1×3×2×2×1 = 96 configurations

# ==============================================================================
# EXAMPLE 4: Production Run - Best Config from Tuning
# ==============================================================================

# After finding best config from grid search, run on all datasets:

DATASETS_TO_RUN = ['bluesky', 'gaming', 'futurology', 'ama']
GRID_SEARCH = False
SAVE_CHECKPOINTS = True
DEBUG_MODE = False

HYPERPARAMETER_CONFIGS = [
    {
        'name': 'best_config',  # Best config found from grid search
        'embedding_dim': 64,
        'hidden_dim': 64,
        'mlp_dim': 128,
        'num_bins': 5,
        'dropout': 0.5,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
    }
]

# ==============================================================================
# EXAMPLE 5: Comparing Different Architectures
# ==============================================================================

# Test different model sizes and regularization:

DATASETS_TO_RUN = ['bluesky']
GRID_SEARCH = False
SAVE_CHECKPOINTS = True
DEBUG_MODE = False

HYPERPARAMETER_CONFIGS = [
    {
        'name': 'shallow_wide',
        'embedding_dim': 128,
        'hidden_dim': 128,
        'mlp_dim': 512,
        'num_bins': 5,
        'dropout': 0.1,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'batch_size': 32,
    },
    {
        'name': 'deep_narrow',
        'embedding_dim': 32,
        'hidden_dim': 32,
        'mlp_dim': 64,
        'num_bins': 10,
        'dropout': 0.7,
        'lr': 5e-4,
        'weight_decay': 1e-3,
        'batch_size': 128,
    },
    {
        'name': 'balanced',
        'embedding_dim': 64,
        'hidden_dim': 64,
        'mlp_dim': 128,
        'num_bins': 5,
        'dropout': 0.5,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
    },
]

# ==============================================================================
# NOTES
# ==============================================================================

"""
Tips for choosing hyperparameters:

1. Start small for quick iteration:
   - embedding_dim, hidden_dim = 32
   - Use DEBUG_MODE = True with LIMIT_TREES = 100

2. Increase model capacity if underfitting:
   - Increase embedding_dim, hidden_dim, mlp_dim
   - Reduce dropout, weight_decay

3. Add regularization if overfitting:
   - Increase dropout (0.5 → 0.7)
   - Increase weight_decay (1e-4 → 1e-3)
   - Reduce model size

4. Learning rate:
   - 1e-3 is a good starting point
   - Try 5e-4 if training is unstable
   - Try 5e-4 with larger models

5. Batch size:
   - 64 is a good default
   - Reduce to 32 if OOM
   - Increase to 128 for faster training (if memory allows)

6. num_bins (time discretization):
   - 5 is typical
   - Try 10 for finer temporal resolution
   - Try 3 for simpler temporal modeling
"""
