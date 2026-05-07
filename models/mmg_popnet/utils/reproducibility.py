"""
Practical seeding helpers for experiment reproducibility.

This intentionally avoids strict deterministic algorithm settings because
PyG/AMP/DDP/transformer kernels may rely on nondeterministic CUDA operations.
"""

import random

import numpy as np


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch RNGs without forcing strict determinism."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_seed_list(seed_list, fallback_seed, use_seed=True):
    """Return configured training seeds, preserving scalar-seed compatibility."""
    if not use_seed:
        return [None]
    if seed_list:
        return list(seed_list)
    return [fallback_seed]
