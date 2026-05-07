# deephawkes/__init__.py
"""
DeepHawkes Baseline Model for Social Network Cascade Prediction
"""

__version__ = "1.0.0"

from .model import DeepHawkes
from .data import DeepHawkesPathsDataset, collate_paths_batch
from .metrics import per_target_metrics

__all__ = [
    "DeepHawkes",
    "DeepHawkesPathsDataset",
    "collate_paths_batch",
    "per_target_metrics",
]
