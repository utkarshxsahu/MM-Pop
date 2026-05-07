"""
Preprocessing utilities for CasSeqGCN.
"""

import numpy as np


class Preprocessor:
    """Applies the target transform used during training."""

    def fit(self, meta_df):
        """Compatibility hook. No learned preprocessing is required."""
        return self

    @staticmethod
    def transform_targets(values):
        """Clip negatives to zero and apply log1p elementwise."""
        arr = np.asarray(values, dtype=np.float32)
        return np.log1p(np.clip(arr, a_min=0.0, a_max=None)).astype(np.float32)

    @staticmethod
    def inverse_transform_targets(values):
        """Invert the log1p transform and clip negative numerical drift."""
        arr = np.asarray(values, dtype=np.float32)
        return np.clip(np.expm1(arr), a_min=0.0, a_max=None).astype(np.float32)
