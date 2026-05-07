"""
Feature preprocessing for DeepCas.
"""

import numpy as np

import config


class Preprocessor:
    """Preprocess optional global features and targets."""

    def __init__(self):
        self.available_targets = config.OUTPUT_TARGETS

    def fit(self, meta_df):
        """Fit scalers on training data."""
        print("\nFitting preprocessor...")
        print("Using path-only inputs; scalar global features are disabled.")

    def transform_global_features(self, meta_row, horizon_one_hot=None):
        """Transform global features for one tree/sample."""
        if config.USE_FUTURE_HORIZONS and horizon_one_hot is not None:
            return np.array(horizon_one_hot, dtype=np.float32)

        return np.array([], dtype=np.float32)

    def set_available_targets(self, targets):
        """Set which targets are available in the dataset."""
        self.available_targets = targets
