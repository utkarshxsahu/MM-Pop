"""
Preprocessing: standardization, one-hot encoding, log transforms
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import config.feature_config as feat_cfg

class Preprocessor:
    """
    Handles all feature preprocessing:
    - Log1p transforms
    - Standardization
    - One-hot encoding
    """
    def __init__(self):
        self.scalers = {}
        self.fitted = False
    
    def fit(self, meta_df, posts_df):
        """
        Fit scalers on training data
        """
        print("\nFitting preprocessor on training data...")
        
        # Fit scalers for global features
        for group_name, features in feat_cfg.GLOBAL_FEATURES.items():
            for feat_name, config in features.items():
                if config.get('standardize', False):
                    if feat_name not in meta_df.columns:
                        print(f"  Skipping scaler for '{feat_name}' — not in dataset")
                        continue

                    values = meta_df[feat_name].values

                    # For log_follower_count, only fit on real values (non-zero)
                    # to avoid Reddit zero-fills skewing the scaler
                    if feat_name == 'log_follower_count':
                        real_values = values[values > 0]
                        if len(real_values) > 0:
                            values = real_values

                    if config.get('transform') == 'log1p':
                        values = np.log1p(values)
                    
                    scaler = StandardScaler()
                    scaler.fit(values.reshape(-1, 1))
                    self.scalers[f'global_{feat_name}'] = scaler
        
        # Fit scalers for node features
        for group_name, features in feat_cfg.NODE_FEATURES.items():
            for feat_name, config in features.items():
                if config.get('standardize', False):
                    values = posts_df[feat_name].values
                    scaler = StandardScaler()
                    scaler.fit(values.reshape(-1, 1))
                    self.scalers[f'node_{feat_name}'] = scaler
        
        self.fitted = True
        print(f"Fitted {len(self.scalers)} scalers")
    
    def transform_global_features(self, meta_row):
        """
        Transform global features for a single tree
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        features = []
        
        for group_name, feat_dict in feat_cfg.GLOBAL_FEATURES.items():
            for feat_name, config in feat_dict.items():
                feat_in_row = (feat_name in meta_row.index
                           if hasattr(meta_row, 'index')
                           else feat_name in meta_row)
                if not feat_in_row:
                    if config.get('type') == 'categorical':
                        features.append(np.zeros(config['num_classes']))
                    else:
                        features.append([0.0])
                    continue

                value = meta_row[feat_name]
                
                if config.get('type') == 'categorical':
                    num_classes = config['num_classes']
                    if pd.isna(value):
                        onehot = np.zeros(num_classes)
                    else:
                        value_int = int(value)
                        onehot = np.zeros(num_classes)
                        onehot[value_int] = 1.0
                    features.append(onehot)
                else:
                    if config.get('transform') == 'log1p':
                        value = np.log1p(value)
                    
                    if pd.isna(value) or np.isinf(value):
                        value = 0.0
                    
                    if config.get('standardize', False):
                        scaler = self.scalers[f'global_{feat_name}']
                        value = scaler.transform([[value]])[0, 0]
                    
                    features.append([value])    
        
        return np.concatenate(features).astype(np.float32)
    
    def transform_node_features(self, posts_df, embedding_array, exclude_features=None):
        """
        Transform node features for a tree.
        
        Args:
            posts_df: dataframe of posts
            embedding_array: numpy array of embeddings (or None if excluded)
            exclude_features: list of feature names to skip (e.g., ['text_embedding'])
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        if exclude_features is None:
            exclude_features = []

        features_list = []
        
        for group_name, feat_dict in feat_cfg.NODE_FEATURES.items():
            for feat_name, config in feat_dict.items():
                
                # SKIP if requested
                if feat_name in exclude_features:
                    continue

                if feat_name == 'text_embedding':
                    # Embeddings are already loaded, don't standardize
                    if embedding_array is not None:
                        features_list.append(embedding_array)
                
                elif config.get('type') == 'categorical':
                    num_classes = config['num_classes']
                    values = posts_df[feat_name].values
                    
                    if feat_name == 'emotion_label':
                        values = values - 1
                    
                    onehot = np.zeros((len(values), num_classes), dtype=np.float32)
                    for i, val in enumerate(values):
                        if not np.isnan(val):
                            onehot[i, int(val)] = 1.0
                    features_list.append(onehot)
                
                else:
                    values = posts_df[feat_name].values.reshape(-1, 1)
                    
                    if config.get('standardize', False):
                        scaler = self.scalers[f'node_{feat_name}']
                        values = scaler.transform(values)
                    
                    features_list.append(values.astype(np.float32))
        
        return np.concatenate(features_list, axis=1)
    
    def transform_targets(self, meta_row):
        targets = []
        for target_name in feat_cfg.OUTPUT_TARGETS:
            value = meta_row[target_name]
            targets.append(np.log1p(value))
        
        return np.array(targets, dtype=np.float32)
    
    def compute_aggregated_node_features(self, node_features):
        return node_features.mean(axis=0)