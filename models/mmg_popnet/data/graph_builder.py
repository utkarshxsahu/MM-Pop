"""
Graph builder - creates PyTorch Geometric Data objects (Optimized)

Future-horizon support (USE_FUTURE_HORIZONS=True):
  Each tree expands to up to 5 samples (4h, 8h, 16h, 24h, final).
  A 5-class horizon one-hot is appended to global_features on the fly
  (the feature cache is NOT modified — no rebuild required).
  Data objects gain two extra fields:
    horizon_mask  : BoolTensor (1, n_targets) — False at root_score idx for non-final
    horizon_idx   : LongTensor (1,)           — index into cfg.HORIZON_HOURS
  Invalid tree-horizon pairs (valid_mask_df == False) are skipped entirely.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from torch.utils.data import Dataset
from tqdm import tqdm

import config.config as cfg
import config.feature_config as feat_cfg


_ZERO_IMAGE_EMBEDDING = torch.zeros(1, 512)
_ZERO_IMAGE_PIXELS = torch.zeros(1, 3, 224, 224)


class GraphDataset(Dataset):
    """
    Dataset for GraphSAGE/GAT/TextGraphSAGE models.
    Supports both Frozen (numeric x) and Unfrozen (input_ids + numeric x).
    Also supports optional image embeddings.

    When cfg.USE_FUTURE_HORIZONS is True, pass horizon_dfs and valid_mask_df
    (returned by load_future_horizon_data()) to enable multi-horizon augmentation.
    """
    def __init__(self, tree_ids, feature_cache,
                 horizon_dfs=None, valid_mask_df=None):
        self.tree_ids = [tid for tid in tree_ids if tid in feature_cache]

        use_horizons = cfg.USE_FUTURE_HORIZONS and horizon_dfs is not None

        # --- Pre-compute horizon lookup structures (done once, not per tree) ---
        if use_horizons:
            # Index valid_mask by tree_id for O(1) lookup
            _valid_mask = valid_mask_df.set_index('tree_id') \
                          if valid_mask_df.index.name != 'tree_id' \
                          else valid_mask_df
            # Each horizon_dfs[h] already indexed by tree_id from load_future_horizon_data
            _horizon_lkp = horizon_dfs

            n_targets      = len(feat_cfg.OUTPUT_TARGETS)
            rs_idx         = feat_cfg.ROOT_SCORE_IDX
            # horizon_mask tensors (shared across all samples of same type)
            _final_mask        = torch.ones(n_targets, dtype=torch.bool)
            _intermediate_mask = torch.ones(n_targets, dtype=torch.bool)
            _intermediate_mask[rs_idx] = False

            # Precompute (gt_col, target_idx) pairs for intermediate horizon assembly
            _gt_target_pairs = [
                (gt_col, feat_cfg.OUTPUT_TARGETS.index(tgt))
                for gt_col, tgt in feat_cfg.HORIZON_GT_COL_MAP.items()
            ]

        print(f"Pre-converting {len(self.tree_ids)} graphs to PyG Data objects"
              + (" [horizon-augmented]" if use_horizons else "") + "...")
        self.data_list = []
        n_skipped = 0

        for tree_id in tqdm(self.tree_ids, desc="Building Graph Dataset"):
            data_dict = feature_cache[tree_id]

            # --- Graph structure (identical across all horizons for this tree) ---
            x = torch.FloatTensor(data_dict['node_features'])

            input_ids     = None
            attention_mask = None
            if 'input_ids' in data_dict:
                input_ids = torch.LongTensor(data_dict['input_ids'])
            if 'attention_mask' in data_dict:
                attention_mask = torch.LongTensor(data_dict['attention_mask'])

            has_image = data_dict.get('has_image', False)
            if has_image:
                if data_dict.get('image_embedding') is not None:
                    image_embedding = torch.FloatTensor(data_dict['image_embedding']).unsqueeze(0)
                    image_pixels    = _ZERO_IMAGE_PIXELS
                elif data_dict.get('image_pixels') is not None:
                    image_pixels    = data_dict['image_pixels'].unsqueeze(0)
                    image_embedding = _ZERO_IMAGE_EMBEDDING
                else:
                    image_embedding = _ZERO_IMAGE_EMBEDDING
                    image_pixels    = _ZERO_IMAGE_PIXELS
                    has_image       = False
            else:
                image_embedding = _ZERO_IMAGE_EMBEDDING
                image_pixels    = _ZERO_IMAGE_PIXELS

            edge_index = torch.LongTensor(data_dict['edge_index'])
            edge_index = sort_edge_index(edge_index, sort_by_row=False)
            if edge_index.size(1) > 0:
                edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
                edge_index_rev = sort_edge_index(edge_index_rev, sort_by_row=False)
            else:
                edge_index_rev = torch.LongTensor([[], []])

            root_mask   = self._get_root_mask(data_dict)
            has_img_t   = torch.tensor([has_image], dtype=torch.bool)

            # Base global features from cache (without horizon one-hot)
            base_global = torch.FloatTensor(data_dict['global_features'])  # (global_dim,)

            if not use_horizons:
                # ---- Original single-sample path (default behaviour) ----
                global_feat = base_global.unsqueeze(0)                     # (1, global_dim)
                y           = torch.FloatTensor(data_dict['targets']).unsqueeze(0)
                data = Data(
                    x=x, edge_index=edge_index, edge_index_rev=edge_index_rev,
                    y=y, global_features=global_feat, root_mask=root_mask,
                    tree_id=tree_id, input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embedding=image_embedding, image_pixels=image_pixels,
                    has_image=has_img_t
                )
                self.data_list.append(data)

            else:
                # ---- Horizon-augmented path: one sample per valid horizon ----
                for h in cfg.HORIZON_HOURS:
                    h_idx = cfg.HORIZON_INDEX[h]

                    # Validity check for intermediate horizons
                    if h != 'final':
                        valid_col = f'valid_{h}'
                        if tree_id not in _valid_mask.index:
                            n_skipped += 1
                            continue
                        if not _valid_mask.loc[tree_id, valid_col]:
                            n_skipped += 1
                            continue

                    # Build targets
                    if h == 'final':
                        y_vals = np.array(data_dict['targets'], dtype=np.float32)
                        hmask  = _final_mask
                    else:
                        y_vals = np.zeros(len(feat_cfg.OUTPUT_TARGETS), dtype=np.float32)
                        if tree_id not in _horizon_lkp[h].index:
                            n_skipped += 1
                            continue
                        row = _horizon_lkp[h].loc[tree_id]
                        for gt_col, t_idx in _gt_target_pairs:
                            if gt_col in row.index:
                                y_vals[t_idx] = float(row[gt_col])
                        hmask = _intermediate_mask

                    # Append horizon one-hot to global features
                    h_oh        = torch.zeros(cfg.N_HORIZONS)
                    h_oh[h_idx] = 1.0
                    global_feat = torch.cat([base_global, h_oh]).unsqueeze(0)  # (1, global_dim+5)

                    y = torch.FloatTensor(y_vals).unsqueeze(0)  # (1, n_targets)

                    data = Data(
                        x=x, edge_index=edge_index, edge_index_rev=edge_index_rev,
                        y=y, global_features=global_feat, root_mask=root_mask,
                        tree_id=tree_id, input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_embedding=image_embedding, image_pixels=image_pixels,
                        has_image=has_img_t,
                        # Horizon-specific fields
                        horizon_mask=hmask.unsqueeze(0),                          # (1, n_targets)
                        horizon_idx=torch.tensor([h_idx], dtype=torch.long)       # (1,)
                    )
                    self.data_list.append(data)

        if use_horizons:
            print(f"  Horizon augmentation: {len(self.data_list)} samples "
                  f"from {len(self.tree_ids)} trees "
                  f"({n_skipped} tree-horizon pairs skipped as invalid)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def _get_root_mask(self, data):
        mask = torch.zeros(data['num_nodes'], dtype=torch.bool)
        mask[data['root_node_idx']] = True
        return mask


class MLPDataset(Dataset):
    """
    Dataset for MLP model.

    When cfg.USE_FUTURE_HORIZONS is True, each tree expands to up to 5 samples.
    The horizon one-hot is appended to the feature vector.
    __getitem__ returns:
      - Default  : (x, y)
      - Horizons : (x, y, horizon_mask, horizon_idx)
    """
    def __init__(self, tree_ids, feature_cache, preprocessor,
                 horizon_dfs=None, valid_mask_df=None):
        valid_ids    = [tid for tid in tree_ids if tid in feature_cache]
        use_horizons = cfg.USE_FUTURE_HORIZONS and horizon_dfs is not None

        if use_horizons:
            _valid_mask = valid_mask_df.set_index('tree_id') \
                          if valid_mask_df.index.name != 'tree_id' \
                          else valid_mask_df
            _horizon_lkp = horizon_dfs
            n_targets    = len(feat_cfg.OUTPUT_TARGETS)
            rs_idx       = feat_cfg.ROOT_SCORE_IDX
            _final_mask        = torch.ones(n_targets, dtype=torch.bool)
            _intermediate_mask = torch.ones(n_targets, dtype=torch.bool)
            _intermediate_mask[rs_idx] = False
            _gt_target_pairs = [
                (gt_col, feat_cfg.OUTPUT_TARGETS.index(tgt))
                for gt_col, tgt in feat_cfg.HORIZON_GT_COL_MAP.items()
            ]

        print(f"Pre-computing MLP tensors for {len(valid_ids)} trees"
              + (" [horizon-augmented]" if use_horizons else "") + "...")

        features_list  = []
        targets_list   = []
        hmask_list     = []
        hidx_list      = []
        n_skipped      = 0

        for tree_id in tqdm(valid_ids, desc="Building MLP Dataset"):
            data = feature_cache[tree_id]
            root_feat = data['node_features'][data['root_node_idx']]
            agg_feat  = preprocessor.compute_aggregated_node_features(data['node_features'])
            base_feat = np.concatenate([root_feat, agg_feat, data['global_features']]).astype(np.float32)

            if not use_horizons:
                features_list.append(base_feat)
                targets_list.append(data['targets'])
            else:
                for h in cfg.HORIZON_HOURS:
                    h_idx = cfg.HORIZON_INDEX[h]
                    if h != 'final':
                        valid_col = f'valid_{h}'
                        if tree_id not in _valid_mask.index or \
                                not _valid_mask.loc[tree_id, valid_col]:
                            n_skipped += 1
                            continue
                    if h == 'final':
                        y_vals = np.array(data['targets'], dtype=np.float32)
                        hmask  = _final_mask
                    else:
                        y_vals = np.zeros(len(feat_cfg.OUTPUT_TARGETS), dtype=np.float32)
                        if tree_id not in _horizon_lkp[h].index:
                            n_skipped += 1
                            continue
                        row = _horizon_lkp[h].loc[tree_id]
                        for gt_col, t_idx in _gt_target_pairs:
                            if gt_col in row.index:
                                y_vals[t_idx] = float(row[gt_col])
                        hmask = _intermediate_mask
                    h_oh   = np.zeros(cfg.N_HORIZONS, dtype=np.float32)
                    h_oh[h_idx] = 1.0
                    feat   = np.concatenate([base_feat, h_oh])
                    features_list.append(feat)
                    targets_list.append(y_vals)
                    hmask_list.append(hmask.numpy())
                    hidx_list.append(h_idx)

        self.use_horizons = use_horizons

        if features_list:
            self.x = torch.FloatTensor(np.stack(features_list))
            self.y = torch.FloatTensor(np.stack(targets_list))
            if use_horizons:
                self.horizon_mask = torch.BoolTensor(np.stack(hmask_list))
                self.horizon_idx  = torch.LongTensor(hidx_list)
                print(f"  Horizon augmentation: {len(self.x)} MLP samples "
                      f"({n_skipped} tree-horizon pairs skipped)")
        else:
            self.x = torch.FloatTensor([])
            self.y = torch.FloatTensor([])
            if use_horizons:
                self.horizon_mask = torch.BoolTensor([])
                self.horizon_idx  = torch.LongTensor([])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.use_horizons:
            return self.x[idx], self.y[idx], self.horizon_mask[idx], self.horizon_idx[idx]
        return self.x[idx], self.y[idx]
