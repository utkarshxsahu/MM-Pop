"""
Experiment runner — DDP-aware.

Location: training/experiment_runner.py

Single GPU:  python main.py
4-GPU DDP:   torchrun --nproc_per_node=4 main.py

Root-only condition:
  Add 0 to a dataset's windows list in dataset_config.py to enable root-only.
  Examples:
      'windows': [0, 20, 50, 90]  → root-only + cascade windows
      'windows': [0]              → root-only only (needs root_only_reference_window)

  When window_minutes == 0:
    - The reference window's data is loaded (posts, meta, splits)
    - Reference window priority: first non-zero in windows[] >
                                 root_only_reference_window in dataset_config
    - RootOnlyFeatureBuilder / RootOnlyTokenizedFeatureBuilder are used
    - Results saved to 0min/ directory
    - Separate independent model trained for root-only

Batch size configuration (text_graphsage only):
  TEXT_GRAPHSAGE_BATCH_SIZE    = graphs per GPU per step
  TEXT_GRAPHSAGE_EFFECTIVE_BATCH = target effective batch
  accumulation_steps = max(1, EFFECTIVE_BATCH // (BATCH_SIZE * world_size))
"""

import os
import json
import math
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.data_loader import load_all_data, load_future_horizon_data
from data.embedding_loader import EmbeddingLoader
from data.image_loader import ImageEmbeddingLoader, ImagePixelLoader
from data.preprocessing import Preprocessor
from data.feature_builder import (FeatureBuilder, TokenizedFeatureBuilder,
                                   RootOnlyFeatureBuilder,
                                   RootOnlyTokenizedFeatureBuilder)
from data.graph_builder import GraphDataset, MLPDataset
from data.feature_cache import precompute_features
from models.model_factory import create_model
from training.trainer import train_model, evaluate_text_graphsage
from training.evaluator import (get_predictions_mlp, get_predictions_graphsage,
                                 get_predictions_mlp_with_horizon,
                                 get_predictions_graphsage_with_horizon,
                                 evaluate_predictions, evaluate_predictions_by_horizon,
                                 save_horizon_predictions)
from utils.visualization import (plot_training_curves, plot_prediction_scatter,
                                  plot_prediction_scatter_linear, plot_residuals)
from utils.logger import Logger
from utils.metrics_summary import (add_metric_records, add_source,
                                   create_summary, write_summary)
from utils.reproducibility import resolve_seed_list, seed_everything

import config.config as cfg
import config.feature_config as feat_cfg

# ============================================================================
# BATCH SIZE CONFIGURATION
# Only affects text_graphsage — graphsage/gat/mlp use cfg.BATCH_SIZE directly.
# ============================================================================
TEXT_GRAPHSAGE_BATCH_SIZE     = 16
TEXT_GRAPHSAGE_EFFECTIVE_BATCH = 256

USE_OPTUNA = False

CANONICAL_PARAMS = {
    'mlp': {
        'hidden_dims': [128, 128],
        'dropout': 0.3,
        'lr': 1e-3
    },
    'graphsage': {
        'num_layers': 3,
        'hidden_dim': 128,
        'aggregator_type': 'mean',
        'dropout': 0.3,
        # Previous shared-head baseline kept for reference:
        # final_mlp_layers=2, final_mlp_hidden=128
        'final_mlp_layers': 3,
        'final_mlp_hidden': 256,
        'lr': 1e-3,
        'use_images': None
    },
    'gat': {
        'num_layers': 3,
        'hidden_dim': 32,
        'heads': 4,
        'dropout': 0.3,
        'final_mlp_layers': 2,
        'final_mlp_hidden': 64,
        'lr': 1e-3,
        'use_images': None
    },
    'text_graphsage': {
        'num_layers': 3,
        'hidden_dim': 128,
        'aggregator_type': 'mean',
        'dropout': 0.3,
        # Previous shared-head baseline kept for reference:
        # final_mlp_layers=2, final_mlp_hidden=128
        'final_mlp_layers': 2,
        'final_mlp_hidden': 128,
        'transformer_lr': 5e-6,
        'gnn_lr': 1e-3,
        'use_images': None
    }
}

TEXT_GRAPHSAGE_SCHEDULER = {
    'type': 'linear_warmup_cosine_decay',
    'warmup_ratio': 0.05,
    'min_lr_factor': 0.0,
}


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_ddp():
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_ddp():
        dist.barrier()


def _build_cache(tree_ids, edges, meta, posts, builder,
                 cache_path, desc, rank, force_recompute):
    """Rank 0 builds; others barrier-wait then load."""
    if rank == 0:
        cache = precompute_features(
            tree_ids, edges, meta, posts, builder,
            cache_path, desc=desc, force_recompute=force_recompute
        )
    _barrier()
    if rank != 0:
        cache = precompute_features(
            tree_ids, edges, meta, posts, builder,
            cache_path, desc=desc, force_recompute=False
        )
    return cache


def _seeded_generator(seed):
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _build_pyg_loader(dataset, batch_size, shuffle, rank, world_size, seed=None):
    """PyG DataLoader; DistributedSampler when world_size > 1."""
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle,
            seed=seed or 0
        )
        return PyGDataLoader(dataset, batch_size=batch_size,
                             sampler=sampler, num_workers=0)
    generator = _seeded_generator(seed) if shuffle else None
    return PyGDataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=0,
                         generator=generator)


def _compute_accum_steps(model_type, world_size):
    """Accumulation steps for text_graphsage only. Returns 1 for others."""
    if model_type != 'text_graphsage':
        return 1
    return max(1, TEXT_GRAPHSAGE_EFFECTIVE_BATCH //
                  (TEXT_GRAPHSAGE_BATCH_SIZE * world_size))


def _build_text_scheduler(optimizer, train_loader, accumulation_steps,
                          max_epochs):
    """
    Linear warmup followed by cosine decay, stepped once per optimizer update.

    The optimizer update count differs from the raw batch count because
    TextGraphSAGE uses gradient accumulation.
    """
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
    total_steps = max(1, max_epochs * steps_per_epoch)
    warmup_ratio = TEXT_GRAPHSAGE_SCHEDULER['warmup_ratio']
    warmup_steps = int(math.ceil(total_steps * warmup_ratio))
    if total_steps <= 1:
        warmup_steps = 0
    else:
        warmup_steps = min(max(1, warmup_steps), total_steps - 1)

    min_lr_factor = TEXT_GRAPHSAGE_SCHEDULER['min_lr_factor']

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(current_step - warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler_config = {
        **TEXT_GRAPHSAGE_SCHEDULER,
        'steps_per_epoch': steps_per_epoch,
        'total_optimizer_steps': total_steps,
        'warmup_steps': warmup_steps,
    }
    return scheduler, scheduler_config


# ---------------------------------------------------------------------------
# Utilities (imported by foundational_runner.py)
# ---------------------------------------------------------------------------

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


def save_test_predictions(y_true, y_pred, target_names, output_dir, model_name):
    rows = []
    for i, target in enumerate(target_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        for j in range(len(yt)):
            rows.append({
                'target': target,
                'y_true_log': yt[j],    'y_pred_log': yp[j],
                'y_true_actual': np.expm1(yt[j]),
                'y_pred_actual': np.expm1(yp[j])
            })
    path = os.path.join(output_dir, f'predictions_{model_name}.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n✓ Saved predictions: {path}")


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(window_minutes, model_type, data_dict, embedding_loader,
                   preprocessor, output_dir, cache_dir, logger,
                   image_loader=None, rank=0, world_size=1,
                   is_root_only=False,
                   horizon_dfs=None, valid_mask_df=None):
    """
    Run one experiment for a given window and model type.

    is_root_only=True: uses RootOnly*FeatureBuilder.
    window_minutes=0 in labels/filenames indicates root-only condition.
    All other logic is identical to cascade experiments.
    """
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir,  exist_ok=True)
        label = "ROOT-ONLY" if is_root_only else f"{window_minutes}MIN"
        logger.section(f"EXPERIMENT: {model_type.upper()} - {label}")
        logger.log(
            f"Training seed: "
            f"{cfg.TRAINING_SEED if cfg.USE_TRAINING_SEED else 'disabled'}"
        )
    _barrier()

    # ------------------------------------------------------------------
    # 1. Feature builder
    # ------------------------------------------------------------------
    if is_root_only:
        if model_type == 'text_graphsage':
            if rank == 0:
                logger.log("Using RootOnlyTokenizedFeatureBuilder")
            feature_builder = RootOnlyTokenizedFeatureBuilder(
                preprocessor, data_dict['posts'],
                data_dict['user_degrees'], image_loader=image_loader
            )
        else:
            if rank == 0:
                logger.log("Using RootOnlyFeatureBuilder")
            feature_builder = RootOnlyFeatureBuilder(
                embedding_loader, preprocessor, data_dict['posts'],
                data_dict['user_degrees'], image_loader=image_loader
            )
    else:
        if model_type == 'text_graphsage':
            if rank == 0:
                logger.log("Using TokenizedFeatureBuilder")
            feature_builder = TokenizedFeatureBuilder(
                preprocessor, data_dict['posts'],
                data_dict['user_degrees'], image_loader=image_loader
            )
        else:
            if rank == 0:
                logger.log("Using FeatureBuilder (frozen embeddings)")
            feature_builder = FeatureBuilder(
                embedding_loader, preprocessor, data_dict['posts'],
                data_dict['user_degrees'], image_loader=image_loader
            )

    meta_indexed = data_dict['meta'].set_index('tree_id')

    # ------------------------------------------------------------------
    # 2. Feature caches
    # ------------------------------------------------------------------
    train_cache = _build_cache(
        data_dict['train_ids'], data_dict['edges'], meta_indexed,
        data_dict['posts'], feature_builder,
        os.path.join(cache_dir, 'train_features.pkl'),
        desc="Train", rank=rank, force_recompute=cfg.FORCE_RECOMPUTE
    )
    val_cache = _build_cache(
        data_dict['val_ids'], data_dict['edges'], meta_indexed,
        data_dict['posts'], feature_builder,
        os.path.join(cache_dir, 'val_features.pkl'),
        desc="Val", rank=rank, force_recompute=cfg.FORCE_RECOMPUTE
    )
    test_cache = _build_cache(
        data_dict['test_ids'], data_dict['edges'], meta_indexed,
        data_dict['posts'], feature_builder,
        os.path.join(cache_dir, 'test_features.pkl'),
        desc="Test", rank=rank, force_recompute=cfg.FORCE_RECOMPUTE
    )

    # ------------------------------------------------------------------
    # 3. Datasets and DataLoaders
    # ------------------------------------------------------------------
    if rank == 0:
        logger.log("\nCreating datasets...")

    # Horizon data (None when USE_FUTURE_HORIZONS=False — no-op)
    _h_dfs  = horizon_dfs   if cfg.USE_FUTURE_HORIZONS else None
    _v_mask = valid_mask_df if cfg.USE_FUTURE_HORIZONS else None

    if model_type == 'mlp':
        train_dataset = MLPDataset(data_dict['train_ids'], train_cache, preprocessor,
                                   horizon_dfs=_h_dfs, valid_mask_df=_v_mask)
        val_dataset   = MLPDataset(data_dict['val_ids'],   val_cache,   preprocessor,
                                   horizon_dfs=_h_dfs, valid_mask_df=_v_mask)
        test_dataset  = MLPDataset(data_dict['test_ids'],  test_cache,  preprocessor,
                                   horizon_dfs=_h_dfs, valid_mask_df=_v_mask)

        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                seed=cfg.TRAINING_SEED if cfg.USE_TRAINING_SEED else 0)
            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                                      sampler=train_sampler, num_workers=0)
        else:
            generator = (_seeded_generator(cfg.TRAINING_SEED)
                         if cfg.USE_TRAINING_SEED else None)
            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                                      shuffle=True, num_workers=0,
                                      generator=generator)
        val_loader  = DataLoader(val_dataset,  batch_size=cfg.BATCH_SIZE,
                                 shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                                 shuffle=False, num_workers=0)

        sample_batch = next(iter(train_loader))
        input_dim    = sample_batch[0].shape[1]
        output_dim  = len(feat_cfg.OUTPUT_TARGETS)

    elif model_type in ['graphsage', 'gat']:
        train_dataset = GraphDataset(data_dict['train_ids'], train_cache,
                                     horizon_dfs=_h_dfs, valid_mask_df=_v_mask)
        val_dataset   = GraphDataset(data_dict['val_ids'],   val_cache,
                                     horizon_dfs=_h_dfs, valid_mask_df=_v_mask)
        test_dataset  = GraphDataset(data_dict['test_ids'],  test_cache,
                                     horizon_dfs=_h_dfs, valid_mask_df=_v_mask)

        train_loader = _build_pyg_loader(
            train_dataset, cfg.BATCH_SIZE, shuffle=True,
            rank=rank, world_size=world_size,
            seed=cfg.TRAINING_SEED if cfg.USE_TRAINING_SEED else None)
        val_loader   = _build_pyg_loader(val_dataset,   cfg.BATCH_SIZE,
                                         shuffle=False, rank=rank, world_size=world_size)
        test_loader  = PyGDataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                                     shuffle=False, num_workers=0)

        sb = next(iter(train_loader))
        node_feat_dim   = sb.x.shape[1]
        global_feat_dim = sb.global_features.shape[1]
        output_dim      = sb.y.shape[1]

    elif model_type == 'text_graphsage':
        train_dataset = GraphDataset(data_dict['train_ids'], train_cache,
                                     horizon_dfs=_h_dfs, valid_mask_df=_v_mask)
        val_dataset   = GraphDataset(data_dict['val_ids'],   val_cache,
                                     horizon_dfs=_h_dfs, valid_mask_df=_v_mask)
        test_dataset  = GraphDataset(data_dict['test_ids'],  test_cache,
                                     horizon_dfs=_h_dfs, valid_mask_df=_v_mask)

        train_loader = _build_pyg_loader(
            train_dataset, TEXT_GRAPHSAGE_BATCH_SIZE, shuffle=True,
            rank=rank, world_size=world_size,
            seed=cfg.TRAINING_SEED if cfg.USE_TRAINING_SEED else None)
        val_loader   = _build_pyg_loader(val_dataset,   TEXT_GRAPHSAGE_BATCH_SIZE,
                                         shuffle=False, rank=rank, world_size=world_size)
        test_loader  = PyGDataLoader(test_dataset,
                                     batch_size=TEXT_GRAPHSAGE_BATCH_SIZE,
                                     shuffle=False, num_workers=0)

        sb = next(iter(train_loader))
        structural_feat_dim = sb.x.shape[1]
        global_feat_dim     = sb.global_features.shape[1]
        output_dim          = sb.y.shape[1]

    # ------------------------------------------------------------------
    # 4. Accumulation steps
    # ------------------------------------------------------------------
    accum_steps = _compute_accum_steps(model_type, world_size)

    if rank == 0 and model_type == 'text_graphsage':
        logger.log(f"\nBatch config:")
        logger.log(f"  per-GPU batch      = {TEXT_GRAPHSAGE_BATCH_SIZE}")
        logger.log(f"  world_size         = {world_size}")
        logger.log(f"  accumulation_steps = {accum_steps}")
        logger.log(f"  effective batch    = "
                   f"{TEXT_GRAPHSAGE_BATCH_SIZE * world_size * accum_steps}")

    # ------------------------------------------------------------------
    # 5. Model and optimizer
    # ------------------------------------------------------------------
    canonical  = CANONICAL_PARAMS[model_type]
    use_images = (cfg.USE_IMAGE_FEATURES and
                  cfg.DATASET_CONF.get('has_images', False) and
                  image_loader is not None and
                  model_type in ['graphsage', 'gat', 'text_graphsage'])

    if rank == 0:
        logger.log(f"\nuse_images = {use_images}")

    if model_type == 'mlp':
        model_params = {
            'input_dim':   input_dim,
            'output_dim':  output_dim,
            'hidden_dims': canonical['hidden_dims'],
            'dropout':     canonical['dropout']
        }
        lr = canonical['lr']

    elif model_type in ['graphsage', 'gat']:
        model_params = {
            'node_feat_dim':          node_feat_dim,
            'global_feat_dim':        global_feat_dim,
            'output_dim':             output_dim,
            'num_layers':             canonical['num_layers'],
            'hidden_dim':             canonical['hidden_dim'],
            'aggregator_type':        canonical.get('aggregator_type', 'mean'),
            'heads':                  canonical.get('heads', 4),
            'dropout':                canonical['dropout'],
            'final_mlp_layers':       canonical['final_mlp_layers'],
            'final_mlp_hidden':       canonical['final_mlp_hidden'],
            'use_images':             use_images,
            'image_projection_dim':   cfg.IMAGE_PROJECTION_DIM,
            'freeze_image_encoder':   cfg.FREEZE_IMAGE_ENCODER,
            'unfreeze_last_n_layers': cfg.UNFREEZE_LAST_N_LAYERS
        }
        lr = canonical['lr']

    elif model_type == 'text_graphsage':
        model_params = {
            'structural_feat_dim':    structural_feat_dim,
            'global_feat_dim':        global_feat_dim,
            'output_dim':             output_dim,
            'num_layers':             canonical['num_layers'],
            'hidden_dim':             canonical['hidden_dim'],
            'aggregator_type':        canonical['aggregator_type'],
            'dropout':                canonical['dropout'],
            'final_mlp_layers':       canonical['final_mlp_layers'],
            'final_mlp_hidden':       canonical['final_mlp_hidden'],
            'transformer_name':       'sentence-transformers/all-MiniLM-L6-v2',
            'use_images':             use_images,
            'text_projection_dim':    cfg.TEXT_PROJECTION_DIM,
            'text_projection_type':   'bottleneck_mlp',
            'text_projection_hidden': 64,
            'text_projection_dropout': 0.15,
            'image_projection_dim':   cfg.IMAGE_PROJECTION_DIM,
            'freeze_image_encoder':   cfg.FREEZE_IMAGE_ENCODER,
            'unfreeze_last_n_layers': cfg.UNFREEZE_LAST_N_LAYERS
        }
        transformer_lr = canonical['transformer_lr']
        gnn_lr         = canonical['gnn_lr']

    model = create_model(model_type, model_params, cfg.DEVICE)
    if rank == 0:
        logger.log(f"Model parameters: {model.get_num_parameters():,}")

    if model_type == 'text_graphsage':
        if (use_images and not cfg.FREEZE_IMAGE_ENCODER and
                hasattr(model.gnn, 'clip_model') and model.gnn.clip_model is not None):
            clip_params     = [p for p in model.gnn.clip_model.parameters()
                               if p.requires_grad]
            proj_params     = list(model.gnn.image_projection.parameters())
            exclude_ids     = set(id(p) for p in clip_params) | \
                              set(id(p) for p in proj_params)
            gnn_only_params = [p for p in model.gnn.parameters()
                               if id(p) not in exclude_ids]
            param_groups = [
                {'params': model.transformer.parameters(), 'lr': transformer_lr},
                {'params': gnn_only_params,                'lr': gnn_lr},
                {'params': clip_params, 'lr': cfg.IMAGE_LR,
                 'weight_decay': cfg.IMAGE_WEIGHT_DECAY},
                {'params': proj_params, 'lr': cfg.IMAGE_PROJECTION_LR,
                 'weight_decay': cfg.IMAGE_WEIGHT_DECAY}
            ]
        else:
            param_groups = [
                {'params': model.transformer.parameters(), 'lr': transformer_lr},
                {'params': model.gnn.parameters(),         'lr': gnn_lr}
            ]
        optimizer = torch.optim.Adam(param_groups)

    elif model_type in ['graphsage', 'gat']:
        if (use_images and not cfg.FREEZE_IMAGE_ENCODER and
                hasattr(model, 'clip_model') and model.clip_model is not None):
            gnn_ps  = [p for n, p in model.named_parameters()
                       if not n.startswith('clip_model') and
                          not n.startswith('image_projection')]
            clip_ps = [p for p in model.clip_model.parameters() if p.requires_grad]
            proj_ps = list(model.image_projection.parameters())
            param_groups = [
                {'params': gnn_ps,  'lr': lr},
                {'params': clip_ps, 'lr': cfg.IMAGE_LR,
                 'weight_decay': cfg.IMAGE_WEIGHT_DECAY},
                {'params': proj_ps, 'lr': cfg.IMAGE_PROJECTION_LR,
                 'weight_decay': cfg.IMAGE_WEIGHT_DECAY}
            ]
        else:
            param_groups = [{'params': model.parameters(), 'lr': lr}]
        optimizer = torch.optim.Adam(param_groups)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    if rank == 0:
        logger.log("\n--- Training ---")
    max_epochs = 100 if model_type == 'text_graphsage' else cfg.MAX_EPOCHS
    scheduler = None
    scheduler_config = None
    if model_type == 'text_graphsage':
        scheduler, scheduler_config = _build_text_scheduler(
            optimizer, train_loader, accum_steps, max_epochs
        )
        if rank == 0:
            logger.log(f"Scheduler: {scheduler_config}")

    model, history = train_model(
        model, model_type, train_loader, val_loader,
        optimizer, cfg.DEVICE, max_epochs=max_epochs,
        rank=rank, world_size=world_size,
        accumulation_steps=accum_steps,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate and save — rank 0 only
    # ------------------------------------------------------------------
    _barrier()

    if rank == 0:
        logger.log("\n--- Test Set Evaluation ---")

        label = "root_only" if is_root_only else f"{window_minutes}min"

        if cfg.USE_FUTURE_HORIZONS:
            # Collect predictions with horizon metadata
            if model_type == 'mlp':
                y_pred, y_true, horizon_idxs = get_predictions_mlp_with_horizon(
                    model, test_loader, cfg.DEVICE)
                tree_ids = None  # MLP dataset does not track tree_ids
            else:
                y_pred, y_true, tree_ids, horizon_idxs = \
                    get_predictions_graphsage_with_horizon(model, test_loader, cfg.DEVICE)

            # Overall metrics (all horizons combined, using full target set)
            test_results = evaluate_predictions(y_true, y_pred, feat_cfg.OUTPUT_TARGETS)

            # Per-horizon metrics
            horizon_metrics = evaluate_predictions_by_horizon(y_true, y_pred, horizon_idxs)

            # Save per-tree, per-horizon CSV
            if tree_ids is not None:
                hz_csv = os.path.join(output_dir,
                                      f'horizon_predictions_{model_type}_{label}.csv')
                save_horizon_predictions(y_true, y_pred, tree_ids, horizon_idxs, hz_csv)

            # Log per-horizon summary
            METRICS_TO_LOG = ('MSE', 'R2', 'Spearman')
            for h_name, h_res in horizon_metrics.items():
                logger.log(f"\n  [HORIZON: {h_name}]")
                for target, m in h_res.items():
                    if target in ('group_summaries',):
                        continue
                    logger.log(f"    {target}: " +
                               "  ".join(f"{k}={m[k]:.3f}"
                                         for k in METRICS_TO_LOG if k in m))
        else:
            if model_type == 'mlp':
                y_pred, y_true = get_predictions_mlp(model, test_loader, cfg.DEVICE)
            else:
                y_pred, y_true = get_predictions_graphsage(model, test_loader, cfg.DEVICE)

            test_results   = evaluate_predictions(y_true, y_pred, feat_cfg.OUTPUT_TARGETS)
            horizon_metrics = {}

            METRICS_TO_LOG = ('MSE', 'R2', 'Spearman')
            for group_name, group_targets in feat_cfg.TARGET_GROUPS.items():
                logger.log(f"\n  [{group_name.upper()}]")
                for target in group_targets:
                    if target not in test_results:
                        continue
                    m = test_results[target]
                    logger.log(f"  {target}:")
                    for metric in METRICS_TO_LOG:
                        if metric in m:
                            logger.log(f"    {metric}: {m[metric]:.3f}")

        plot_training_curves(history,
            os.path.join(output_dir, 'training_curve.png'),
            f"{model_type} - {label}")
        plot_prediction_scatter(y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
            output_dir, f"{model_type}_{label}")
        plot_prediction_scatter_linear(y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
            output_dir, f"{model_type}_{label}")
        plot_residuals(y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
            output_dir, f"{model_type}_{label}")
        save_test_predictions(y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
            output_dir, f"{model_type}_{label}")

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

        results = {
            'window_minutes':       window_minutes,
            'is_root_only':         is_root_only,
            'model_type':           model_type,
            'use_training_seed':    cfg.USE_TRAINING_SEED,
            'training_seed':        cfg.TRAINING_SEED if cfg.USE_TRAINING_SEED else None,
            'hyperparameters':      canonical,
            'test_metrics':         test_results,
            'per_horizon_metrics':  convert_numpy_types(horizon_metrics),
            'training_epochs':      len(history['train_loss']),
            'final_train_loss':     history['train_loss'][-1],
            'final_val_loss':       history['val_loss'][-1],
            'scheduler':            scheduler_config,
            'model_parameters':     model.get_num_parameters()
        }
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)

        return results

    return None


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------

def run_all_experiments(rank=0, world_size=1):
    from datetime import datetime
    import config.dataset_config as ds_cfg

    datasets_to_process = cfg.DATASETS_TO_RUN
    timestamp           = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id              = f"run_{timestamp}"
    run_root            = os.path.join(cfg.OUTPUT_DIR, run_id)
    training_seeds      = resolve_seed_list(
        getattr(cfg, "TRAINING_SEEDS", None),
        cfg.TRAINING_SEED,
        use_seed=cfg.USE_TRAINING_SEED
    )
    metrics_summary     = None
    metrics_summary_path = None
    if rank == 0:
        metrics_summary = create_summary(
            run_type="main",
            run_id=run_id,
            runner="main.py",
            seed=training_seeds[0] if len(training_seeds) == 1 else None,
            settings={
                "use_training_seed": cfg.USE_TRAINING_SEED,
                "seed_list": training_seeds,
                "models_to_run": cfg.MODELS_TO_RUN,
                "datasets_to_run": datasets_to_process,
                "use_images": cfg.USE_IMAGE_FEATURES,
                "future_horizons": cfg.USE_FUTURE_HORIZONS,
                "world_size": world_size,
            },
        )
        metrics_summary_path = write_summary(metrics_summary)

    for dataset_name in datasets_to_process:
        if rank == 0:
            print(f"\nSTARTING: {dataset_name.upper()}")

        cfg.CURRENT_DATASET_NAME = dataset_name
        cfg.DATASET_CONF         = ds_cfg.DATASETS[dataset_name]
        cfg.EARLY_WINDOWS        = cfg.DATASET_CONF['windows']
        paths                    = cfg.DATASET_CONF['paths']
        cfg.THREAD_METADATA      = paths['thread_metadata']
        cfg.THREAD_POSTS         = paths['thread_posts']
        cfg.USER_DEGREES         = paths['user_degrees']
        cfg.EARLY_WINDOW_DIR     = paths['early_window_dir']
        cfg.EMBEDDING_DIR        = paths['embedding_dir']
        cfg.OUTPUT_DIR           = paths['output_dir']

        # Determine reference window for root-only data loading.
        # Priority 1: first non-zero window in windows[]
        # Priority 2: root_only_reference_window from dataset_config
        nonzero_windows  = [w for w in cfg.EARLY_WINDOWS if w != 0]
        reference_window = (
            nonzero_windows[0]
            if nonzero_windows
            else cfg.DATASET_CONF.get('root_only_reference_window')
        )

        # Validate early — fail clearly if root-only has no data source
        if 0 in cfg.EARLY_WINDOWS and reference_window is None:
            raise ValueError(
                f"Dataset '{dataset_name}' has window=0 (root-only) but no "
                f"cascade windows and no 'root_only_reference_window' set in "
                f"dataset_config.py. Add "
                f"'root_only_reference_window': <window_minutes> to fix this."
            )

        output_base = os.path.join(run_root, dataset_name)
        if rank == 0:
            os.makedirs(output_base, exist_ok=True)
        _barrier()

        log_file = os.path.join(output_base, 'experiment.log')
        logger = Logger(log_file) \
                 if rank == 0 else None

        if rank == 0:
            add_source(metrics_summary, log_file=log_file, output_dir=output_base)
            metrics_summary_path = write_summary(metrics_summary)
            logger.section(f"{dataset_name.upper()} THREAD PREDICTION")
            has_root_only = 0 in cfg.EARLY_WINDOWS
            logger.log(f"Windows          : {cfg.EARLY_WINDOWS}")
            logger.log(f"Root-only        : {has_root_only}")
            if has_root_only:
                logger.log(f"Reference window : {reference_window}min "
                           f"(posts/meta/splits for root-only)")

        embedding_loader = None
        if any(m in cfg.MODELS_TO_RUN for m in ['mlp', 'graphsage', 'gat']):
            embedding_loader = EmbeddingLoader(cfg.EMBEDDING_DIR)

        # Load future horizon ground truth once per dataset (shared across all windows)
        horizon_dfs = valid_mask_df = None
        if cfg.USE_FUTURE_HORIZONS:
            if rank == 0:
                logger.log(f"\nLoading future horizon data from {cfg.EARLY_WINDOW_DIR} ...")
            try:
                horizon_dfs, valid_mask_df = load_future_horizon_data(cfg.EARLY_WINDOW_DIR)
                if rank == 0:
                    logger.log("  Future horizon data loaded successfully.")
            except FileNotFoundError as e:
                if rank == 0:
                    logger.log(f"  ERROR: {e}")
                raise

        image_loader = None
        if cfg.USE_IMAGE_FEATURES and cfg.DATASET_CONF.get('has_images', False):
            image_dir = (cfg.RAW_IMAGE_DIRS.get(dataset_name)
                         if not cfg.FREEZE_IMAGE_ENCODER else cfg.IMAGE_DIR)
            if image_dir is not None:
                try:
                    image_loader = (ImageEmbeddingLoader(image_dir)
                                    if cfg.FREEZE_IMAGE_ENCODER
                                    else ImagePixelLoader(image_dir))
                    if rank == 0:
                        logger.log(f"\nImage loader ready: {image_dir}")
                except Exception as e:
                    if rank == 0:
                        logger.log(f"Warning: could not load images: {e}")

        all_results = {}

        # Cache reference window data_dict to avoid loading it twice when
        # both root-only and that cascade window appear in windows[]
        reference_data_dict = None

        for window_minutes in cfg.EARLY_WINDOWS:
            is_root_only = (window_minutes == 0)

            if is_root_only:
                # Root-only: load reference window data if not already loaded
                if reference_data_dict is None:
                    if rank == 0:
                        logger.section(
                            f"LOADING REFERENCE DATA ({reference_window}min) "
                            f"FOR ROOT-ONLY")
                    reference_data_dict = load_all_data(reference_window)
                data_dict = reference_data_dict
                if rank == 0:
                    logger.section("PROCESSING ROOT-ONLY CONDITION")
            else:
                if rank == 0:
                    logger.section(f"PROCESSING {window_minutes}MIN WINDOW")
                data_dict = load_all_data(window_minutes)

                # Cache if this is the reference window so root-only can reuse it
                if window_minutes == reference_window:
                    reference_data_dict = data_dict

            # Fit preprocessor on this window's training data.
            # For root-only: fitted on reference window data (same trees → correct).
            train_meta  = data_dict['meta'][
                data_dict['meta']['tree_id'].isin(data_dict['train_ids'])]
            train_posts = data_dict['posts'][
                data_dict['posts']['tree_id'].isin(data_dict['train_ids'])]
            preprocessor = Preprocessor()
            preprocessor.fit(train_meta, train_posts)

            for model_type in cfg.MODELS_TO_RUN:
                # Separate cache dirs for root-only vs cascade
                if is_root_only:
                    cache_subdir = ("feature_cache_rootonly_tokenized"
                                    if model_type == 'text_graphsage'
                                    else "feature_cache_rootonly_frozen")
                else:
                    cache_subdir = ("feature_cache_tokenized"
                                    if model_type == 'text_graphsage'
                                    else "feature_cache_frozen")

                slot_label = (
                    "rootonly" if is_root_only else f"{window_minutes}min"
                )
                cache_kind = cache_subdir.replace("feature_cache_", "")
                cache_dir = os.path.join(
                    cfg.MAIN_CACHE_BASE,
                    f"{dataset_name}_{slot_label}_{cache_kind}"
                )

                for seed in training_seeds:
                    if cfg.USE_TRAINING_SEED:
                        cfg.TRAINING_SEED = seed
                        seed_everything(seed)
                        if rank == 0:
                            logger.log(f"\n[seed] main training seed = {seed}")

                    seed_label = "noseed" if seed is None else f"seed_{seed}"
                    exp_dir = os.path.join(
                        output_base, f"{window_minutes}min", model_type,
                        seed_label
                    )

                    try:
                        results = run_experiment(
                            window_minutes, model_type, data_dict,
                            embedding_loader, preprocessor,
                            exp_dir, cache_dir, logger,
                            image_loader=image_loader,
                            rank=rank, world_size=world_size,
                            is_root_only=is_root_only,
                            horizon_dfs=horizon_dfs,
                            valid_mask_df=valid_mask_df
                        )
                        if rank == 0 and results is not None:
                            key = ("root_only" if is_root_only
                                   else f"{window_minutes}min")
                            result_key = f"{key}_{model_type}_{seed_label}"
                            all_results[result_key] = results
                            add_metric_records(
                                metrics_summary,
                                results["test_metrics"],
                                dataset=dataset_name,
                                model_type=model_type,
                                window_minutes=window_minutes,
                                window_label=key,
                                variant=None,
                                slot_idx=None,
                                seed=seed,
                            )
                            metrics_summary_path = write_summary(metrics_summary)
                    except Exception as e:
                        if rank == 0:
                            label = "root_only" if is_root_only \
                                    else f"{window_minutes}min"
                            logger.log(
                                f"ERROR: {model_type} {label} seed={seed}: {e}"
                            )
                            import traceback
                            logger.log(traceback.format_exc())

        if rank == 0:
            with open(os.path.join(output_base, 'summary_results.json'), 'w') as f:
                json.dump(convert_numpy_types(all_results), f, indent=2)

        if embedding_loader:
            embedding_loader.clear_cache()

        _barrier()

    if rank == 0 and metrics_summary is not None:
        summary_json_path = write_summary(metrics_summary)
        if summary_json_path:
            print(f"\nNormalized metrics summary: {summary_json_path}")


if __name__ == '__main__':
    run_all_experiments()
