"""
Foundational model runner — ALL datasets × ALL window slots combined.

Location: foundational_runner.py  (root level, same directory as main.py)

Single GPU:  python main_foundational.py
4-GPU DDP:   torchrun --nproc_per_node=4 main_foundational.py

Trains ONE model on the combined train splits of all datasets × all slots.
Evaluates on test sets broken down by:
  - Per dataset (bluesky, gaming, futurology, ama)
  - Per slot (slot 0, slot 1, slot 2)
  - Per dataset × slot (most granular)

Conditioning features added to global features:
  - dataset_idx : len(DATASET_INDEX)-class one-hot — which platform/community
  - slot_idx    : n_slots-class one-hot — which maturity window (0=root/earliest)
  - log_follower_count : real for Bluesky, 0.0 for Reddit
  - has_follower_data  : 1.0 for Bluesky, 0.0 for Reddit

Root-only slot (window=0):
  Add 0 as the first entry of a dataset's windows list to enable root-only as
  slot 0 (e.g., 'windows': [0, 20, 50, 90]). The reference window (first non-zero
  window, or root_only_reference_window) provides posts/meta/splits.
  RootOnlyFeatureBuilder / RootOnlyTokenizedFeatureBuilder are used for that slot.
  All datasets in DATASETS_TO_RUN must have the same total number of windows.

Design decisions:
  - Same tree appears N× in training (once per slot) with identical targets
    but different graph structures — this is valid, not leakage.
    Train/test splits are consistent across slots.
  - One preprocessor fitted on ALL training data (all datasets × all slots,
    deduplicated so root-only and its reference cascade aren't counted twice).
  - Rank 0 builds caches; other ranks barrier-wait then load.
  - Only rank 0 logs, saves, plots, writes JSONs.
"""

import os
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader

import config.config as cfg
import config.feature_config as feat_cfg
import config.dataset_config as ds_cfg

from data.data_loader import load_all_data, load_future_horizon_data
from data.embedding_loader import EmbeddingLoader
from data.image_loader import ImageEmbeddingLoader, ImagePixelLoader
from data.preprocessing import Preprocessor
from data.feature_builder import (FeatureBuilder, TokenizedFeatureBuilder,
                                   RootOnlyFeatureBuilder,
                                   RootOnlyTokenizedFeatureBuilder)
from data.graph_builder import GraphDataset
from data.feature_cache import precompute_features
from models.model_factory import create_model
from training.trainer import train_model
from training.evaluator import (get_predictions_graphsage,
                                get_predictions_graphsage_with_horizon,
                                evaluate_predictions, evaluate_predictions_by_horizon,
                                save_horizon_predictions)
from utils.visualization import (plot_training_curves, plot_prediction_scatter,
                                  plot_prediction_scatter_linear, plot_residuals)
from utils.logger import Logger
from utils.metrics_summary import (active_training_seed, add_metric_records,
                                   add_source, create_summary, write_summary)

from training.experiment_runner import (save_test_predictions,
                                        convert_numpy_types,
                                        CANONICAL_PARAMS,
                                        _build_text_scheduler)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FOUNDATIONAL_OUTPUT_BASE = os.path.join(cfg.OUTPUT_DIR, "foundational")
FOUNDATIONAL_CACHE_BASE = os.path.join(cfg.OUTPUT_DIR, "foundational_cache")

# Per-GPU batch size for text_graphsage.
TEXT_GRAPHSAGE_BATCH_SIZE      = 8
TEXT_GRAPHSAGE_EFFECTIVE_BATCH = 256

# Dataset identity → integer index for one-hot encoding
DATASET_INDEX = {
    'bluesky':    0,
    'gaming':     1,
    'futurology': 2,
    'ama':        3
}

# N_SLOTS is computed dynamically from dataset configs at runtime.
# All datasets used together must have the same number of windows (including window=0).
# Do NOT hardcode this — it is set in run_foundational_experiments().


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


def _build_loader(dataset, batch_size, shuffle, rank, world_size, seed=None):
    """PyG DataLoader with DistributedSampler when world_size > 1."""
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
    if model_type != 'text_graphsage':
        return 1
    return max(1, TEXT_GRAPHSAGE_EFFECTIVE_BATCH //
                  (TEXT_GRAPHSAGE_BATCH_SIZE * world_size))


# ---------------------------------------------------------------------------
# Feature setup helpers
# ---------------------------------------------------------------------------

def _add_foundational_features(n_slots):
    """
    Register conditioning features in GLOBAL_FEATURES.

    dataset_idx : len(DATASET_INDEX)-class one-hot — platform identity
    slot_idx    : n_slots-class one-hot — maturity window (0=earliest/root)
    log_follower_count : real for Bluesky, zero-filled for Reddit
    has_follower_data  : missingness flag (1=Bluesky, 0=Reddit)

    preprocessing.py handles categorical one-hot and scalar standardization
    automatically — no changes needed there.
    """
    feat_cfg.GLOBAL_FEATURES['root_metadata']['dataset_idx'] = {
        'type': 'categorical',
        'num_classes': len(DATASET_INDEX)
    }
    feat_cfg.GLOBAL_FEATURES['root_metadata']['slot_idx'] = {
        'type': 'categorical',
        'num_classes': n_slots
    }
    # Keep log_follower_count — zero-filled for Reddit, real for Bluesky
    # Scaler will be fit only on non-zero (Bluesky) values in preprocessing.py
    feat_cfg.GLOBAL_FEATURES['root_metadata']['log_follower_count'] = {
        'standardize': True
    }
    feat_cfg.GLOBAL_FEATURES['root_metadata']['has_follower_data'] = {
        'standardize': False
    }
    print("[Foundational] Global features registered:")
    print(f"  dataset_idx  : {len(DATASET_INDEX)}-class one-hot")
    print(f"  slot_idx     : {n_slots}-class one-hot")
    print(f"  log_follower_count : real (Bluesky) / 0.0 (Reddit)")
    print(f"  has_follower_data  : 1.0 (Bluesky) / 0.0 (Reddit)")


def _stamp_context(data_dict, dataset_name, slot_idx):
    """
    Add conditioning columns to meta DataFrame in-place.
    Called after load_all_data() for each (dataset, slot) combination.
    """
    meta = data_dict['meta']
    meta['dataset_idx'] = DATASET_INDEX[dataset_name]
    meta['slot_idx']    = slot_idx

    if dataset_name == 'bluesky':
        # log_follower_count already present from data_loader.py
        meta['has_follower_data'] = 1.0
    else:
        meta['log_follower_count'] = 0.0
        meta['has_follower_data']  = 0.0


# ---------------------------------------------------------------------------
# Dataset / config helpers
# ---------------------------------------------------------------------------

def _setup_dataset_config(dataset_name):
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


def _get_image_loader(dataset_name, logger, rank):
    conf = ds_cfg.DATASETS[dataset_name]
    if not cfg.USE_IMAGE_FEATURES or not conf.get('has_images', False):
        return None
    loader_cls = ImageEmbeddingLoader if cfg.FREEZE_IMAGE_ENCODER else ImagePixelLoader
    image_dir  = (conf['paths'].get('image_dir') if cfg.FREEZE_IMAGE_ENCODER
                  else cfg.RAW_IMAGE_DIRS.get(dataset_name))
    if image_dir is None:
        if rank == 0 and logger:
            logger.log(f"  [{dataset_name}] No image dir — skipping")
        return None
    try:
        loader = loader_cls(image_dir)
        if rank == 0 and logger:
            logger.log(f"  [{dataset_name}] Image loader ready")
        return loader
    except Exception as e:
        if rank == 0 and logger:
            logger.log(f"  [{dataset_name}] Image load failed: {e}")
        return None


def _get_embedding_loader(dataset_name, model_type, logger, rank):
    if model_type == 'text_graphsage':
        return None
    _setup_dataset_config(dataset_name)
    try:
        loader = EmbeddingLoader(cfg.EMBEDDING_DIR)
        if rank == 0 and logger:
            logger.log(f"  [{dataset_name}] Embedding loader ready")
        return loader
    except Exception as e:
        if rank == 0 and logger:
            logger.log(f"  [{dataset_name}] Embedding load failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _log_and_save_results(model, test_loaders_by_key, datasets_to_process,
                           model_dir, model_type, logger, rank, n_slots):
    """
    Evaluate on all (dataset, slot) test sets.
    Logs and saves:
      - Per (dataset, slot) metrics  [and per-horizon when USE_FUTURE_HORIZONS]
      - Per dataset summary (mean across slots)
      - Per slot summary (mean across datasets)
      - Overall summary
      - horizon_predictions_{dataset}_slot{n}.csv when USE_FUTURE_HORIZONS

    test_loaders_by_key: dict keyed by (dataset_name, slot_idx)
    """
    METRICS = ('MSE', 'R2', 'Spearman')
    all_results         = {}   # (dataset, slot) → test_results dict
    all_horizon_results = {}   # (dataset, slot) → per-horizon results dict

    # -------------------------------------------------------------------
    # Per (dataset, slot) evaluation
    # -------------------------------------------------------------------
    for (dataset_name, slot_idx), test_loader in sorted(test_loaders_by_key.items()):
        logger.log(f"\n{'─'*60}")
        logger.log(f"Test: {dataset_name.upper()} | Slot {slot_idx}")
        logger.log(f"{'─'*60}")

        result_dir = os.path.join(model_dir, dataset_name, f"slot{slot_idx}")
        os.makedirs(result_dir, exist_ok=True)
        run_name = f"{model_type}_{dataset_name}_slot{slot_idx}"

        if cfg.USE_FUTURE_HORIZONS:
            y_pred, y_true, tree_ids, horizon_idxs = \
                get_predictions_graphsage_with_horizon(model, test_loader, cfg.DEVICE)
            test_results    = evaluate_predictions(y_true, y_pred, feat_cfg.OUTPUT_TARGETS)
            horizon_results = evaluate_predictions_by_horizon(y_true, y_pred, horizon_idxs)
            all_horizon_results[(dataset_name, slot_idx)] = horizon_results

            # Save per-tree per-horizon CSV
            hz_csv = os.path.join(result_dir,
                                  f'horizon_predictions_{run_name}.csv')
            save_horizon_predictions(y_true, y_pred, tree_ids, horizon_idxs, hz_csv)

            # Log per-horizon summary
            for h_name, h_res in horizon_results.items():
                logger.log(f"\n  [HORIZON: {h_name}]")
                for target, m in h_res.items():
                    if target == 'group_summaries':
                        continue
                    logger.log(f"    {target}: " +
                               "  ".join(f"{k}={m[k]:.3f}"
                                         for k in METRICS if k in m))
        else:
            y_pred, y_true = get_predictions_graphsage(model, test_loader, cfg.DEVICE)
            test_results   = evaluate_predictions(y_true, y_pred, feat_cfg.OUTPUT_TARGETS)

            for group_name, group_targets in feat_cfg.TARGET_GROUPS.items():
                logger.log(f"\n  [{group_name.upper()}]")
                for target in group_targets:
                    if target not in test_results:
                        continue
                    m = test_results[target]
                    logger.log(f"  {target}:")
                    for metric in METRICS:
                        if metric in m:
                            logger.log(f"    {metric}: {m[metric]:.3f}")

        save_test_predictions(y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
                              result_dir, run_name)
        plot_prediction_scatter(y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
                                result_dir, run_name)
        plot_prediction_scatter_linear(y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
                                       result_dir, run_name)
        plot_residuals(y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
                       result_dir, run_name)

        all_results[(dataset_name, slot_idx)] = test_results

    # -------------------------------------------------------------------
    # Per-dataset summary (mean across slots)
    # -------------------------------------------------------------------
    logger.log(f"\n{'='*60}")
    logger.log("SUMMARY: Per Dataset (mean across slots)")
    logger.log(f"{'='*60}")

    dataset_summary = {}
    for dataset_name in datasets_to_process:
        slot_results = [all_results[(dataset_name, s)]
                        for s in range(n_slots)
                        if (dataset_name, s) in all_results]
        if not slot_results:
            continue
        ds_mean = {}
        for target in feat_cfg.OUTPUT_TARGETS:
            ds_mean[target] = {
                metric: np.mean([r[target][metric]
                                 for r in slot_results if target in r])
                for metric in METRICS
                if metric in slot_results[0].get(target, {})
            }
        dataset_summary[dataset_name] = ds_mean
        logger.log(f"\n  {dataset_name.upper()}:")
        for target, metrics in ds_mean.items():
            logger.log(f"    {target}: " +
                       "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    # -------------------------------------------------------------------
    # Per-slot summary (mean across datasets)
    # -------------------------------------------------------------------
    logger.log(f"\n{'='*60}")
    logger.log("SUMMARY: Per Slot (mean across datasets)")
    logger.log(f"{'='*60}")

    slot_summary = {}
    for slot_idx in range(n_slots):
        slot_results = [all_results[(d, slot_idx)]
                        for d in datasets_to_process
                        if (d, slot_idx) in all_results]
        if not slot_results:
            continue
        sl_mean = {}
        for target in feat_cfg.OUTPUT_TARGETS:
            sl_mean[target] = {
                metric: np.mean([r[target][metric]
                                 for r in slot_results if target in r])
                for metric in METRICS
                if metric in slot_results[0].get(target, {})
            }
        slot_summary[f"slot{slot_idx}"] = sl_mean
        logger.log(f"\n  SLOT {slot_idx} (0=root/earliest, {n_slots-1}=latest):")
        for target, metrics in sl_mean.items():
            logger.log(f"    {target}: " +
                       "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    # -------------------------------------------------------------------
    # Overall summary
    # -------------------------------------------------------------------
    all_r2 = np.mean([
        all_results[(d, s)][t]['R2']
        for (d, s) in all_results
        for t in feat_cfg.OUTPUT_TARGETS
        if t in all_results[(d, s)]
    ])
    all_pearson = np.mean([
        all_results[(d, s)][t]['Pearson']
        for (d, s) in all_results
        for t in feat_cfg.OUTPUT_TARGETS
        if t in all_results[(d, s)]
    ])
    logger.log(f"\n{'='*60}")
    logger.log(f"OVERALL: mean R²={all_r2:.4f}  mean Pearson={all_pearson:.4f}")
    logger.log(f"{'='*60}")

    # Serialise for JSON (convert tuple keys to strings)
    json_results = {
        'per_dataset_slot': {
            f"{d}_slot{s}": convert_numpy_types(r)
            for (d, s), r in all_results.items()
        },
        'per_dataset_summary': convert_numpy_types(dataset_summary),
        'per_slot_summary':    convert_numpy_types(slot_summary),
        'overall': {'mean_R2': float(all_r2), 'mean_Pearson': float(all_pearson)},
    }
    if all_horizon_results:
        json_results['per_horizon_metrics'] = {
            f"{d}_slot{s}": convert_numpy_types(r)
            for (d, s), r in all_horizon_results.items()
        }
    return json_results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_foundational_experiments(rank=0, world_size=1):

    datasets_to_process = cfg.DATASETS_TO_RUN
    timestamp           = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id              = f"run_{timestamp}"
    output_base         = os.path.join(FOUNDATIONAL_OUTPUT_BASE, run_id)

    if rank == 0:
        os.makedirs(output_base, exist_ok=True)
    _barrier()

    log_file = os.path.join(output_base, 'foundational_experiment.log')
    logger = Logger(log_file) \
             if rank == 0 else None
    metrics_summary = None
    metrics_summary_path = None
    if rank == 0:
        metrics_summary = create_summary(
            run_type="foundational",
            run_id=run_id,
            runner="main_foundational.py",
            seed=active_training_seed("foundational"),
            settings={
                "use_training_seed": cfg.USE_TRAINING_SEED,
                "models_to_run": cfg.MODELS_TO_RUN,
                "datasets_to_run": datasets_to_process,
                "use_images": cfg.USE_IMAGE_FEATURES,
                "future_horizons": cfg.USE_FUTURE_HORIZONS,
                "world_size": world_size,
            },
        )
        add_source(metrics_summary, log_file=log_file, output_dir=output_base)
        metrics_summary_path = write_summary(metrics_summary)

    # Compute n_slots dynamically — all datasets must have the same number of windows
    slot_counts = [len(ds_cfg.DATASETS[d]['windows']) for d in datasets_to_process]
    n_slots = max(slot_counts)

    # Must happen before any data loading or preprocessor fitting
    _add_foundational_features(n_slots)

    if rank == 0:
        logger.section("FOUNDATIONAL MODEL — ALL DATASETS × ALL SLOTS")
        logger.log(f"Datasets    : {datasets_to_process}")
        logger.log(f"Models      : {cfg.MODELS_TO_RUN}")
        logger.log(f"Slots       : {n_slots} max")
        logger.log(f"Targets     : {feat_cfg.OUTPUT_TARGETS}")
        logger.log(f"DDP         : world_size={world_size}")
        logger.log(f"\nWindow map:")
        for d in datasets_to_process:
            windows = ds_cfg.DATASETS[d]['windows']
            slot_labels = []
            for w in windows:
                slot_labels.append("root" if w == 0 else f"{w}min")
            logger.log(f"  {d:12s}: {windows}  ({', '.join(slot_labels)})")

    any_has_images = any(
        ds_cfg.DATASETS[d].get('has_images', False) for d in datasets_to_process
    )

    # -----------------------------------------------------------------------
    # 1. Load ALL data upfront — all datasets × all slots
    #    Keyed by (dataset_name, slot_idx)
    # -----------------------------------------------------------------------
    if rank == 0:
        logger.section("LOADING ALL DATA")

    # per_data[(dataset, slot)] = {data_dict, window_minutes, is_root_only}
    per_data          = {}
    # image loaders — one per dataset (reused across slots)
    image_loaders     = {}
    # future horizon data — one per dataset (same across all slots)
    horizon_data      = {}   # dataset_name → (horizon_dfs, valid_mask_df) or (None, None)
    # reference data per dataset — for root-only slots (window=0)
    reference_data_by_dataset = {}  # dataset_name → data_dict of first non-zero window

    for dataset_name in datasets_to_process:
        _setup_dataset_config(dataset_name)
        image_loaders[dataset_name] = _get_image_loader(dataset_name, logger, rank)

        # Load future horizon ground truth (dataset-level, not slot-level)
        if cfg.USE_FUTURE_HORIZONS:
            if rank == 0:
                logger.log(f"\n  [{dataset_name}] Loading future horizon data ...")
            try:
                h_dfs, v_mask = load_future_horizon_data(cfg.EARLY_WINDOW_DIR)
                horizon_data[dataset_name] = (h_dfs, v_mask)
                if rank == 0:
                    logger.log(f"  [{dataset_name}] Future horizon data loaded.")
            except FileNotFoundError as e:
                if rank == 0:
                    logger.log(f"  [{dataset_name}] WARNING: {e} — horizons disabled for this dataset.")
                horizon_data[dataset_name] = (None, None)
        else:
            horizon_data[dataset_name] = (None, None)

        dataset_windows = ds_cfg.DATASETS[dataset_name]['windows']
        nonzero_windows = [w for w in dataset_windows if w != 0]

        # Validate root-only config
        if 0 in dataset_windows:
            ref_win = (nonzero_windows[0] if nonzero_windows
                       else ds_cfg.DATASETS[dataset_name].get('root_only_reference_window'))
            if ref_win is None:
                raise ValueError(
                    f"Dataset '{dataset_name}' has window=0 but no non-zero cascade "
                    f"windows and no 'root_only_reference_window'. Set one of these."
                )

        for slot_idx, window_minutes in enumerate(dataset_windows):
            is_root_only   = (window_minutes == 0)
            _setup_dataset_config(dataset_name)   # reset cfg paths per dataset

            if is_root_only:
                # Determine reference window for loading posts/meta/splits
                ref_win = (nonzero_windows[0] if nonzero_windows
                           else ds_cfg.DATASETS[dataset_name].get('root_only_reference_window'))

                # Load reference data once and cache it
                if dataset_name not in reference_data_by_dataset:
                    if rank == 0:
                        logger.log(f"\nLoading {dataset_name} reference data "
                                   f"({ref_win}min) for root-only slot {slot_idx} ...")
                    _setup_dataset_config(dataset_name)
                    reference_data_by_dataset[dataset_name] = load_all_data(ref_win)

                ref = reference_data_by_dataset[dataset_name]
                # Make a shallow copy with an independent meta DataFrame so
                # slot_idx conditioning does not collide between slots sharing data.
                data_dict = {k: ref[k].copy() if k == 'meta' else ref[k] for k in ref}

                if rank == 0:
                    logger.log(f"\nLoading {dataset_name} slot {slot_idx} "
                               f"(root-only, reference={ref_win}min) ...")
            else:
                if rank == 0:
                    logger.log(f"\nLoading {dataset_name} slot {slot_idx} "
                               f"({window_minutes}min) ...")
                data_dict = load_all_data(window_minutes)
                # Cache as reference if this is the first non-zero window and not yet cached
                if (window_minutes == nonzero_windows[0]
                        and dataset_name not in reference_data_by_dataset):
                    reference_data_by_dataset[dataset_name] = data_dict

            # Stamp conditioning features (slot_idx, dataset_idx, follower info) into meta
            _stamp_context(data_dict, dataset_name, slot_idx)

            per_data[(dataset_name, slot_idx)] = {
                'data_dict':      data_dict,
                'window_minutes': window_minutes,
                'is_root_only':   is_root_only
            }

    # -----------------------------------------------------------------------
    # 2. Fit ONE preprocessor on ALL training data (all datasets × all slots)
    # -----------------------------------------------------------------------
    if rank == 0:
        logger.section("FITTING COMBINED PREPROCESSOR")

    all_train_meta, all_train_posts = [], []
    # Deduplicate: root-only slots share underlying data with their reference cascade
    # window. Track by (dataset_name, actual_data_window) to avoid fitting the
    # preprocessor twice on identical posts/meta.
    seen_data_keys = set()
    for (dataset_name, slot_idx), entry in sorted(per_data.items()):
        dd           = entry['data_dict']
        win          = entry['window_minutes']
        is_root_only = entry.get('is_root_only', False)

        if is_root_only:
            # Actual data comes from the reference window of this dataset
            ref_win = reference_data_by_dataset.get(dataset_name)
            # Use id of the reference data dict as dedup key
            data_key = (dataset_name, id(reference_data_by_dataset.get(dataset_name)))
        else:
            data_key = (dataset_name, win)

        if data_key in seen_data_keys:
            continue
        seen_data_keys.add(data_key)

        train_mask = dd['meta']['tree_id'].isin(dd['train_ids'])
        post_mask  = dd['posts']['tree_id'].isin(dd['train_ids'])
        all_train_meta.append(dd['meta'][train_mask])
        all_train_posts.append(dd['posts'][post_mask])

    combined_meta  = pd.concat(all_train_meta,  ignore_index=True)
    combined_posts = pd.concat(all_train_posts, ignore_index=True)

    preprocessor = Preprocessor()
    preprocessor.fit(combined_meta, combined_posts)

    if rank == 0:
        logger.log(f"Preprocessor fitted on {len(combined_meta):,} trees "
                   f"from {len(per_data)} (dataset, slot) combinations")
        logger.log(f"Total training posts: {len(combined_posts):,}")

    # -----------------------------------------------------------------------
    # 3. Per model type: build caches → datasets → loaders → train → evaluate
    # -----------------------------------------------------------------------
    all_final_results = {}

    for model_type in cfg.MODELS_TO_RUN:
        if rank == 0:
            logger.section(f"MODEL: {model_type.upper()}")

        model_dir = os.path.join(output_base, model_type)
        if rank == 0:
            os.makedirs(model_dir, exist_ok=True)
        _barrier()

        use_images  = (cfg.USE_IMAGE_FEATURES and any_has_images and
                       model_type in ['graphsage', 'gat', 'text_graphsage'])
        BATCH_SIZE  = (TEXT_GRAPHSAGE_BATCH_SIZE
                       if model_type == 'text_graphsage'
                       else cfg.BATCH_SIZE)
        accum_steps = _compute_accum_steps(model_type, world_size)

        if rank == 0:
            logger.log(f"use_images     = {use_images}")
            logger.log(f"per-GPU batch  = {BATCH_SIZE}")
            logger.log(f"accum_steps    = {accum_steps}")
            logger.log(f"effective batch= {BATCH_SIZE * world_size * accum_steps}")

        # Accumulate splits across ALL (dataset, slot) combinations
        train_graph_datasets = []
        val_graph_datasets   = []
        # test loaders keyed by (dataset_name, slot_idx) for granular evaluation
        test_loaders_by_key  = {}

        node_feat_dim       = None
        global_feat_dim     = None
        output_dim          = None
        structural_feat_dim = None

        for (dataset_name, slot_idx), entry in sorted(per_data.items()):
            _setup_dataset_config(dataset_name)
            dd             = entry['data_dict']
            window_minutes = entry['window_minutes']
            is_root_only   = entry.get('is_root_only', False)
            image_loader   = image_loaders[dataset_name]

            # Separate cache dir per (dataset, slot, model_type)
            slot_tag = f"rootonly" if is_root_only else f"{window_minutes}min"
            cache_tag = (f"foundational_allslots_{dataset_name}"
                         f"_slot{slot_idx}_{slot_tag}")
            cache_dir = os.path.join(
                FOUNDATIONAL_CACHE_BASE,
                f"cache_tokenized_{cache_tag}"
                if model_type == 'text_graphsage'
                else f"cache_frozen_{cache_tag}"
            )
            if rank == 0:
                os.makedirs(cache_dir, exist_ok=True)
            _barrier()

            emb_loader = _get_embedding_loader(
                dataset_name, model_type, logger, rank)

            if is_root_only:
                if rank == 0 and logger:
                    logger.log(f"  [{dataset_name}] slot{slot_idx}: RootOnly builder")
                fb = (RootOnlyTokenizedFeatureBuilder(
                          preprocessor, dd['posts'],
                          dd['user_degrees'], image_loader=image_loader)
                      if model_type == 'text_graphsage'
                      else RootOnlyFeatureBuilder(
                          emb_loader, preprocessor, dd['posts'],
                          dd['user_degrees'], image_loader=image_loader))
            else:
                fb = (TokenizedFeatureBuilder(
                          preprocessor, dd['posts'],
                          dd['user_degrees'], image_loader=image_loader)
                      if model_type == 'text_graphsage'
                      else FeatureBuilder(
                          emb_loader, preprocessor, dd['posts'],
                          dd['user_degrees'], image_loader=image_loader))

            mi = dd['meta'].set_index('tree_id')

            desc_tag = (f"{dataset_name} slot{slot_idx} (root-only)"
                        if is_root_only else f"{dataset_name} slot{slot_idx}")

            tr_cache = _build_cache(
                dd['train_ids'], dd['edges'], mi, dd['posts'], fb,
                os.path.join(cache_dir, 'train_features.pkl'),
                f"Train [{desc_tag}]", rank, cfg.FORCE_RECOMPUTE)
            va_cache = _build_cache(
                dd['val_ids'], dd['edges'], mi, dd['posts'], fb,
                os.path.join(cache_dir, 'val_features.pkl'),
                f"Val   [{desc_tag}]", rank, cfg.FORCE_RECOMPUTE)
            te_cache = _build_cache(
                dd['test_ids'], dd['edges'], mi, dd['posts'], fb,
                os.path.join(cache_dir, 'test_features.pkl'),
                f"Test  [{desc_tag}]", rank, cfg.FORCE_RECOMPUTE)

            h_dfs, v_mask = horizon_data.get(dataset_name, (None, None))
            _h  = h_dfs  if cfg.USE_FUTURE_HORIZONS else None
            _vm = v_mask if cfg.USE_FUTURE_HORIZONS else None

            train_ds = GraphDataset(dd['train_ids'], tr_cache,
                                    horizon_dfs=_h, valid_mask_df=_vm)
            val_ds   = GraphDataset(dd['val_ids'],   va_cache,
                                    horizon_dfs=_h, valid_mask_df=_vm)
            test_ds  = GraphDataset(dd['test_ids'],  te_cache,
                                    horizon_dfs=_h, valid_mask_df=_vm)

            train_graph_datasets.append(train_ds)
            val_graph_datasets.append(val_ds)

            # Test loader — full dataset, no DDP sampler (rank 0 evaluates)
            test_loaders_by_key[(dataset_name, slot_idx)] = PyGDataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
            )

            if rank == 0:
                win_label = "root-only" if is_root_only else f"{window_minutes}min"
                logger.log(f"  {dataset_name} slot{slot_idx} "
                           f"({win_label}): "
                           f"{len(train_ds)} train | "
                           f"{len(val_ds)} val | "
                           f"{len(test_ds)} test")

            # Infer feature dims from first combination
            if node_feat_dim is None and structural_feat_dim is None:
                probe = PyGDataLoader(train_ds, batch_size=2,
                                      shuffle=False, num_workers=0)
                sb = next(iter(probe))
                if model_type == 'text_graphsage':
                    structural_feat_dim = sb.x.shape[1]
                else:
                    node_feat_dim = sb.x.shape[1]
                global_feat_dim = sb.global_features.shape[1]
                output_dim      = sb.y.shape[1]
                del probe, sb

            if emb_loader is not None:
                emb_loader.clear_cache()

        # -------------------------------------------------------------------
        # Combined loaders across ALL datasets × ALL slots
        # -------------------------------------------------------------------
        combined_train = ConcatDataset(train_graph_datasets)
        combined_val   = ConcatDataset(val_graph_datasets)

        train_loader = _build_loader(
            combined_train, BATCH_SIZE, shuffle=True,
            rank=rank, world_size=world_size,
            seed=cfg.TRAINING_SEED if cfg.USE_TRAINING_SEED else None)
        val_loader   = _build_loader(combined_val,   BATCH_SIZE,
                                     shuffle=False, rank=rank, world_size=world_size)

        if rank == 0:
            logger.log(f"\nTotal combined train : {len(combined_train):,} graphs")
            logger.log(f"Total combined val   : {len(combined_val):,} graphs")
            logger.log(f"Total test sets      : "
                       f"{sum(len(l.dataset) for l in test_loaders_by_key.values()):,} "
                       f"graphs across {len(test_loaders_by_key)} (dataset, slot) pairs")

        # -------------------------------------------------------------------
        # Build model
        # -------------------------------------------------------------------
        canonical = CANONICAL_PARAMS[model_type]

        if model_type == 'text_graphsage':
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
                'image_projection_dim':   cfg.IMAGE_PROJECTION_DIM,
                'freeze_image_encoder':   cfg.FREEZE_IMAGE_ENCODER,
                'unfreeze_last_n_layers': cfg.UNFREEZE_LAST_N_LAYERS
            }
        else:
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
                'freeze_image_encoder':   cfg.FREEZE_IMAGE_ENCODER,
                'unfreeze_last_n_layers': cfg.UNFREEZE_LAST_N_LAYERS
            }

        model = create_model(model_type, model_params, cfg.DEVICE)
        if rank == 0:
            horizon_note = (f" + {cfg.N_HORIZONS}-class horizon OH"
                            if cfg.USE_FUTURE_HORIZONS else "")
            logger.log(f"\nModel parameters: {model.get_num_parameters():,}")
            logger.log(f"Global feature dim: {global_feat_dim} "
                       f"(includes {len(DATASET_INDEX)}-class dataset OH "
                       f"+ {n_slots}-class slot OH{horizon_note})")

        # Optimizer
        if model_type == 'text_graphsage':
            has_trainable_clip = (
                use_images and not cfg.FREEZE_IMAGE_ENCODER and
                hasattr(model.gnn, 'clip_model') and model.gnn.clip_model is not None
            )
            if has_trainable_clip:
                clip_params = [p for p in model.gnn.clip_model.parameters()
                               if p.requires_grad]
                proj_params = list(model.gnn.image_projection.parameters())
                exclude_ids = (set(id(p) for p in clip_params) |
                               set(id(p) for p in proj_params))
                gnn_only = [p for p in model.gnn.parameters()
                            if id(p) not in exclude_ids]
                param_groups = [
                    {'params': model.transformer.parameters(),
                     'lr': canonical['transformer_lr']},
                    {'params': gnn_only, 'lr': canonical['gnn_lr']},
                    {'params': clip_params, 'lr': cfg.IMAGE_LR,
                     'weight_decay': cfg.IMAGE_WEIGHT_DECAY},
                    {'params': proj_params, 'lr': cfg.IMAGE_PROJECTION_LR,
                     'weight_decay': cfg.IMAGE_WEIGHT_DECAY},
                ]
            else:
                param_groups = [
                    {'params': model.transformer.parameters(),
                     'lr': canonical['transformer_lr']},
                    {'params': model.gnn.parameters(), 'lr': canonical['gnn_lr']},
                ]
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=canonical['lr'])

        # -------------------------------------------------------------------
        # Train
        # -------------------------------------------------------------------
        max_epochs = 50 if model_type == 'text_graphsage' else cfg.MAX_EPOCHS

        if model_type == 'text_graphsage':
            scheduler, scheduler_config = _build_text_scheduler(
                optimizer, train_loader, accum_steps, max_epochs
            )
        else:
            scheduler, scheduler_config = None, None

        if rank == 0:
            logger.log(f"\nTraining (max_epochs={max_epochs}, "
                       f"patience={cfg.PATIENCE}) ...")
            if scheduler_config:
                logger.log(f"Scheduler: {scheduler_config}")

        model, history = train_model(
            model, model_type, train_loader, val_loader,
            optimizer, cfg.DEVICE, max_epochs=max_epochs,
            rank=rank, world_size=world_size,
            accumulation_steps=accum_steps,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
        )

        _barrier()

        # -------------------------------------------------------------------
        # Evaluate and save — rank 0 only
        # -------------------------------------------------------------------
        if rank == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model.pt'))
            plot_training_curves(
                history,
                os.path.join(model_dir, 'training_curve.png'),
                f"Foundational {model_type} — All Datasets × All Slots"
            )

            json_results = _log_and_save_results(
                model, test_loaders_by_key, datasets_to_process,
                model_dir, model_type, logger, rank, n_slots
            )

            with open(os.path.join(model_dir, 'results.json'), 'w') as f:
                json.dump(json_results, f, indent=2)

            all_final_results[model_type] = json_results
            for key, metrics in json_results.get('per_dataset_slot', {}).items():
                dataset_name, slot_str = key.rsplit('_slot', 1)
                slot_idx = int(slot_str)
                window_minutes = ds_cfg.DATASETS[dataset_name]['windows'][slot_idx]
                window_label = (
                    "root" if window_minutes == 0 else f"{window_minutes}min"
                )
                add_metric_records(
                    metrics_summary,
                    metrics,
                    dataset=dataset_name,
                    model_type=model_type,
                    window_minutes=window_minutes,
                    window_label=window_label,
                    variant=None,
                    slot_idx=slot_idx,
                )
            metrics_summary_path = write_summary(metrics_summary)

        _barrier()

    # -----------------------------------------------------------------------
    # Final summary JSON
    # -----------------------------------------------------------------------
    if rank == 0:
        summary_path = os.path.join(output_base, 'foundational_allslots_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(convert_numpy_types(all_final_results), f, indent=2)
        summary_json_path = write_summary(metrics_summary)
        logger.log(f"\n{'='*80}")
        logger.log("FOUNDATIONAL (ALL SLOTS) EXPERIMENTS COMPLETE")
        logger.log(f"Output  : {output_base}")
        logger.log(f"Summary : {summary_path}")
        if summary_json_path:
            logger.log(f"Metrics : {summary_json_path}")


if __name__ == '__main__':
    run_foundational_experiments()
