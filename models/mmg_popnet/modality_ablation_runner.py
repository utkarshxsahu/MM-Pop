"""
Modality ablation runner for TextGraphSAGE / UniSD.

Single GPU:
    python modality_ablation_runner.py

DDP:
    torchrun --nproc_per_node=4 modality_ablation_runner.py

Optional overrides:
    python modality_ablation_runner.py --variants no_text,no_image
    python modality_ablation_runner.py --datasets gaming
    python modality_ablation_runner.py --windows gaming:20,50,90
"""

import argparse
import copy
import gc
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader as PyGDataLoader

import config.config as cfg
import config.dataset_config as ds_cfg
import config.feature_config as feat_cfg

from data.data_loader import load_all_data
from data.feature_builder import TokenizedFeatureBuilder
from data.graph_builder import GraphDataset
from data.image_loader import ImageEmbeddingLoader, ImagePixelLoader
from data.preprocessing import Preprocessor
from models.model_factory import create_model
from training.evaluator import evaluate_predictions
from training.experiment_runner import (
    CANONICAL_PARAMS,
    _barrier,
    _build_cache,
    _build_pyg_loader,
    _build_text_scheduler,
    TEXT_GRAPHSAGE_SCHEDULER,
    convert_numpy_types,
)
from training.trainer import train_model
from utils.logger import Logger
from utils.metrics_summary import (add_metric_records, add_source,
                                   create_summary, write_summary)
from utils.reproducibility import resolve_seed_list, seed_everything
from utils.visualization import (
    plot_prediction_scatter,
    plot_prediction_scatter_linear,
    plot_residuals,
    plot_training_curves,
)


# ---------------------------------------------------------------------------
# Editable experiment config
# ---------------------------------------------------------------------------

MODALITY_ABLATION_OUTPUT_BASE = "/scratch/dgl/Social_Network/ablation_modality"
MODALITY_ABLATION_CACHE_BASE = "/scratch/dgl/Social_Network/ablation_modality_cache"

ABLATION_VARIANTS = [
    #"no_text",
    "no_image",
    "no_topology",
    "no_temporal",
    #"full_unisd",
]

ABLATION_DATASETS = {
    "gaming": [90],
    #"futurology": [180],
}

ABLATION_MODEL = "text_graphsage"
TEXT_GRAPHSAGE_BATCH_SIZE = 4
TEXT_GRAPHSAGE_EFFECTIVE_BATCH = 256
TEXT_GRAPHSAGE_MAX_EPOCHS = 100
DDP_TIMEOUT_HOURS = 6

VALID_VARIANTS = {
    "no_text": {
        "ablate_text": True,
        "ablate_image": False,
        "ablate_topology": False,
        "ablate_temporal": False,
    },
    "no_image": {
        "ablate_text": False,
        "ablate_image": True,
        "ablate_topology": False,
        "ablate_temporal": False,
    },
    "no_topology": {
        "ablate_text": False,
        "ablate_image": False,
        "ablate_topology": True,
        "ablate_temporal": False,
    },
    "no_temporal": {
        "ablate_text": False,
        "ablate_image": False,
        "ablate_topology": False,
        "ablate_temporal": True,
    },
    "full_unisd": {
        "ablate_text": False,
        "ablate_image": False,
        "ablate_topology": False,
        "ablate_temporal": False,
    },
}


# ---------------------------------------------------------------------------
# DDP / config helpers
# ---------------------------------------------------------------------------

def _is_ddp():
    return dist.is_available() and dist.is_initialized()


def _setup_dataset_config(dataset_name):
    cfg.CURRENT_DATASET_NAME = dataset_name
    cfg.DATASET_CONF = ds_cfg.DATASETS[dataset_name]
    cfg.EARLY_WINDOWS = cfg.DATASET_CONF["windows"]
    paths = cfg.DATASET_CONF["paths"]
    cfg.THREAD_METADATA = paths["thread_metadata"]
    cfg.THREAD_POSTS = paths["thread_posts"]
    cfg.USER_DEGREES = paths["user_degrees"]
    cfg.EARLY_WINDOW_DIR = paths["early_window_dir"]
    cfg.EMBEDDING_DIR = paths["embedding_dir"]
    cfg.OUTPUT_DIR = paths["output_dir"]
    cfg.IMAGE_DIR = paths.get("image_dir", None)


def _validate_variants(variants):
    unknown = [v for v in variants if v not in VALID_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}")
    return variants


def _compute_text_accum_steps(world_size):
    return max(1, TEXT_GRAPHSAGE_EFFECTIVE_BATCH //
                  (TEXT_GRAPHSAGE_BATCH_SIZE * world_size))


def _parse_csv_list(value):
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_windows_override(value):
    """
    Parse strings like:
        gaming:20,50,90;futurology:30,90,180
    """
    if value is None:
        return None

    parsed = {}
    for spec in value.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(
                "--windows must use dataset:w1,w2 format, "
                "for example gaming:20,50,90"
            )
        dataset_name, windows_str = spec.split(":", 1)
        windows = [int(w.strip()) for w in windows_str.split(",") if w.strip()]
        parsed[dataset_name.strip()] = windows
    return parsed


def _resolve_run_config(args):
    variants = _parse_csv_list(args.variants) or copy.deepcopy(ABLATION_VARIANTS)
    variants = _validate_variants(variants)

    datasets = copy.deepcopy(ABLATION_DATASETS)
    selected_datasets = _parse_csv_list(args.datasets)
    windows_override = _parse_windows_override(args.windows)

    if selected_datasets is not None:
        datasets = {k: datasets.get(k, []) for k in selected_datasets}

    if windows_override is not None:
        for dataset_name, windows in windows_override.items():
            datasets[dataset_name] = windows

    missing = [name for name, windows in datasets.items()
               if name not in ds_cfg.DATASETS or not windows]
    if missing:
        raise ValueError(f"Missing dataset config or window list for: {missing}")

    for dataset_name, windows in datasets.items():
        if 0 in windows:
            raise ValueError(
                f"Root-only window 0 is not supported by this modality "
                f"ablation runner ({dataset_name}: {windows})"
            )

    return variants, datasets


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _collect_predictions_with_tree_ids(model, data_loader, device):
    model.eval()
    all_preds, all_targets, all_tree_ids = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            tids = batch.tree_id
            if isinstance(tids, list):
                all_tree_ids.extend(tids)
            else:
                all_tree_ids.append(tids)

            batch = batch.to(device)
            y_pred = model(batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    return np.vstack(all_preds), np.vstack(all_targets), all_tree_ids


def _save_predictions_with_tree_ids(y_true, y_pred, tree_ids, target_names,
                                    output_dir, model_name):
    rows = []
    for sample_idx, tree_id in enumerate(tree_ids):
        for target_idx, target in enumerate(target_names):
            yt = float(y_true[sample_idx, target_idx])
            yp = float(y_pred[sample_idx, target_idx])
            rows.append({
                "tree_id": tree_id,
                "target": target,
                "y_true_log": yt,
                "y_pred_log": yp,
                "y_true_actual": float(np.expm1(yt)),
                "y_pred_actual": float(np.expm1(yp)),
            })

    path = os.path.join(output_dir, f"predictions_{model_name}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved predictions: {path}")
    return path


def _log_target_metrics(logger, results):
    metrics = ("MSE", "R2", "Pearson", "Spearman")
    for group_name, group_targets in feat_cfg.TARGET_GROUPS.items():
        logger.log(f"\n  [{group_name.upper()}]")
        for target in group_targets:
            if target not in results:
                continue
            m = results[target]
            logger.log(
                f"  {target}: " +
                "  ".join(f"{metric}={m[metric]:.3f}"
                          for metric in metrics if metric in m)
            )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_image_loader(dataset_name, logger, rank, load_images=True):
    if not load_images:
        if rank == 0:
            logger.log("  Image loader skipped: all selected variants ablate images")
        return None

    if not (cfg.USE_IMAGE_FEATURES and cfg.DATASET_CONF.get("has_images", False)):
        return None

    image_dir = (
        cfg.RAW_IMAGE_DIRS.get(dataset_name)
        if not cfg.FREEZE_IMAGE_ENCODER
        else cfg.DATASET_CONF["paths"].get("image_dir")
    )
    if not image_dir:
        return None

    try:
        loader = (
            ImageEmbeddingLoader(image_dir)
            if cfg.FREEZE_IMAGE_ENCODER
            else ImagePixelLoader(image_dir)
        )
        if rank == 0:
            logger.log(f"  Image loader ready: {image_dir}")
        return loader
    except Exception as exc:
        if rank == 0:
            logger.log(f"  Image loader failed: {exc}")
        return None


def _prepare_dataset_window(dataset_name, window_minutes, output_base,
                            logger, rank, world_size, load_images=True):
    _setup_dataset_config(dataset_name)

    if rank == 0:
        logger.section(f"DATA: {dataset_name} @ {window_minutes}min")

    data_dict = load_all_data(window_minutes)

    train_meta = data_dict["meta"][
        data_dict["meta"]["tree_id"].isin(data_dict["train_ids"])
    ]
    train_posts = data_dict["posts"][
        data_dict["posts"]["tree_id"].isin(data_dict["train_ids"])
    ]
    preprocessor = Preprocessor()
    preprocessor.fit(train_meta, train_posts)

    image_loader = _build_image_loader(dataset_name, logger, rank, load_images)

    feature_builder = TokenizedFeatureBuilder(
        preprocessor,
        data_dict["posts"],
        data_dict["user_degrees"],
        image_loader=image_loader,
    )

    cache_dir = os.path.join(
        MODALITY_ABLATION_CACHE_BASE,
        f"{dataset_name}_{window_minutes}min_tokenized",
    )
    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
    _barrier()

    meta_indexed = data_dict["meta"].set_index("tree_id")

    train_cache = _build_cache(
        data_dict["train_ids"], data_dict["edges"], meta_indexed,
        data_dict["posts"], feature_builder,
        os.path.join(cache_dir, "train_features.pkl"),
        desc=f"Train [{dataset_name} {window_minutes}min]",
        rank=rank, force_recompute=cfg.FORCE_RECOMPUTE,
    )
    val_cache = _build_cache(
        data_dict["val_ids"], data_dict["edges"], meta_indexed,
        data_dict["posts"], feature_builder,
        os.path.join(cache_dir, "val_features.pkl"),
        desc=f"Val [{dataset_name} {window_minutes}min]",
        rank=rank, force_recompute=cfg.FORCE_RECOMPUTE,
    )
    test_cache = _build_cache(
        data_dict["test_ids"], data_dict["edges"], meta_indexed,
        data_dict["posts"], feature_builder,
        os.path.join(cache_dir, "test_features.pkl"),
        desc=f"Test [{dataset_name} {window_minutes}min]",
        rank=rank, force_recompute=cfg.FORCE_RECOMPUTE,
    )

    train_ds = GraphDataset(data_dict["train_ids"], train_cache)
    val_ds = GraphDataset(data_dict["val_ids"], val_cache)
    test_ds = GraphDataset(data_dict["test_ids"], test_cache)

    sample = train_ds[0]
    structural_feat_dim = sample.x.shape[1]
    global_feat_dim = sample.global_features.shape[1]
    output_dim = sample.y.shape[1]

    if rank == 0:
        logger.log(
            f"  Dataset sizes: train={len(train_ds)} "
            f"val={len(val_ds)} test={len(test_ds)}"
        )
        logger.log(
            f"  Dims: structural={structural_feat_dim} "
            f"global={global_feat_dim} output={output_dim}"
        )
        logger.log(f"  Cache: {cache_dir}")

    return {
        "data_dict": data_dict,
        "preprocessor": preprocessor,
        "image_loader": image_loader,
        "window_minutes": window_minutes,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "structural_feat_dim": structural_feat_dim,
        "global_feat_dim": global_feat_dim,
        "output_dim": output_dim,
        "cache_dir": cache_dir,
    }


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------

def _build_model_and_optimizer(entry, dataset_name, variant, rank):
    flags = VALID_VARIANTS[variant]
    canonical = CANONICAL_PARAMS[ABLATION_MODEL]

    use_images = (
        cfg.USE_IMAGE_FEATURES and
        cfg.DATASET_CONF.get("has_images", False) and
        ABLATION_MODEL == "text_graphsage" and
        (flags["ablate_image"] or entry["image_loader"] is not None)
    )

    model_params = {
        "structural_feat_dim": entry["structural_feat_dim"],
        "global_feat_dim": entry["global_feat_dim"],
        "output_dim": entry["output_dim"],
        "num_layers": canonical["num_layers"],
        "hidden_dim": canonical["hidden_dim"],
        "aggregator_type": canonical["aggregator_type"],
        "dropout": canonical["dropout"],
        "final_mlp_layers": canonical["final_mlp_layers"],
        "final_mlp_hidden": canonical["final_mlp_hidden"],
        "transformer_name": "sentence-transformers/all-MiniLM-L6-v2",
        "use_images": use_images,
        "text_projection_dim": cfg.TEXT_PROJECTION_DIM,
        "text_projection_type": "bottleneck_mlp",
        "text_projection_hidden": 64,
        "text_projection_dropout": 0.15,
        "image_projection_dim": cfg.IMAGE_PROJECTION_DIM,
        "freeze_image_encoder": cfg.FREEZE_IMAGE_ENCODER,
        "unfreeze_last_n_layers": cfg.UNFREEZE_LAST_N_LAYERS,
        **flags,
    }

    model = create_model(ABLATION_MODEL, model_params, cfg.DEVICE)

    has_trainable_clip = (
        use_images and not flags["ablate_image"] and
        not cfg.FREEZE_IMAGE_ENCODER and
        hasattr(model.gnn, "clip_model") and
        model.gnn.clip_model is not None
    )

    if flags["ablate_text"] and has_trainable_clip:
        clip_params = [
            p for p in model.gnn.clip_model.parameters() if p.requires_grad
        ]
        proj_params = list(model.gnn.image_projection.parameters())
        exclude_ids = set(id(p) for p in clip_params) | set(id(p) for p in proj_params)
        gnn_only_params = [
            p for p in model.gnn.parameters() if id(p) not in exclude_ids
        ]
        param_groups = [
            {"params": gnn_only_params, "lr": canonical["gnn_lr"]},
            {"params": clip_params, "lr": cfg.IMAGE_LR,
             "weight_decay": cfg.IMAGE_WEIGHT_DECAY},
            {"params": proj_params, "lr": cfg.IMAGE_PROJECTION_LR,
             "weight_decay": cfg.IMAGE_WEIGHT_DECAY},
        ]
    elif flags["ablate_text"]:
        param_groups = [{
            "params": model.gnn.parameters(),
            "lr": canonical["gnn_lr"],
        }]
    elif has_trainable_clip:
        clip_params = [
            p for p in model.gnn.clip_model.parameters() if p.requires_grad
        ]
        proj_params = list(model.gnn.image_projection.parameters())
        exclude_ids = set(id(p) for p in clip_params) | set(id(p) for p in proj_params)
        gnn_only_params = [
            p for p in model.gnn.parameters() if id(p) not in exclude_ids
        ]
        param_groups = [
            {"params": model.transformer.parameters(),
             "lr": canonical["transformer_lr"]},
            {"params": gnn_only_params, "lr": canonical["gnn_lr"]},
            {"params": clip_params, "lr": cfg.IMAGE_LR,
             "weight_decay": cfg.IMAGE_WEIGHT_DECAY},
            {"params": proj_params, "lr": cfg.IMAGE_PROJECTION_LR,
             "weight_decay": cfg.IMAGE_WEIGHT_DECAY},
        ]
    else:
        param_groups = [
            {"params": model.transformer.parameters(),
             "lr": canonical["transformer_lr"]},
            {"params": model.gnn.parameters(), "lr": canonical["gnn_lr"]},
        ]

    optimizer = torch.optim.Adam(param_groups)
    return model, optimizer, model_params, canonical, use_images


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_modality_ablation(variants, datasets, rank=0, world_size=1):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    output_base = os.path.join(MODALITY_ABLATION_OUTPUT_BASE, run_id)
    training_seeds = resolve_seed_list(
        getattr(cfg, "MODALITY_ABLATION_SEEDS", None),
        cfg.MODALITY_ABLATION_SEED,
        use_seed=cfg.USE_TRAINING_SEED
    )

    if rank == 0:
        os.makedirs(output_base, exist_ok=True)
        os.makedirs(MODALITY_ABLATION_CACHE_BASE, exist_ok=True)
    _barrier()

    logger = (
        Logger(os.path.join(output_base, "modality_ablation.log"))
        if rank == 0 else None
    )

    metrics_summary = None
    metrics_summary_path = None
    if rank == 0:
        log_file = os.path.join(output_base, "modality_ablation.log")
        metrics_summary = create_summary(
            run_type="modality_ablation",
            run_id=run_id,
            runner="modality_ablation_runner.py",
            seed=training_seeds[0] if len(training_seeds) == 1 else None,
            settings={
                "use_training_seed": cfg.USE_TRAINING_SEED,
                "seed_list": training_seeds,
                "model_type": ABLATION_MODEL,
                "variants": variants,
                "datasets": datasets,
                "use_images": cfg.USE_IMAGE_FEATURES,
                "future_horizons": cfg.USE_FUTURE_HORIZONS,
                "world_size": world_size,
                "max_epochs": TEXT_GRAPHSAGE_MAX_EPOCHS,
            },
        )
        add_source(metrics_summary, log_file=log_file, output_dir=output_base)
        metrics_summary_path = write_summary(metrics_summary)
        logger.section("MODALITY ABLATION")
        logger.log(f"Model: {ABLATION_MODEL}")
        logger.log(f"Variants: {variants}")
        logger.log(f"Datasets/windows: {datasets}")
        logger.log(f"Output: {output_base}")
        logger.log(f"Cache: {MODALITY_ABLATION_CACHE_BASE}")
        logger.log(f"DDP world_size: {world_size}")
        with open(os.path.join(output_base, "config_snapshot.json"), "w") as f:
            json.dump(convert_numpy_types({
                "model": ABLATION_MODEL,
                "variants": variants,
                "datasets": datasets,
                "flags": {v: VALID_VARIANTS[v] for v in variants},
                "canonical_params": CANONICAL_PARAMS[ABLATION_MODEL],
                "text_projection_dim": cfg.TEXT_PROJECTION_DIM,
                "image_projection_dim": cfg.IMAGE_PROJECTION_DIM,
                "freeze_image_encoder": cfg.FREEZE_IMAGE_ENCODER,
                "unfreeze_last_n_layers": cfg.UNFREEZE_LAST_N_LAYERS,
                "use_image_features": cfg.USE_IMAGE_FEATURES,
                "use_training_seed": cfg.USE_TRAINING_SEED,
                "training_seeds": training_seeds,
                "max_epochs": TEXT_GRAPHSAGE_MAX_EPOCHS,
                "scheduler": TEXT_GRAPHSAGE_SCHEDULER,
            }), f, indent=2)

    all_results = {variant: {} for variant in variants}
    accum_steps = _compute_text_accum_steps(world_size)

    load_images = any(
        not VALID_VARIANTS[variant]["ablate_image"] for variant in variants
    )

    for dataset_name, windows in datasets.items():
        for variant in variants:
            all_results[variant].setdefault(dataset_name, {})

        for window_minutes in windows:
            entry = _prepare_dataset_window(
                dataset_name, window_minutes, output_base,
                logger, rank, world_size, load_images=load_images,
            )

            for variant in variants:
                flags = VALID_VARIANTS[variant]
                _setup_dataset_config(dataset_name)

                variant_dir = os.path.join(
                    output_base, dataset_name, f"{window_minutes}min", variant
                )
                if rank == 0:
                    os.makedirs(variant_dir, exist_ok=True)
                    logger.section(
                        f"{variant.upper()} | {dataset_name} | "
                        f"{window_minutes}min"
                    )
                    logger.log(f"Flags: {flags}")
                _barrier()

                for seed in training_seeds:
                    if cfg.USE_TRAINING_SEED:
                        cfg.MODALITY_ABLATION_SEED = seed
                        seed_everything(seed)
                        if rank == 0:
                            logger.log(
                                f"\n[seed] modality ablation training seed = {seed}"
                            )

                    seed_label = "noseed" if seed is None else f"seed_{seed}"
                    result_dir = os.path.join(variant_dir, seed_label)
                    if rank == 0:
                        os.makedirs(result_dir, exist_ok=True)
                    _barrier()

                    train_loader = _build_pyg_loader(
                        entry["train_ds"], TEXT_GRAPHSAGE_BATCH_SIZE,
                        shuffle=True, rank=rank, world_size=world_size,
                        seed=seed if cfg.USE_TRAINING_SEED else None,
                    )
                    val_loader = _build_pyg_loader(
                        entry["val_ds"], TEXT_GRAPHSAGE_BATCH_SIZE,
                        shuffle=False, rank=rank, world_size=world_size,
                    )
                    test_loader = PyGDataLoader(
                        entry["test_ds"], batch_size=TEXT_GRAPHSAGE_BATCH_SIZE,
                        shuffle=False, num_workers=0,
                    )

                    model, optimizer, model_params, canonical, use_images = (
                        _build_model_and_optimizer(entry, dataset_name, variant, rank)
                    )
                    scheduler, scheduler_config = _build_text_scheduler(
                        optimizer, train_loader, accum_steps,
                        TEXT_GRAPHSAGE_MAX_EPOCHS,
                    )

                    if rank == 0:
                        logger.log(f"use_images: {use_images}")
                        logger.log(
                            f"Parameters: trainable={model.get_num_parameters():,} "
                            f"total={_count_all_parameters(model):,}"
                        )
                        logger.log(f"accumulation_steps: {accum_steps}")
                        logger.log(f"Scheduler: {scheduler_config}")

                    model, history = train_model(
                        model, ABLATION_MODEL, train_loader, val_loader,
                        optimizer, cfg.DEVICE,
                        max_epochs=TEXT_GRAPHSAGE_MAX_EPOCHS,
                        rank=rank, world_size=world_size,
                        accumulation_steps=accum_steps,
                        scheduler=scheduler,
                        scheduler_config=scheduler_config,
                    )

                    _barrier()

                    if rank == 0:
                        y_pred, y_true, tree_ids = _collect_predictions_with_tree_ids(
                            model, test_loader, cfg.DEVICE
                        )
                        test_results = evaluate_predictions(
                            y_true, y_pred, feat_cfg.OUTPUT_TARGETS
                        )

                        _log_target_metrics(logger, test_results)

                        model_name = (
                            f"{variant}_{dataset_name}_{window_minutes}min_{seed_label}"
                        )
                        plot_training_curves(
                            history,
                            os.path.join(result_dir, "training_curve.png"),
                            model_name,
                        )
                        plot_prediction_scatter(
                            y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
                            result_dir, model_name,
                        )
                        plot_prediction_scatter_linear(
                            y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
                            result_dir, model_name,
                        )
                        plot_residuals(
                            y_true, y_pred, feat_cfg.OUTPUT_TARGETS,
                            result_dir, model_name,
                        )
                        prediction_path = _save_predictions_with_tree_ids(
                            y_true, y_pred, tree_ids, feat_cfg.OUTPUT_TARGETS,
                            result_dir, model_name,
                        )

                        torch.save(model.state_dict(),
                                   os.path.join(result_dir, "model.pt"))

                        result_payload = {
                            "variant": variant,
                            "ablation_flags": flags,
                            "dataset": dataset_name,
                            "window_minutes": window_minutes,
                            "model_type": ABLATION_MODEL,
                            "targets": feat_cfg.OUTPUT_TARGETS,
                            "hyperparameters": canonical,
                            "model_params": model_params,
                            "use_images": use_images,
                            "use_training_seed": cfg.USE_TRAINING_SEED,
                            "training_seed": seed,
                            "test_metrics": test_results,
                            "training_epochs": len(history["train_loss"]),
                            "final_train_loss": history["train_loss"][-1],
                            "final_val_loss": history["val_loss"][-1],
                            "scheduler": scheduler_config,
                            "trainable_parameters": model.get_num_parameters(),
                            "total_parameters": _count_all_parameters(model),
                            "prediction_csv": prediction_path,
                        }

                        with open(os.path.join(result_dir, "results.json"), "w") as f:
                            json.dump(convert_numpy_types(result_payload), f, indent=2)

                        all_results[variant][dataset_name].setdefault(
                            f"{window_minutes}min", {}
                        )[seed_label] = convert_numpy_types(test_results)
                        add_metric_records(
                            metrics_summary,
                            test_results,
                            dataset=dataset_name,
                            model_type=ABLATION_MODEL,
                            window_minutes=window_minutes,
                            window_label=f"{window_minutes}min",
                            variant=variant,
                            slot_idx=None,
                            seed=seed,
                        )
                        metrics_summary_path = write_summary(metrics_summary)

                        del y_pred, y_true, tree_ids, test_results
                        del result_payload, prediction_path

                    del model, optimizer, train_loader, val_loader, test_loader, history
                    _cleanup_memory()
                    _barrier()

            del entry
            _cleanup_memory()
            _barrier()

    if rank == 0:
        logger.section("MODALITY ABLATION SUMMARY")
        for variant in variants:
            logger.log(f"\nVARIANT: {variant}")
            for dataset_name, windows in datasets.items():
                for window_minutes in windows:
                    seed_metrics = all_results.get(variant, {}).get(
                        dataset_name, {}
                    ).get(f"{window_minutes}min", {})
                    logger.log(f"\n  {dataset_name} {window_minutes}min")
                    for seed_label, metrics in seed_metrics.items():
                        logger.log(f"    [{seed_label}]")
                        for target in feat_cfg.OUTPUT_TARGETS:
                            if target not in metrics:
                                continue
                            m = metrics[target]
                            logger.log(
                                f"      {target:<24} "
                                f"MSE={m['MSE']:.3f} "
                                f"R2={m['R2']:.3f} "
                                f"Pearson={m['Pearson']:.3f} "
                                f"Spearman={m['Spearman']:.3f}"
                            )

        summary_path = os.path.join(output_base, "summary_results.json")
        with open(summary_path, "w") as f:
            json.dump(convert_numpy_types(all_results), f, indent=2)

        summary_json_path = write_summary(metrics_summary)
        if summary_json_path:
            logger.log(f"Normalized metrics summary: {summary_json_path}")
        logger.log(f"\nComplete. Summary: {summary_path}")
        logger.log(f"Output directory: {output_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Run TextGraphSAGE modality ablations."
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants, e.g. no_text,no_image. "
             "Defaults to ABLATION_VARIANTS in this file.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets to keep from config, e.g. gaming,futurology.",
    )
    parser.add_argument(
        "--windows",
        default=None,
        help="Window override, e.g. gaming:20,50,90;futurology:30,90,180.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override TEXT_GRAPHSAGE_MAX_EPOCHS for smoke tests or retries.",
    )
    args = parser.parse_args()

    global TEXT_GRAPHSAGE_MAX_EPOCHS
    if args.max_epochs is not None:
        TEXT_GRAPHSAGE_MAX_EPOCHS = args.max_epochs

    variants, datasets = _resolve_run_config(args)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # Rank 0 can spend a long time building CPU feature caches while the
        # other ranks wait at barriers, so use a longer DDP watchdog timeout.
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
        )
        torch.cuda.set_device(local_rank)
        cfg.DEVICE = torch.device(f"cuda:{local_rank}")
        cfg.DEVICE_TRANSFORMER = cfg.DEVICE
        cfg.DEVICE_GNN = cfg.DEVICE

        if rank == 0:
            print(f"[DDP] world_size={world_size}")

    run_modality_ablation(variants, datasets, rank=rank, world_size=world_size)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
