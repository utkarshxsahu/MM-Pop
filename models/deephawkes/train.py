from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data import DeepHawkesPathsDataset, collate_paths_batch
from metrics import METRIC_NAMES, per_target_metrics
from model import DeepHawkes
from utils import (
    append_summary_log,
    create_dataset_output_dir,
    create_run_output_dir,
    format_metrics_for_logging,
    plot_training_curves,
    save_checkpoint,
    set_seed,
    setup_logging,
)


SUMMARY_METRICS = ["MSE", "Spearman", "R2"]


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, horizon_mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (y_pred - y_true) ** 2
    if horizon_mask is not None:
        mask = horizon_mask.float()
        valid = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / valid
    return torch.mean(loss)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _metrics_for_scope(y_true: np.ndarray, y_pred: np.ndarray, horizon_idx: np.ndarray | None = None) -> Dict[str, Dict]:
    results = {"overall": per_target_metrics(y_true, y_pred, config.TARGETS)}
    if horizon_idx is None:
        return results

    for horizon_label in config.HORIZON_HOURS:
        idx = config.HORIZON_INDEX[horizon_label]
        mask = horizon_idx == idx
        if not np.any(mask):
            continue
        if horizon_label == "final":
            target_indices = range(len(config.TARGETS))
        else:
            target_indices = range(config.ROOT_SCORE_IDX)
        results[horizon_label] = per_target_metrics(y_true[mask], y_pred[mask], config.TARGETS, target_indices=target_indices)
    return results


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    ys, preds, masks, horizon_idxs, tree_ids, horizon_labels = [], [], [], [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = _to_device(batch, device)
        y_true = batch["y"]
        y_pred = model(batch)
        loss = mse_loss(y_pred, y_true, batch["horizon_mask"] if config.USE_FUTURE_HORIZONS else None)

        total_loss += float(loss.item())
        n_batches += 1
        ys.append(y_true.cpu().numpy())
        preds.append(y_pred.cpu().numpy())
        masks.append(batch["horizon_mask"].cpu().numpy())
        horizon_idxs.append(batch["horizon_idx"].cpu().numpy())
        tree_ids.append(batch["tree_id"].cpu().numpy())
        horizon_labels.extend(batch["horizon_label"])

    y_true_np = np.concatenate(ys, axis=0)
    y_pred_np = np.concatenate(preds, axis=0)
    horizon_idx_np = np.concatenate(horizon_idxs, axis=0)
    tree_id_np = np.concatenate(tree_ids, axis=0)

    return {
        "loss": round(total_loss / max(1, n_batches), 3),
        "metrics": _metrics_for_scope(
            y_true_np,
            y_pred_np,
            horizon_idx_np if config.USE_FUTURE_HORIZONS else None,
        ),
        "predictions": {
            "y_true": y_true_np,
            "y_pred": y_pred_np,
            "horizon_idx": horizon_idx_np,
            "tree_id": tree_id_np,
            "horizon_label": horizon_labels,
        },
    }


def _dataset_for_split(
    dataset_name: str,
    dataset_paths: Dict,
    dataset_config: Dict,
    window_minutes: int,
    split: str,
    hparams: Dict,
) -> DeepHawkesPathsDataset:
    split_window_minutes = dataset_config["root_only_split_window"] if window_minutes == 0 else window_minutes
    edges_path = None if window_minutes == 0 else os.path.join(dataset_paths["early_window_dir"], f"edges_{window_minutes}min.parquet")
    limit_trees = config.DEBUG_LIMIT_TREES if config.DEBUG_MODE else None

    return DeepHawkesPathsDataset(
        meta_path=dataset_paths["thread_metadata"],
        posts_path=dataset_paths["thread_posts"],
        edges_path=edges_path,
        window_minutes=window_minutes,
        targets=config.TARGETS,
        num_bins=hparams["num_bins"],
        split=split,
        early_window_dir=dataset_paths["early_window_dir"],
        split_window_minutes=split_window_minutes,
        include_root_path=True,
        use_future_horizons=config.USE_FUTURE_HORIZONS,
        future_horizons_dir=dataset_paths["future_horizons_dir"],
        score_column=dataset_config["root_score_column"],
        limit_trees=limit_trees,
    )


def _build_loader(dataset: DeepHawkesPathsDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_paths_batch,
    )


def _save_predictions_csv(predictions: Dict, output_path: Path) -> None:
    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    tree_ids = predictions["tree_id"]
    horizon_labels = predictions["horizon_label"]

    y_true_actual = np.expm1(y_true)
    y_pred_actual = np.expm1(y_pred)

    rows = []
    for idx in range(len(tree_ids)):
        row = {
            "tree_id": int(tree_ids[idx]),
            "horizon": horizon_labels[idx],
        }
        for target_idx, target in enumerate(config.TARGETS):
            gt_value = y_true_actual[idx, target_idx]
            pred_value = y_pred_actual[idx, target_idx]
            if horizon_labels[idx] != "final" and target_idx == config.ROOT_SCORE_IDX:
                gt_value = float("nan")
                pred_value = float("nan")
            row[f"{target}_gt"] = gt_value
            row[f"{target}_pred"] = pred_value
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: _sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 3)
    return value


def train_one_window(
    dataset_name: str,
    dataset_config: Dict,
    window_minutes: int,
    hparams: Dict,
    output_dir: Path,
    run_dir: Path,
    logger,
    device: torch.device,
) -> Dict:
    dataset_paths = dataset_config["paths"]
    ds_train = _dataset_for_split(dataset_name, dataset_paths, dataset_config, window_minutes, "train", hparams)
    ds_val = _dataset_for_split(dataset_name, dataset_paths, dataset_config, window_minutes, "val", hparams)
    ds_test = _dataset_for_split(dataset_name, dataset_paths, dataset_config, window_minutes, "test", hparams)

    logger.info(f"Loading datasets for {window_minutes}min window")
    logger.info(f"Train={len(ds_train)} Val={len(ds_val)} Test={len(ds_test)} Users={ds_train.num_users}")

    train_loader = _build_loader(ds_train, hparams["batch_size"], shuffle=True)
    val_loader = _build_loader(ds_val, hparams["batch_size"], shuffle=False)
    test_loader = _build_loader(ds_test, hparams["batch_size"], shuffle=False)

    model = DeepHawkes(
        num_users=ds_train.num_users,
        embedding_dim=hparams["embedding_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_bins=hparams["num_bins"],
        out_dim=len(config.TARGETS),
        dropout=hparams["dropout"],
    ).to(device)

    emb_params = list(model.user_emb.parameters())
    other_params = [param for name, param in model.named_parameters() if not name.startswith("user_emb")]
    optim = torch.optim.Adam(
        [
            {"params": emb_params, "lr": hparams["lr_emb"]},
            {"params": other_params, "lr": hparams["lr"]},
        ],
        weight_decay=hparams["weight_decay"],
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = _to_device(batch, device)
            y_true = batch["y"]
            y_pred = model(batch)
            loss = mse_loss(y_pred, y_true, batch["horizon_mask"] if config.USE_FUTURE_HORIZONS else None)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        train_losses.append(round(train_loss, 3))

        val_eval = run_eval(model, val_loader, device)
        val_loss = val_eval["loss"]
        val_losses.append(val_loss)

        logger.info(
            f"Epoch {epoch:03d}/{config.EPOCHS} "
            f"train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
            f"val_r2={val_eval['metrics']['overall']['avg']['R2']:.3f}"
        )

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if config.SAVE_CHECKPOINTS:
                save_checkpoint(
                    model,
                    hparams,
                    epoch,
                    val_loss,
                    output_dir / f"checkpoint_{dataset_name}_{window_minutes}min.pt",
                )
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_val_eval = run_eval(model, val_loader, device)
    test_eval = run_eval(model, test_loader, device)

    window_label = f"{window_minutes}min"
    plot_training_curves(
        train_losses,
        val_losses,
        output_dir / f"training_curves_{window_label}.png",
        title=f"{dataset_name} - {window_label} - {hparams['name']}",
    )
    _save_predictions_csv(test_eval["predictions"], output_dir / f"predictions_{window_label}.csv")

    logger.info("Validation metrics")
    logger.info(format_metrics_for_logging(best_val_eval["metrics"]["overall"], config.TARGETS, METRIC_NAMES))
    if config.USE_FUTURE_HORIZONS:
        for horizon_label in config.HORIZON_HOURS:
            if horizon_label in best_val_eval["metrics"]:
                logger.info(f"Validation horizon={horizon_label}")
                logger.info(format_metrics_for_logging(best_val_eval["metrics"][horizon_label], config.TARGETS, METRIC_NAMES))

    logger.info("Test metrics")
    logger.info(format_metrics_for_logging(test_eval["metrics"]["overall"], config.TARGETS, METRIC_NAMES))
    if config.USE_FUTURE_HORIZONS:
        for horizon_label in config.HORIZON_HOURS:
            if horizon_label in test_eval["metrics"]:
                logger.info(f"Test horizon={horizon_label}")
                logger.info(format_metrics_for_logging(test_eval["metrics"][horizon_label], config.TARGETS, METRIC_NAMES))

    metrics_by_scope = {"overall": test_eval["metrics"]["overall"]}
    if config.USE_FUTURE_HORIZONS:
        for horizon_label in config.HORIZON_HOURS:
            if horizon_label in test_eval["metrics"]:
                metrics_by_scope[horizon_label] = test_eval["metrics"][horizon_label]
    append_summary_log(output_dir, dataset_name, window_label, hparams["name"], metrics_by_scope, config.TARGETS)
    append_summary_log(run_dir, dataset_name, window_label, hparams["name"], metrics_by_scope, config.TARGETS, log_filename="combined_run_summary.log")

    return {
        "window_minutes": window_minutes,
        "window_label": window_label,
        "config_name": hparams["name"],
        "use_future_horizons": config.USE_FUTURE_HORIZONS,
        "best_epoch": best_epoch,
        "best_val_mse": round(best_val_loss, 3),
        "val_metrics": best_val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "summary_metrics": SUMMARY_METRICS,
    }


def train_dataset_all_windows(dataset_name: str, run_dir: Path) -> None:
    set_seed(config.SEED)
    dataset_config = config.DATASETS[dataset_name]
    output_dir = create_dataset_output_dir(run_dir, dataset_name)
    logger = setup_logging(output_dir)

    logger.info(f"Dataset={dataset_name}")
    logger.info(f"Windows={dataset_config['windows']}")
    logger.info(f"Targets={config.TARGETS}")
    logger.info(f"FutureHorizons={config.USE_FUTURE_HORIZONS}")

    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
    logger.info(f"Device={device}")

    results = {
        "dataset": dataset_name,
        "targets": config.TARGETS,
        "use_future_horizons": config.USE_FUTURE_HORIZONS,
        "config": config.HYPERPARAMETER_CONFIG,
        "windows": {},
    }

    for window_minutes in dataset_config["windows"]:
        logger.info(f"{'=' * 80}\nWINDOW {window_minutes}min\n{'=' * 80}")
        window_result = train_one_window(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            window_minutes=window_minutes,
            hparams=config.HYPERPARAMETER_CONFIG,
            output_dir=output_dir,
            run_dir=run_dir,
            logger=logger,
            device=device,
        )
        results["windows"][f"{window_minutes}min"] = _sanitize_for_json(window_result)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(results), f, indent=2)

    logger.info(f"Results saved to {results_path}")


def main() -> None:
    run_dir = create_run_output_dir(config.RESULTS_BASE_DIR)
    for dataset_name in config.DATASETS_TO_RUN:
        if dataset_name not in config.DATASETS:
            print(f"Error: dataset '{dataset_name}' not found in config.DATASETS")
            continue
        try:
            train_dataset_all_windows(dataset_name, run_dir)
        except Exception as exc:
            print(f"Error training dataset {dataset_name}: {exc}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
