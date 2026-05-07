"""
Evaluation metrics.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

import config


def get_predictions(model, data_loader, device):
    """Get predictions and metadata from model."""
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    tree_ids = []
    horizon_tags = []

    with torch.no_grad():
        for batch in data_loader:
            paths = batch["paths"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"]
            target_mask = batch["target_mask"]
            tree_sizes = batch["tree_sizes"].to(device)

            predictions = model(paths, global_features, tree_sizes)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())
            all_masks.append(target_mask.numpy())
            tree_ids.extend(batch["tree_ids"])
            horizon_tags.extend(batch["horizon_tags"])

    return {
        "predictions": np.vstack(all_preds),
        "targets": np.vstack(all_targets),
        "target_masks": np.vstack(all_masks),
        "tree_ids": tree_ids,
        "horizon_tags": horizon_tags,
    }


def _safe_corr(func, y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return np.nan
    try:
        return func(y_true, y_pred)[0]
    except Exception:
        return np.nan


def _compute_metrics(y_true, y_pred):
    diff = y_pred - y_true
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = float(np.sqrt(mse))

    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = np.mean(np.abs(diff[nonzero_mask] / y_true[nonzero_mask])) * 100.0
    else:
        mape = np.nan

    if len(y_true) >= 2 and not np.allclose(y_true, y_true[0]):
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    return {
        "MSE": mse,
        "R2": r2,
        "Spearman": _safe_corr(spearmanr, y_true, y_pred),
        "Pearson": _safe_corr(pearsonr, y_true, y_pred),
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
    }


def _round_metrics(metrics):
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            rounded[key] = _round_metrics(value)
        elif isinstance(value, (float, np.floating, int, np.integer)):
            rounded[key] = None if np.isnan(value) else round(float(value), 3)
        else:
            rounded[key] = value
    return rounded


def build_prediction_table(prediction_bundle, space="raw"):
    """Build an exportable prediction table in log or raw space."""
    rows = []
    if space == "raw":
        preds = np.expm1(prediction_bundle["predictions"])
        targets = np.expm1(prediction_bundle["targets"])
    elif space == "log":
        preds = prediction_bundle["predictions"]
        targets = prediction_bundle["targets"]
    else:
        raise ValueError(f"Unsupported prediction table space: {space}")

    for row_idx, tree_id in enumerate(prediction_bundle["tree_ids"]):
        horizon_tag = prediction_bundle["horizon_tags"][row_idx]
        target_mask = prediction_bundle["target_masks"][row_idx]
        row = {"tree_id": tree_id, "horizon_tag": horizon_tag}
        for idx, target_name in enumerate(config.OUTPUT_TARGETS):
            gt_value = targets[row_idx, idx]
            pred_value = preds[row_idx, idx]
            if horizon_tag != "final" and target_name == "root_score":
                gt_value = np.nan
                pred_value = np.nan
            row[f"{target_name}_gt"] = gt_value
            row[f"{target_name}_pred"] = pred_value
            row[f"{target_name}_supervised"] = bool(target_mask[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(prediction_bundle):
    """
    Compute metrics by horizon and target in log space.
    """
    y_true = prediction_bundle["targets"]
    y_pred = prediction_bundle["predictions"]
    masks = prediction_bundle["target_masks"].astype(bool)
    horizon_tags = np.array(prediction_bundle["horizon_tags"], dtype=object)

    results = {}
    for horizon_tag in config.HORIZON_TAGS:
        horizon_idx = horizon_tags == horizon_tag
        if not horizon_idx.any():
            continue

        horizon_results = {}
        for target_idx, target_name in enumerate(config.OUTPUT_TARGETS):
            valid_idx = horizon_idx & masks[:, target_idx]
            if not valid_idx.any():
                continue

            target_metrics = _compute_metrics(
                y_true[valid_idx, target_idx],
                y_pred[valid_idx, target_idx],
            )
            horizon_results[target_name] = _round_metrics(target_metrics)

        if horizon_results:
            results[horizon_tag] = horizon_results

    return results
