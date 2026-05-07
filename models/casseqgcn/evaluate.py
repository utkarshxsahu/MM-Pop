"""
Evaluation utilities for CasSeqGCN.
"""

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


METRIC_NAMES = ["MSE", "R2", "Spearman", "Pearson", "MAE", "MAPE", "RMSE"]


def evaluate_predictions(pred_bundle, target_names, group_by_horizon=False):
    """Compute masked metrics for the full prediction set and optionally by horizon."""
    preds = pred_bundle["preds"]
    targets = pred_bundle["targets"]
    target_mask = pred_bundle["target_mask"]
    horizon_tags = np.asarray(pred_bundle["horizon_tags"])

    results = {"overall": _evaluate_block(preds, targets, target_mask, target_names)}
    if group_by_horizon:
        results["by_horizon"] = {}
        for horizon in np.unique(horizon_tags):
            horizon_mask = horizon_tags == horizon
            results["by_horizon"][str(horizon)] = _evaluate_block(
                preds[horizon_mask],
                targets[horizon_mask],
                target_mask[horizon_mask],
                target_names,
            )
    return results


def _evaluate_block(preds, targets, target_mask, target_names):
    block = {}
    for idx, target_name in enumerate(target_names):
        valid_rows = target_mask[:, idx].astype(bool)
        if not np.any(valid_rows):
            continue
        y_true = targets[valid_rows, idx]
        y_pred = preds[valid_rows, idx]
        block[target_name] = _compute_metrics(y_true, y_pred)
    return block


def _compute_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {metric_name: math.nan for metric_name in METRIC_NAMES}

    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs(diff[nonzero_mask] / y_true[nonzero_mask])) * 100.0)
    else:
        mape = math.nan

    r2 = _safe_r2(y_true, y_pred)
    pearson = _safe_corr(y_true, y_pred, pearsonr)
    spearman = _safe_corr(y_true, y_pred, spearmanr)

    return {
        "MSE": mse,
        "R2": r2,
        "Spearman": spearman,
        "Pearson": pearson,
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
    }


def _safe_r2(y_true, y_pred):
    if y_true.size < 2:
        return 0.0
    try:
        value = float(r2_score(y_true, y_pred))
        return 0.0 if abs(value) < 5e-4 else value
    except Exception:
        return 0.0


def _safe_corr(y_true, y_pred, corr_fn):
    if y_true.size < 2:
        return 0.0
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0
    try:
        value = float(corr_fn(y_true, y_pred)[0])
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return 0.0 if abs(value) < 5e-4 else value
    except Exception:
        return 0.0


def round_result_metrics(results, decimals=3):
    """Round all numeric metrics for JSON/log output."""
    if isinstance(results, dict):
        return {key: round_result_metrics(value, decimals=decimals) for key, value in results.items()}
    if isinstance(results, list):
        return [round_result_metrics(value, decimals=decimals) for value in results]
    if isinstance(results, (float, np.floating)):
        if math.isnan(results) or math.isinf(results):
            return float(results)
        return round(float(results), decimals)
    if isinstance(results, (int, np.integer)):
        return int(results)
    return results
