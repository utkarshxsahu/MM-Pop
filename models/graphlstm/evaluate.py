"""
Evaluation utilities.
"""

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def _round3(value):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), 3)


def _safe_corr(fn, y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return np.nan
    return fn(y_true, y_pred)[0]


def _safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]):
        return np.nan
    return r2_score(y_true, y_pred)


def compute_target_metrics(y_true_log, y_pred_log):
    y_true = y_true_log
    y_pred = y_pred_log
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))

    denom = np.where(np.abs(y_true) > 1e-8, np.abs(y_true), np.nan)
    mape = float(np.nanmean(np.abs((y_pred - y_true) / denom)) * 100.0) if np.any(~np.isnan(denom)) else np.nan

    return {
        "MSE": mse,
        "R2": _safe_r2(y_true, y_pred),
        "Pearson": _safe_corr(pearsonr, y_true, y_pred),
        "Spearman": _safe_corr(spearmanr, y_true, y_pred),
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
    }


def evaluate_predictions(y_true, y_pred, target_names, mask=None):
    results = {}
    mask = np.ones_like(y_true, dtype=bool) if mask is None else mask.astype(bool)

    for idx, target in enumerate(target_names):
        valid = mask[:, idx]
        if not np.any(valid):
            results[target] = {key: None for key in ["MSE", "R2", "Pearson", "Spearman", "MAE", "MAPE", "RMSE"]}
            continue
        metrics = compute_target_metrics(y_true[valid, idx], y_pred[valid, idx])
        results[target] = {name: _round3(value) for name, value in metrics.items()}
    return results


def evaluate_grouped_predictions(y_true, y_pred, target_names, horizon_tags, mask=None):
    grouped = {}
    horizon_tags = np.asarray(horizon_tags)
    for horizon in sorted(set(horizon_tags.tolist()), key=lambda tag: ["4h", "8h", "16h", "24h", "final"].index(tag)):
        keep = horizon_tags == horizon
        horizon_mask = mask[keep] if mask is not None else None
        grouped[horizon] = evaluate_predictions(y_true[keep], y_pred[keep], target_names, horizon_mask)
    return grouped
