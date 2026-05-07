from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy import stats


METRIC_NAMES = ["MSE", "R2", "Spearman", "Pearson", "MAE", "MAPE", "RMSE"]


def _round_metric(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return float(value)
    return round(float(value), 3)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = y_true != 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _safe_corr(func, x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    corr = func(x, y)
    if hasattr(corr, "statistic"):
        corr = corr.statistic
    elif isinstance(corr, tuple):
        corr = corr[0]
    return float(corr)


def _compute_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {
        "MSE": mse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Spearman": _safe_corr(stats.spearmanr, y_true, y_pred),
        "Pearson": _safe_corr(stats.pearsonr, y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }
    return {name: _round_metric(value) for name, value in out.items()}


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: List[str],
    target_indices: Optional[Iterable[int]] = None,
) -> Dict[str, Dict[str, float]]:
    if target_indices is None:
        target_indices = range(len(targets))
    target_indices = list(target_indices)

    out: Dict[str, Dict[str, float]] = {}
    for idx in target_indices:
        target = targets[idx]
        out[target] = _compute_metric_bundle(y_true[:, idx], y_pred[:, idx])

    yt_flat = y_true[:, target_indices].reshape(-1)
    yp_flat = y_pred[:, target_indices].reshape(-1)
    out["avg"] = _compute_metric_bundle(yt_flat, yp_flat)
    return out
