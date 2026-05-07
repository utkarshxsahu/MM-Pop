"""
Evaluation metrics — with target-group-level summaries and optional per-horizon breakdowns.
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import config.config as cfg
import config.feature_config as feat_cfg


def get_predictions_mlp(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch[0].to(device)
            Y_pred = model(X_batch)
            all_preds.append(Y_pred.cpu().numpy())
            all_targets.append(batch[1].numpy())
    return np.vstack(all_preds), np.vstack(all_targets)


def get_predictions_mlp_with_horizon(model, data_loader, device):
    """Variant of get_predictions_mlp that also returns horizon_idx arrays."""
    model.eval()
    all_preds, all_targets, all_hidx = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch[0].to(device)
            Y_pred  = model(X_batch)
            all_preds.append(Y_pred.cpu().numpy())
            all_targets.append(batch[1].numpy())
            all_hidx.append(batch[3].numpy())   # (x, y, hmask, hidx)
    return np.vstack(all_preds), np.vstack(all_targets), np.concatenate(all_hidx)


def get_predictions_graphsage(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            Y_pred = model(batch)
            all_preds.append(Y_pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    return np.vstack(all_preds), np.vstack(all_targets)


def get_predictions_graphsage_with_horizon(model, data_loader, device):
    """
    Variant of get_predictions_graphsage that also collects tree_ids and horizon_idx.

    Returns:
        y_pred       : (n_samples, n_targets) predictions in log-space
        y_true       : (n_samples, n_targets) ground truth in log-space
        tree_ids     : list of n_samples tree_id values
        horizon_idxs : (n_samples,) integer array of horizon indices
    """
    model.eval()
    all_preds, all_targets, all_tree_ids, all_hidx = [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            Y_pred = model(batch)
            all_preds.append(Y_pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            all_hidx.append(batch.horizon_idx.cpu().numpy())
            tids = batch.tree_id
            if isinstance(tids, list):
                all_tree_ids.extend(tids)
            else:
                all_tree_ids.append(tids)
    return (np.vstack(all_preds), np.vstack(all_targets),
            all_tree_ids, np.concatenate(all_hidx))


def evaluate_predictions(y_true, y_pred, target_names):
    """
    Compute per-target metrics plus group-level summaries.

    Args:
        y_true       : (n_samples, n_targets) in log-space
        y_pred       : (n_samples, n_targets) in log-space
        target_names : list of target column names (must match OUTPUT_TARGETS order)

    Returns:
        dict with per-target metrics AND a 'group_summaries' sub-dict.
        If y_pred contains NaN/Inf (training diverged), returns NaN metrics
        rather than crashing so the rest of the ablation can continue.
    """
    # Guard: training may have diverged and left NaN weights → NaN predictions.
    # Return sentinel NaN results instead of crashing with sklearn ValueError.
    pred_is_finite = np.isfinite(y_pred).all()
    true_is_finite = np.isfinite(y_true).all()

    results = {}

    for i, target in enumerate(target_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        if not pred_is_finite or not true_is_finite or len(y_t) == 0:
            results[target] = {
                'MSE': float('nan'), 'RMSE': float('nan'),
                'MAE': float('nan'), 'MAPE': float('nan'),
                'R2': float('nan'),
                'Pearson': float('nan'), 'Pearson_pval': float('nan'),
                'Spearman': float('nan'), 'Spearman_pval': float('nan'),
                'group': feat_cfg.TARGET_GROUP_MAP.get(target, ('unknown', 0))[0],
            }
            continue

        mse  = np.mean((y_p - y_t) ** 2)
        rmse = np.sqrt(mse)
        mae  = np.mean(np.abs(y_p - y_t))
        mape = np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + 1e-8))) * 100
        r2   = r2_score(y_t, y_p)
        pearson_corr, pearson_p = pearsonr(y_t, y_p)
        spearman_corr, spearman_p = spearmanr(y_t, y_p)

        results[target] = {
            'MSE':          mse,
            'RMSE':         rmse,
            'MAE':          mae,
            'MAPE':         mape,
            'R2':           r2,
            'Pearson':      pearson_corr,
            'Pearson_pval': pearson_p,
            'Spearman':     spearman_corr,
            'Spearman_pval': spearman_p,
            'group': feat_cfg.TARGET_GROUP_MAP.get(target, ('unknown', 0))[0],
        }

    # Group-level summaries (mean R², mean Pearson per group)
    group_summaries = {}
    for group_name, targets in feat_cfg.TARGET_GROUPS.items():
        group_r2      = np.mean([results[t]['R2']      for t in targets if t in results])
        group_pearson = np.mean([results[t]['Pearson'] for t in targets if t in results])
        group_mse     = np.mean([results[t]['MSE']     for t in targets if t in results])
        group_rmse    = np.mean([results[t]['RMSE']    for t in targets if t in results])
        group_mae     = np.mean([results[t]['MAE']     for t in targets if t in results])
        group_mape    = np.mean([results[t]['MAPE']    for t in targets if t in results])
        group_summaries[group_name] = {
            'mean_R2':      group_r2,
            'mean_Pearson': group_pearson,
            'mean_MSE':     group_mse,
            'mean_RMSE':    group_rmse,
            'mean_MAE':     group_mae,
            'mean_MAPE':    group_mape,
        }

    results['group_summaries'] = group_summaries
    return results


def print_results(results, logger=None):
    """
    Pretty-print evaluation results, organised by group.

    Args:
        results : output of evaluate_predictions
        logger  : optional Logger instance; falls back to print
    """
    def _log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    group_summaries = results.get('group_summaries', {})

    for group_name, targets in feat_cfg.TARGET_GROUPS.items():
        _log(f"\n  ── {group_name.upper()} GROUP ──")
        for target in targets:
            if target not in results:
                continue
            m = results[target]
            _log(f"  {target}:")
            _log(f"    MSE={m['MSE']:.4f}  R²={m['R2']:.4f}  "
                 f"Pearson={m['Pearson']:.4f}  Spearman={m['Spearman']:.4f}")

        if group_name in group_summaries:
            gs = group_summaries[group_name]
            _log(f"  → Group mean:  R²={gs['mean_R2']:.4f}  "
                 f"Pearson={gs['mean_Pearson']:.4f}  MSE={gs['mean_MSE']:.4f}")

    # Overall summary
    all_r2 = np.mean([results[t]['R2'] for t in feat_cfg.OUTPUT_TARGETS if t in results])
    all_pearson = np.mean([results[t]['Pearson'] for t in feat_cfg.OUTPUT_TARGETS if t in results])
    _log(f"\n  ── OVERALL (macro-avg across all targets) ──")
    _log(f"  mean R²={all_r2:.4f}   mean Pearson={all_pearson:.4f}")


# ---------------------------------------------------------------------------
# Future-horizon utilities
# ---------------------------------------------------------------------------

def evaluate_predictions_by_horizon(y_true, y_pred, horizon_idxs):
    """
    Compute per-horizon metrics by splitting predictions on horizon_idxs.

    For intermediate horizons (non-final), root_score is excluded from metrics
    since it was not supervised there.

    Returns:
        dict mapping horizon_name (str) → evaluate_predictions result dict.
    """
    results_by_horizon = {}
    rs_idx = feat_cfg.ROOT_SCORE_IDX
    all_targets = feat_cfg.OUTPUT_TARGETS

    for h_idx, h_name in enumerate(cfg.HORIZON_HOURS):
        sel = (horizon_idxs == h_idx)
        if sel.sum() == 0:
            continue
        y_t = y_true[sel]
        y_p = y_pred[sel]

        if h_name == 'final':
            results_by_horizon[h_name] = evaluate_predictions(y_t, y_p, all_targets)
        else:
            # Exclude root_score column for intermediate horizons
            keep = [i for i in range(len(all_targets)) if i != rs_idx]
            targets_h = [all_targets[i] for i in keep]
            results_by_horizon[h_name] = evaluate_predictions(
                y_t[:, keep], y_p[:, keep], targets_h
            )

    return results_by_horizon


def save_horizon_predictions(y_true, y_pred, tree_ids, horizon_idxs, output_path):
    """
    Save a per-tree, per-horizon predictions CSV.

    Columns: tree_id, horizon, {target}_gt, {target}_pred  (for each target).
    Values are in actual space (expm1 applied).
    root_score_gt and root_score_pred are NaN for non-final horizons.

    Args:
        y_true       : (n_samples, n_targets) log-space ground truth
        y_pred       : (n_samples, n_targets) log-space predictions
        tree_ids     : list of n_samples tree_id values
        horizon_idxs : (n_samples,) integer horizon indices
        output_path  : file path for the output CSV
    """
    targets    = feat_cfg.OUTPUT_TARGETS
    rs_idx     = feat_cfg.ROOT_SCORE_IDX
    h_names    = cfg.HORIZON_HOURS

    rows = []
    for i in range(len(tree_ids)):
        h_name = h_names[int(horizon_idxs[i])]
        row = {'tree_id': tree_ids[i], 'horizon': h_name}
        for j, t in enumerate(targets):
            gt_val   = float(np.expm1(y_true[i, j]))
            pred_val = float(np.expm1(y_pred[i, j]))
            if t == 'root_score' and h_name != 'final':
                gt_val   = float('nan')
                pred_val = float('nan')
            row[f'{t}_gt']   = gt_val
            row[f'{t}_pred'] = pred_val
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved horizon predictions ({len(df):,} rows): {output_path}")