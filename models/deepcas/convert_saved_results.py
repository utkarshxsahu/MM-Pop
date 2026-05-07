"""
Convert saved DeepCas run metrics from raw/original space to log1p space.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


TARGET_NAMES = [
    "max_width",
    "max_depth",
    "structural_virality",
    "num_posts",
    "num_unique_users",
    "root_score",
]


def safe_corr(func, y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return np.nan
    try:
        return func(y_true, y_pred)[0]
    except Exception:
        return np.nan


def compute_metrics(y_true, y_pred):
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = float(np.mean(np.abs(diff[nonzero_mask] / y_true[nonzero_mask])) * 100.0)
    else:
        mape = np.nan

    if len(y_true) >= 2 and not np.allclose(y_true, y_true[0]):
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan

    return {
        "MSE": mse,
        "R2": r2,
        "Spearman": safe_corr(spearmanr, y_true, y_pred),
        "Pearson": safe_corr(pearsonr, y_true, y_pred),
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
    }


def round_metrics(metrics):
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            rounded[key] = round_metrics(value)
        elif isinstance(value, (float, np.floating, int, np.integer)):
            rounded[key] = None if np.isnan(value) else round(float(value), 3)
        else:
            rounded[key] = value
    return rounded


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_log_file(path, rows):
    lines = [
        "dataset\twindow\thorizon\texperiment\tmetric_space\ttarget\tMSE\tSpearman\tR2",
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["dataset"]),
                    str(row["window"]),
                    str(row["horizon"]),
                    str(row["experiment"]),
                    str(row["metric_space"]),
                    str(row["target"]),
                    str(row["MSE"]),
                    str(row["Spearman"]),
                    str(row["R2"]),
                ]
            )
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_metadata(results_json_path):
    if not os.path.exists(results_json_path):
        return {
            "experiment": "unknown",
            "future_horizons_enabled": None,
            "window_minutes": None,
        }
    with open(results_json_path) as f:
        data = json.load(f)
    return {
        "experiment": data.get("experiment", "unknown"),
        "future_horizons_enabled": data.get("future_horizons_enabled"),
        "window_minutes": data.get("window_minutes"),
    }


def discover_run(run_dir):
    datasets = {}
    warnings = []

    for entry in sorted(os.listdir(run_dir)):
        dataset_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(dataset_dir):
            continue

        windows = {}
        for child in sorted(os.listdir(dataset_dir)):
            window_dir = os.path.join(dataset_dir, child)
            if not os.path.isdir(window_dir) or not child.endswith("min"):
                continue

            predictions_path = os.path.join(window_dir, "predictions.csv")
            results_json_path = os.path.join(window_dir, "results.json")
            if not os.path.exists(predictions_path):
                warnings.append(f"Skipping {window_dir}: missing predictions.csv")
                continue

            windows[child] = {
                "window_dir": window_dir,
                "predictions_path": predictions_path,
                "results_json_path": results_json_path,
                "metadata": load_metadata(results_json_path),
            }

        if windows:
            datasets[entry] = windows

    return datasets, warnings


def parse_bool_series(series):
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def convert_window(dataset_name, window_label, window_info):
    df = pd.read_csv(window_info["predictions_path"])
    experiment = window_info["metadata"]["experiment"]
    horizons = list(dict.fromkeys(df["horizon_tag"].astype(str).tolist()))

    results_by_horizon = {}
    log_rows = []

    for horizon_tag in horizons:
        horizon_df = df[df["horizon_tag"].astype(str) == horizon_tag].copy()
        horizon_results = {}

        for target_name in TARGET_NAMES:
            gt_col = f"{target_name}_gt"
            pred_col = f"{target_name}_pred"
            supervised_col = f"{target_name}_supervised"

            if gt_col not in horizon_df.columns or pred_col not in horizon_df.columns:
                continue

            valid_mask = pd.Series(True, index=horizon_df.index)
            if supervised_col in horizon_df.columns:
                valid_mask &= parse_bool_series(horizon_df[supervised_col])
            valid_mask &= horizon_df[gt_col].notna()
            valid_mask &= horizon_df[pred_col].notna()

            if not valid_mask.any():
                continue

            gt_vals = horizon_df.loc[valid_mask, gt_col].astype(float).to_numpy()
            pred_vals = horizon_df.loc[valid_mask, pred_col].astype(float).to_numpy()

            gt_log = np.log1p(np.clip(gt_vals, a_min=0, a_max=None))
            pred_log = np.log1p(np.clip(pred_vals, a_min=0, a_max=None))

            metrics = round_metrics(compute_metrics(gt_log, pred_log))
            horizon_results[target_name] = metrics
            log_rows.append(
                {
                    "dataset": dataset_name,
                    "window": window_label,
                    "horizon": horizon_tag,
                    "experiment": experiment,
                    "metric_space": "log1p",
                    "target": target_name,
                    "MSE": metrics.get("MSE"),
                    "Spearman": metrics.get("Spearman"),
                    "R2": metrics.get("R2"),
                }
            )

        if horizon_results:
            results_by_horizon[horizon_tag] = horizon_results

    converted_results = {
        "dataset": dataset_name,
        "window_minutes": window_info["metadata"]["window_minutes"],
        "experiment": experiment,
        "metric_space": "log1p",
        "future_horizons_enabled": window_info["metadata"]["future_horizons_enabled"],
        "test_metrics": results_by_horizon,
        "source_predictions_file": os.path.basename(window_info["predictions_path"]),
    }

    return converted_results, log_rows


def convert_run(run_dir):
    datasets, warnings = discover_run(run_dir)
    if not datasets:
        raise FileNotFoundError(f"No dataset/window folders with predictions.csv found under {run_dir}")

    combined_summary = {}
    combined_log_rows = []

    for dataset_name, windows in datasets.items():
        dataset_summary = {}
        dataset_log_rows = []

        for window_label, window_info in windows.items():
            converted_results, log_rows = convert_window(dataset_name, window_label, window_info)
            dataset_summary[window_label] = converted_results
            dataset_log_rows.extend(log_rows)
            combined_log_rows.extend(log_rows)

            with open(os.path.join(window_info["window_dir"], "converted_results.json"), "w") as f:
                json.dump(converted_results, f, indent=2, default=json_default)

        combined_summary[dataset_name] = dataset_summary

        with open(os.path.join(run_dir, dataset_name, "converted_summary_results.json"), "w") as f:
            json.dump(dataset_summary, f, indent=2, default=json_default)
        write_log_file(
            os.path.join(run_dir, dataset_name, "converted_experiment.log"),
            dataset_log_rows,
        )

    with open(os.path.join(run_dir, "converted_summary_results.json"), "w") as f:
        json.dump(combined_summary, f, indent=2, default=json_default)
    write_log_file(os.path.join(run_dir, "converted_combined_experiment.log"), combined_log_rows)

    if warnings:
        with open(os.path.join(run_dir, "converted_warnings.txt"), "w") as f:
            f.write("\n".join(warnings) + "\n")

    return {
        "datasets": len(datasets),
        "windows": sum(len(windows) for windows in datasets.values()),
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert saved DeepCas results from raw-space metrics to log1p-space metrics."
    )
    parser.add_argument("run_dir", help="Path to an existing deepcas_run_* folder")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    summary = convert_run(run_dir)

    print(f"Converted run folder: {run_dir}")
    print(f"Datasets converted: {summary['datasets']}")
    print(f"Windows converted: {summary['windows']}")
    if summary["warnings"]:
        print(f"Warnings written: {os.path.join(run_dir, 'converted_warnings.txt')}")


if __name__ == "__main__":
    main()
