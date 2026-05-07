"""
Normalized metrics summary helpers.

The runners keep their existing detailed JSON/log outputs. This module writes
one compact, flat-record JSON per runner invocation for later aggregation.
"""

import json
import math
import os
from datetime import datetime

import numpy as np


SUMMARY_METRICS = ("MSE", "R2", "Spearman")


def active_training_seed(run_type):
    import config.config as cfg

    if not cfg.USE_TRAINING_SEED:
        return None
    if run_type == "modality_ablation":
        return cfg.MODALITY_ABLATION_SEED
    return cfg.TRAINING_SEED


def create_summary(run_type, run_id, runner, seed, settings=None):
    return {
        "schema_version": 1,
        "run_type": run_type,
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "settings": settings or {},
        "source": {
            "runner": runner,
            "log_files": [],
            "output_dirs": [],
        },
        "records": [],
    }


def add_source(summary, log_file=None, output_dir=None):
    if log_file and log_file not in summary["source"]["log_files"]:
        summary["source"]["log_files"].append(log_file)
    if output_dir and output_dir not in summary["source"]["output_dirs"]:
        summary["source"]["output_dirs"].append(output_dir)


def add_metric_records(summary, metrics_by_target, dataset, model_type,
                       window_minutes=None, window_label=None, variant=None,
                       slot_idx=None, seed=None):
    record_seed = summary["seed"] if seed is None else seed
    summary["records"].extend(make_metric_records(
        metrics_by_target,
        run_type=summary["run_type"],
        seed=record_seed,
        dataset=dataset,
        model_type=model_type,
        window_minutes=window_minutes,
        window_label=window_label,
        variant=variant,
        slot_idx=slot_idx,
    ))


def make_metric_records(metrics_by_target, run_type, seed, dataset, model_type,
                        window_minutes=None, window_label=None, variant=None,
                        slot_idx=None):
    records = []
    for target, metrics in metrics_by_target.items():
        if target == "group_summaries" or not isinstance(metrics, dict):
            continue
        records.append({
            "run_type": run_type,
            "seed": seed,
            "dataset": dataset,
            "window_minutes": window_minutes,
            "window_label": window_label,
            "variant": variant,
            "slot_idx": slot_idx,
            "model_type": model_type,
            "target": target,
            "metrics": {
                name: _round_metric(metrics.get(name))
                for name in SUMMARY_METRICS
            },
        })
    return records


def write_summary(summary, output_path=None):
    if output_path is None:
        import config.config as cfg

        if not cfg.WRITE_METRICS_SUMMARY_JSON:
            return None
        os.makedirs(cfg.METRICS_SUMMARY_DIR, exist_ok=True)
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    seed_list = summary.get("settings", {}).get("seed_list", [])
    if summary["seed"] is None and len(seed_list) > 1:
        seed_label = "multiseed"
    else:
        seed_label = "noseed" if summary["seed"] is None else f"seed{summary['seed']}"
    if output_path is None:
        filename = (
            f"metrics_summary_{summary['run_type']}_"
            f"{_safe_name(summary['run_id'])}_{seed_label}.json"
        )
        path = os.path.join(cfg.METRICS_SUMMARY_DIR, filename)
    else:
        path = output_path

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    os.replace(tmp_path, path)
    return path


def _round_metric(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, 3)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _round_metric(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _safe_name(value):
    return str(value).replace(os.sep, "_").replace(" ", "_")
