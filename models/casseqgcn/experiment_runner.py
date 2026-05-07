"""
Unified experiment runner for CasSeqGCN.
"""

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from casseqgcn_dataset import CasSeqGCNDataset, collate_fn
from casseqgcn_model import CasSeqGCN
from casseqgcn_train import get_predictions, train_model
from data_loader import load_dataset
from evaluate import round_result_metrics, evaluate_predictions
from preprocessing import Preprocessor
from visualization import (
    plot_prediction_scatter,
    plot_prediction_scatter_linear,
    plot_residuals,
    plot_training_curves,
)

RUN_CONFIG = {
    "datasets": ["bluesky", "ama", "gaming", "futurology"],
    "windows": ["all"],
    "include_root_only": True,
    "future_horizons": True,
    "run_name": None,
    "output_root": None,
    "max_epochs": config.MAX_EPOCHS,
    "patience": config.PATIENCE,
    "trial_epochs": config.TRIAL_EPOCHS,
    "trial_patience": config.TRIAL_PATIENCE,
    "batch_size": config.BATCH_SIZE,
    "num_workers": config.NUM_WORKERS,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run CasSeqGCN experiments.")
    parser.add_argument("--datasets", nargs="+", default=None, help="'all' or dataset names")
    parser.add_argument("--windows", nargs="+", default=None, help="'all', integer windows, and/or root_only")
    parser.add_argument("--include-root-only", dest="include_root_only", action="store_true", help="Add root-only mode")
    parser.add_argument("--no-include-root-only", dest="include_root_only", action="store_false", help="Disable root-only mode")
    parser.add_argument("--future-horizons", dest="future_horizons", action="store_true", help="Enable future-horizon expansion")
    parser.add_argument("--no-future-horizons", dest="future_horizons", action="store_false", help="Disable future-horizon expansion")
    parser.add_argument("--run-name", default=None, help="Optional output subdirectory name")
    parser.add_argument("--output-root", default=None, help="Optional directory for combined logs")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--trial-epochs", type=int, default=None)
    parser.add_argument("--trial-patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=15)
    parser.set_defaults(include_root_only=None, future_horizons=None)
    return parser.parse_args()


def merge_args_with_run_config(args):
    merged = argparse.Namespace(**RUN_CONFIG)
    for key, value in vars(args).items():
        if value is not None:
            setattr(merged, key, value)
    return merged


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_datasets(dataset_args):
    if "all" in dataset_args:
        return list(config.DATASET_SPECS.keys())
    for dataset_name in dataset_args:
        config.get_dataset_spec(dataset_name)
    return dataset_args


def resolve_windows(dataset_name, window_args, include_root_only):
    if "all" in window_args:
        requested = "all"
    else:
        requested = []
        for value in window_args:
            if value == config.ROOT_ONLY_WINDOW:
                requested.append(config.ROOT_ONLY_WINDOW)
            else:
                requested.append(int(value))

    windows = config.get_window_labels(
        dataset_name,
        include_root_only=include_root_only,
        requested_windows=requested,
    )
    valid_windows = set(config.get_dataset_spec(dataset_name)["windows"]) | {config.ROOT_ONLY_WINDOW}
    filtered = [window for window in windows if window in valid_windows]
    if not filtered:
        raise ValueError(f"No valid windows selected for dataset={dataset_name}")
    return filtered


def make_loader(dataset_obj, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_datasets(data, preprocessor, future_horizons):
    def _dataset(tree_ids):
        return CasSeqGCNDataset(
            tree_ids=tree_ids,
            edges_df=data["edges"],
            meta_df=data["meta"],
            posts_df=data["posts"],
            preprocessor=preprocessor,
            future_horizons=future_horizons,
            future_horizon_gt=data["future_horizon_gt"],
            future_horizon_valid_mask=data["future_horizon_valid_mask"],
        )

    return _dataset(data["train_ids"]), _dataset(data["val_ids"]), _dataset(data["test_ids"])


def run_trial(model_cfg, train_loader, val_loader, args):
    model = CasSeqGCN(model_cfg).to(config.DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
    )
    _, history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        config.DEVICE,
        max_epochs=args.trial_epochs,
        patience=args.trial_patience,
    )
    return min(history["val_loss"])


def run_full(model_cfg, train_loader, val_loader, args):
    model = CasSeqGCN(model_cfg).to(config.DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        config.DEVICE,
        max_epochs=args.max_epochs,
        patience=args.patience,
        scheduler=scheduler,
    )
    return model, history


def create_output_dirs(dataset_name, run_name):
    dataset_root = os.path.join(config.get_dataset_spec(dataset_name)["output_dir"], run_name, dataset_name)
    os.makedirs(dataset_root, exist_ok=True)
    return dataset_root


def format_window_label(window_label):
    return "root_only" if window_label == config.ROOT_ONLY_WINDOW else f"{window_label}min"


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_log(log_path, lines):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def save_prediction_csv(path, prediction_bundle, preprocessor):
    target_names = list(config.TARGET_NAMES)
    preds_raw = preprocessor.inverse_transform_targets(prediction_bundle["preds"])
    targets_raw = preprocessor.inverse_transform_targets(prediction_bundle["targets"])
    target_mask = np.asarray(prediction_bundle["target_mask"], dtype=bool)
    tree_ids = prediction_bundle["tree_ids"]
    horizon_tags = prediction_bundle["horizon_tags"]

    fieldnames = ["tree_id", "horizon"]
    for target_name in target_names:
        fieldnames.extend([f"{target_name}_gt", f"{target_name}_pred"])

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, (tree_id, horizon_tag) in enumerate(zip(tree_ids, horizon_tags)):
            row = {
                "tree_id": tree_id,
                "horizon": horizon_tag,
            }
            for target_idx, target_name in enumerate(target_names):
                gt_key = f"{target_name}_gt"
                pred_key = f"{target_name}_pred"
                if target_mask[row_idx, target_idx]:
                    row[gt_key] = float(targets_raw[row_idx, target_idx])
                    row[pred_key] = float(preds_raw[row_idx, target_idx])
                else:
                    row[gt_key] = ""
                    row[pred_key] = ""
            writer.writerow(row)


def metric_to_str(value):
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return "nan"
        return f"{float(value):.3f}"
    return str(value)


def build_log_lines(dataset_name, window_label, future_horizons, best_cfg, results):
    lines = []
    mode = "future_horizons" if future_horizons else "final_only"
    window_str = format_window_label(window_label)

    overall = results["overall"]
    for target_name, metrics in overall.items():
        lines.append(
            " | ".join(
                [
                    f"dataset={dataset_name}",
                    f"window={window_str}",
                    f"mode={mode}",
                    "horizon=overall",
                    f"config={best_cfg['name']}",
                    f"target={target_name}",
                    f"MSE={metric_to_str(metrics['MSE'])}",
                    f"Spearman={metric_to_str(metrics['Spearman'])}",
                    f"R2={metric_to_str(metrics['R2'])}",
                ]
            )
        )

    for horizon, horizon_results in results.get("by_horizon", {}).items():
        for target_name, metrics in horizon_results.items():
            lines.append(
                " | ".join(
                    [
                        f"dataset={dataset_name}",
                        f"window={window_str}",
                        f"mode={mode}",
                        f"horizon={horizon}",
                        f"config={best_cfg['name']}",
                        f"target={target_name}",
                        f"MSE={metric_to_str(metrics['MSE'])}",
                        f"Spearman={metric_to_str(metrics['Spearman'])}",
                        f"R2={metric_to_str(metrics['R2'])}",
                    ]
                )
            )
    return lines


def save_visualizations(dataset_output_dir, file_stub, history, prediction_bundle):
    plot_training_curves(
        history,
        os.path.join(dataset_output_dir, f"{file_stub}_training_curves.png"),
        file_stub,
    )

    preds = prediction_bundle["preds"]
    targets = prediction_bundle["targets"]
    overall_mask = prediction_bundle["target_mask"].all(axis=1)
    if overall_mask.any():
        plot_prediction_scatter(
            targets[overall_mask],
            preds[overall_mask],
            config.TARGET_NAMES,
            dataset_output_dir,
            file_stub,
        )
        plot_prediction_scatter_linear(
            targets[overall_mask],
            preds[overall_mask],
            config.TARGET_NAMES,
            dataset_output_dir,
            file_stub,
        )
        plot_residuals(
            targets[overall_mask],
            preds[overall_mask],
            config.TARGET_NAMES,
            dataset_output_dir,
            file_stub,
        )

    horizon_tags = prediction_bundle.get("horizon_tags", [])
    if not horizon_tags:
        return

    horizon_tags = np.asarray(horizon_tags)
    for horizon in np.unique(horizon_tags):
        horizon_mask = horizon_tags == horizon
        if not horizon_mask.any():
            continue

        horizon_targets = targets[horizon_mask]
        horizon_preds = preds[horizon_mask]
        if horizon == "final":
            target_names = config.TARGET_NAMES
        else:
            target_names = config.TARGET_NAMES[: config.ROOT_SCORE_INDEX]
            horizon_targets = horizon_targets[:, : config.ROOT_SCORE_INDEX]
            horizon_preds = horizon_preds[:, : config.ROOT_SCORE_INDEX]

        if len(target_names) == 0:
            continue

        horizon_stub = f"{file_stub}_{horizon}"
        plot_prediction_scatter(
            horizon_targets,
            horizon_preds,
            target_names,
            dataset_output_dir,
            horizon_stub,
        )
        plot_prediction_scatter_linear(
            horizon_targets,
            horizon_preds,
            target_names,
            dataset_output_dir,
            horizon_stub,
        )
        plot_residuals(
            horizon_targets,
            horizon_preds,
            target_names,
            dataset_output_dir,
            horizon_stub,
        )


def print_results(dataset_name, window_label, best_cfg, results):
    print(f"\n  Test Results | dataset={dataset_name} | window={format_window_label(window_label)} | cfg={best_cfg['name']}")
    print(f"  {'Target':<22} {'MSE':>8} {'R2':>8} {'Pearson':>10} {'Spearman':>10} {'MAE':>8} {'MAPE':>8} {'RMSE':>8}")
    print(f"  {'-' * 90}")
    for target_name, metrics in results["overall"].items():
        print(
            f"  {target_name:<22} {metric_to_str(metrics['MSE']):>8} {metric_to_str(metrics['R2']):>8} "
            f"{metric_to_str(metrics['Pearson']):>10} {metric_to_str(metrics['Spearman']):>10} "
            f"{metric_to_str(metrics['MAE']):>8} {metric_to_str(metrics['MAPE']):>8} {metric_to_str(metrics['RMSE']):>8}"
        )
    for horizon, horizon_results in results.get("by_horizon", {}).items():
        print(f"\n  Horizon={horizon}")
        for target_name, metrics in horizon_results.items():
            print(
                f"    {target_name:<20} MSE={metric_to_str(metrics['MSE'])} "
                f"R2={metric_to_str(metrics['R2'])} Spearman={metric_to_str(metrics['Spearman'])}"
            )


def run_experiment(dataset_name, window_label, args, run_name, combined_log_path):
    print(f"\n{'=' * 80}")
    print(
        f"CasSeqGCN | dataset={dataset_name} | window={format_window_label(window_label)} "
        f"| future_horizons={args.future_horizons}"
    )
    print(f"{'=' * 80}")

    data = load_dataset(dataset_name, window_label, future_horizons=args.future_horizons)
    preprocessor = Preprocessor().fit(data["meta"])
    train_ds, val_ds, test_ds = build_datasets(data, preprocessor, args.future_horizons)

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    print(f"  Test samples:  {len(test_ds)}")

    trial_scores = {}
    for model_cfg in config.CANDIDATE_CONFIGS:
        val_loss = run_trial(model_cfg, train_loader, val_loader, args)
        trial_scores[model_cfg["name"]] = val_loss
        print(f"  Trial {model_cfg['name']:<10} val_loss={val_loss:.4f}")

    best_name = min(trial_scores, key=trial_scores.get)
    best_cfg = next(cfg for cfg in config.CANDIDATE_CONFIGS if cfg["name"] == best_name)
    print(f"  Best config: {best_name} (val_loss={trial_scores[best_name]:.4f})")

    model, history = run_full(best_cfg, train_loader, val_loader, args)
    prediction_bundle = get_predictions(model, test_loader, config.DEVICE)
    results = evaluate_predictions(
        prediction_bundle,
        target_names=config.TARGET_NAMES,
        group_by_horizon=args.future_horizons,
    )
    rounded_results = round_result_metrics(results, decimals=3)
    print_results(dataset_name, window_label, best_cfg, rounded_results)

    dataset_output_dir = create_output_dirs(dataset_name, run_name)
    window_str = format_window_label(window_label)
    mode_str = "future_horizons" if args.future_horizons else "final_only"
    file_stub = f"{dataset_name}_{window_str}_{mode_str}"

    result_payload = {
        "dataset": dataset_name,
        "window": window_str,
        "future_horizons": args.future_horizons,
        "best_config": best_cfg,
        "trial_scores": round_result_metrics(trial_scores, decimals=3),
        "metrics": rounded_results,
    }

    save_json(os.path.join(dataset_output_dir, f"{file_stub}_results.json"), result_payload)
    save_json(os.path.join(dataset_output_dir, f"{file_stub}_history.json"), round_result_metrics(history, decimals=6))
    torch.save(model.state_dict(), os.path.join(dataset_output_dir, f"{file_stub}_model.pt"))
    save_prediction_csv(
        os.path.join(dataset_output_dir, f"{file_stub}_predictions.csv"),
        prediction_bundle,
        preprocessor,
    )
    save_visualizations(dataset_output_dir, file_stub, history, prediction_bundle)

    log_lines = build_log_lines(dataset_name, window_label, args.future_horizons, best_cfg, rounded_results)
    append_log(os.path.join(dataset_output_dir, "dataset_summary.log"), log_lines)
    append_log(combined_log_path, log_lines)

    return {
        "dataset": dataset_name,
        "window": window_str,
        "future_horizons": args.future_horizons,
        "best_config": best_cfg["name"],
        "metrics": rounded_results,
    }


def main():
    args = merge_args_with_run_config(parse_args())
    set_random_seed(config.RANDOM_SEED)

    datasets = resolve_datasets(args.datasets)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_root = args.output_root or os.path.join(config.RESULTS_ROOT, run_name)
    os.makedirs(combined_root, exist_ok=True)
    combined_log_path = os.path.join(combined_root, "combined_summary.log")
    run_log_path = os.path.join(combined_root, "run.log")
    append_log(run_log_path, [
        f"{datetime.now().isoformat()}\tSTART datasets={datasets} windows={args.windows} future_horizons={args.future_horizons}"
    ])

    all_results = []
    for dataset_name in datasets:
        for window_label in resolve_windows(dataset_name, args.windows, args.include_root_only):
            try:
                all_results.append(
                    run_experiment(dataset_name, window_label, args, run_name, combined_log_path)
                )
                append_log(run_log_path, [
                    f"{datetime.now().isoformat()}\tDONE dataset={dataset_name} window={format_window_label(window_label)}"
                ])
            except Exception as exc:
                import traceback

                append_log(run_log_path, [
                    f"{datetime.now().isoformat()}\tERROR dataset={dataset_name} window={format_window_label(window_label)} error={exc}",
                    traceback.format_exc(),
                ])
                print(f"\nERROR dataset={dataset_name} window={format_window_label(window_label)}: {exc}")

    save_json(os.path.join(combined_root, "combined_results.json"), all_results)
    append_log(run_log_path, [f"{datetime.now().isoformat()}\tCOMPLETE"])


if __name__ == "__main__":
    main()
