"""
Main entry point for DeepCas experiments.
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data_loader import load_dataset
from dataset import DeepCasDataset, collate_fn
from evaluate import build_prediction_table, evaluate_predictions, get_predictions
from model import DeepCasModel
from preprocessing import Preprocessor
from sampler import RandomWalkSampler
from train import train_model
from utils import plot_predictions, plot_training_curves


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _current_experiment_label():
    return "future_horizons" if config.USE_FUTURE_HORIZONS else "final_only"


def _embedding_split_window(window_minutes):
    if window_minutes != 0:
        return window_minutes
    return config.ROOT_ONLY_REFERENCE_WINDOW


def _load_embedding_artifacts(window_minutes):
    split_window = _embedding_split_window(window_minutes)
    paths = config.get_embedding_paths_for_window(split_window)
    if not os.path.exists(paths["vocab"]) or not os.path.exists(paths["embeddings"]):
        raise FileNotFoundError(
            f"Missing embedding artifacts for {config.CURRENT_DATASET}, "
            f"split window {split_window}min: "
            f"{paths['vocab']} and/or {paths['embeddings']}. "
            "Run build_interaction_graph.py for that dataset/window first."
        )

    with open(paths["vocab"], "r") as f:
        user_to_idx = json.load(f)
    pretrained_vectors = np.load(paths["embeddings"])
    return user_to_idx, pretrained_vectors, paths


def _create_optimizer(model):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.Adam(
        trainable_params,
        lr=config.LEARNING_RATE,
        weight_decay=5e-4,
    )


def _final_horizon_arrays(prediction_bundle):
    horizon_tags = np.array(prediction_bundle["horizon_tags"], dtype=object)
    final_mask = horizon_tags == "final"
    if not final_mask.any():
        return None, None
    return (
        prediction_bundle["targets"][final_mask],
        prediction_bundle["predictions"][final_mask],
    )


def _write_log_file(path, rows):
    lines = [
        "dataset\twindow\thorizon\texperiment\ttarget\tMSE\tSpearman\tR2",
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["dataset"]),
                    str(row["window"]),
                    str(row["horizon"]),
                    str(row["experiment"]),
                    str(row["target"]),
                    str(row["MSE"]),
                    str(row["Spearman"]),
                    str(row["R2"]),
                ]
            )
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _append_run_log(path, message):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\t{message}\n")


def _metrics_log_rows(dataset_name, window_minutes, metrics_by_horizon):
    rows = []
    experiment = _current_experiment_label()
    window_label = "0min" if window_minutes == 0 else f"{window_minutes}min"

    for horizon_tag, target_metrics in metrics_by_horizon.items():
        for target_name, metrics in target_metrics.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "window": window_label,
                    "horizon": horizon_tag,
                    "experiment": experiment,
                    "target": target_name,
                    "MSE": metrics.get("MSE"),
                    "Spearman": metrics.get("Spearman"),
                    "R2": metrics.get("R2"),
                }
            )
    return rows


def run_experiment(window_minutes, run_output_dir):
    """Run DeepCas for one dataset/window."""
    print(f"\n{'=' * 80}")
    label = "0MIN (ROOT-ONLY)" if window_minutes == 0 else f"{window_minutes}MIN"
    print(f"DEEPCAS EXPERIMENT: {config.CURRENT_DATASET.upper()} - {label}")
    print(f"{'=' * 80}")

    data = load_dataset(window_minutes)

    preprocessor = Preprocessor()
    preprocessor.set_available_targets(data["available_targets"])
    train_meta = data["meta"][data["meta"]["tree_id"].isin(data["train_ids"])]
    preprocessor.fit(train_meta)

    print("\nSampling random walk paths...")
    sampler = RandomWalkSampler()
    train_paths = sampler.precompute_all_paths(data["train_ids"], data["edges"], data["posts"], "Train")
    val_paths = sampler.precompute_all_paths(data["val_ids"], data["edges"], data["posts"], "Val")
    test_paths = sampler.precompute_all_paths(data["test_ids"], data["edges"], data["posts"], "Test")

    print("\nLoading pre-trained Interaction Graph data...")
    user_to_idx, pretrained_vectors, embedding_paths = _load_embedding_artifacts(window_minutes)
    num_users = len(user_to_idx)

    print(f"  Loaded vocabulary size: {num_users:,}")
    print(f"  Loaded embedding shape: {pretrained_vectors.shape}")
    print(f"  Embeddings: {embedding_paths['embeddings']}")

    print("\nCreating datasets...")
    train_dataset = DeepCasDataset(
        data["train_ids"],
        train_paths,
        data["meta"],
        preprocessor,
        user_to_idx,
        data["sample_specs"],
    )
    val_dataset = DeepCasDataset(
        data["val_ids"],
        val_paths,
        data["meta"],
        preprocessor,
        user_to_idx,
        data["sample_specs"],
    )
    test_dataset = DeepCasDataset(
        data["test_ids"],
        test_paths,
        data["meta"],
        preprocessor,
        user_to_idx,
        data["sample_specs"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=3,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    sample = next(iter(train_loader))
    global_feat_dim = sample["global_features"].shape[1]
    output_dim = sample["targets"].shape[1]

    print("\nModel configuration:")
    print(f"  User vocabulary size: {num_users:,}")
    print(f"  Global features: {global_feat_dim}")
    print(f"  Output targets: {output_dim}")
    print(f"  K={config.K}, T={config.T}")

    model = DeepCasModel(
        vocab_size=num_users,
        global_feat_dim=global_feat_dim,
        output_dim=output_dim,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        K=config.K,
        T=config.T,
        dropout=config.DROPOUT,
        final_mlp_hidden=config.FINAL_MLP_HIDDEN,
        pretrained_weights=pretrained_vectors,
    ).to(config.DEVICE)

    print(f"  Model parameters: {model.get_num_parameters():,}")

    print("\n--- Training ---")
    optimizer = _create_optimizer(model)
    model, history = train_model(model, train_loader, val_loader, optimizer, config.DEVICE)

    print("\n--- Test Evaluation ---")
    prediction_bundle = get_predictions(model, test_loader, config.DEVICE)
    test_results = evaluate_predictions(prediction_bundle)

    print("\nTest Results:")
    for horizon_tag, target_metrics in test_results.items():
        print(f"\n[{horizon_tag}]")
        for target, metrics in target_metrics.items():
            print(
                f"  {target}: "
                f"MSE={metrics.get('MSE')} R2={metrics.get('R2')} "
                f"Spearman={metrics.get('Spearman')} Pearson={metrics.get('Pearson')}"
            )

    window_output_dir = os.path.join(run_output_dir, f"{window_minutes}min")
    os.makedirs(window_output_dir, exist_ok=True)

    plot_training_curves(
        history,
        os.path.join(window_output_dir, "training_curve.png"),
        f"DeepCas - {config.CURRENT_DATASET} - {window_minutes}min",
    )

    final_true, final_pred = _final_horizon_arrays(prediction_bundle)
    if final_true is not None:
        plot_predictions(
            final_true,
            final_pred,
            config.OUTPUT_TARGETS,
            os.path.join(window_output_dir, "predictions_final.png"),
            f"DeepCas - {config.CURRENT_DATASET} - {window_minutes}min - final",
        )

    torch.save(model.state_dict(), os.path.join(window_output_dir, "model.pt"))

    prediction_table_log = build_prediction_table(prediction_bundle, space="log")
    prediction_table_log.to_csv(
        os.path.join(window_output_dir, "predictions_log.csv"), index=False
    )
    prediction_table_raw = build_prediction_table(prediction_bundle, space="raw")
    prediction_table_raw.to_csv(
        os.path.join(window_output_dir, "predictions_raw.csv"), index=False
    )

    results = {
        "dataset": config.CURRENT_DATASET,
        "window_minutes": window_minutes,
        "experiment": _current_experiment_label(),
        "hyperparameters": {
            "K": config.K,
            "T": config.T,
            "p_jump": config.P_JUMP,
            "embedding_dim": config.EMBEDDING_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "final_mlp_hidden": config.FINAL_MLP_HIDDEN,
            "learning_rate": config.LEARNING_RATE,
            "dropout": config.DROPOUT,
            "weight_decay": 5e-4,
        },
        "test_metrics": test_results,
        "model_parameters": model.get_num_parameters(),
        "training_epochs": len(history["train_loss"]),
        "num_users": num_users,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_test_samples": len(test_dataset),
        "future_horizons_enabled": config.USE_FUTURE_HORIZONS,
    }

    with open(os.path.join(window_output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    return results, _metrics_log_rows(config.CURRENT_DATASET, window_minutes, test_results)


def main():
    """Run experiments for configured datasets and windows."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(config.RESULTS_ROOT, f"run_{timestamp}")
    os.makedirs(run_root, exist_ok=True)
    run_log_path = os.path.join(run_root, "run.log")
    _append_run_log(run_log_path, f"Starting DeepCas datasets={config.DATASETS_TO_RUN} future_horizons={config.USE_FUTURE_HORIZONS}")

    print(f"Starting DeepCas experiments - {timestamp}")
    print(f"Datasets: {config.DATASETS_TO_RUN}")
    print(f"Device: {config.DEVICE}")
    print(f"Future horizons: {config.USE_FUTURE_HORIZONS}")

    all_results = {}
    combined_log_rows = []

    for dataset_name in config.DATASETS_TO_RUN:
        config.set_dataset(dataset_name)
        dataset_output_dir = os.path.join(run_root, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        _append_run_log(run_log_path, f"Dataset started: {dataset_name} windows={config.WINDOWS}")

        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"Windows: {config.WINDOWS}")
        print(f"Output directory: {dataset_output_dir}")
        print(f"{'=' * 80}")

        dataset_results = {}
        dataset_log_rows = []

        for window in config.WINDOWS:
            try:
                results, log_rows = run_experiment(window, dataset_output_dir)
                dataset_results[f"{window}min"] = results
                dataset_log_rows.extend(log_rows)
                combined_log_rows.extend(log_rows)
                _append_run_log(run_log_path, f"Completed dataset={dataset_name} window={window}min")
            except Exception as e:
                print(f"\nERROR in dataset={dataset_name}, window={window}min: {e}")
                _append_run_log(run_log_path, f"ERROR dataset={dataset_name} window={window}min error={e}")
                import traceback

                traceback.print_exc()
                _append_run_log(run_log_path, traceback.format_exc())

        all_results[dataset_name] = dataset_results

        with open(os.path.join(dataset_output_dir, "summary_results.json"), "w") as f:
            json.dump(dataset_results, f, indent=2, default=_json_default)
        _write_log_file(os.path.join(dataset_output_dir, "experiment.log"), dataset_log_rows)

    with open(os.path.join(run_root, "summary_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    _write_log_file(os.path.join(run_root, "combined_experiment.log"), combined_log_rows)
    _append_run_log(run_log_path, "All DeepCas experiments complete")

    print(f"\n{'=' * 80}")
    print(f"All experiments complete! Results saved to: {run_root}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
