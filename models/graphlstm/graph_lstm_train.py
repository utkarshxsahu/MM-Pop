"""
Training script for Graph-LSTM.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from data_loader import load_dataset
from evaluate import evaluate_grouped_predictions, evaluate_predictions
from graph_lstm_dataset import GraphLSTMDataset, collate_fn
from graph_lstm_model import GraphLSTMModel
from precompute_features import precompute_vocab_and_tokens, precompute_window
from visualization import plot_prediction_scatter, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train Graph-LSTM experiments.")
    parser.add_argument("--datasets", nargs="+", help="Dataset names or 'all'")
    parser.add_argument("--windows", nargs="+", help="Window tags such as 20 50 root_only")
    parser.add_argument("--include-root-only", action="store_true", help="Include root_only in addition to requested numeric windows")
    parser.add_argument("--mode", choices=[config.MODE_STANDARD, config.MODE_FUTURE_HORIZON], help="Experiment mode")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument("--embedding-mode", choices=["fine_tune", "freeze"], help="Embedding training mode")
    parser.add_argument("--embedding-lr", type=float, help="Embedding learning rate when fine-tuning")
    parser.add_argument("--lr", type=float, help="Main optimizer learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--max-epochs", type=int, help="Maximum epochs")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--pretrained-embeddings", help="Path to pretrained embedding matrix (.npy)")
    parser.add_argument("--pretrained-vocab", help="Path to pretrained embedding vocab map (.json)")
    return parser.parse_args()


def apply_runtime_overrides(args):
    if args.datasets:
        config.DATASETS = args.datasets if args.datasets != ["all"] else "all"
    if args.windows:
        parsed_windows = []
        for token in args.windows:
            parsed_windows.append(config.WINDOW_ROOT_ONLY if token == config.WINDOW_ROOT_ONLY else int(token))
        config.WINDOWS = parsed_windows
    if args.include_root_only:
        config.INCLUDE_ROOT_ONLY = True
    if args.mode:
        config.MODE = args.mode
    if args.optimizer:
        config.OPTIMIZER_NAME = args.optimizer
    if args.embedding_mode:
        config.EMBEDDING_TRAINING_MODE = args.embedding_mode
    if args.embedding_lr is not None:
        config.EMBEDDING_LR = args.embedding_lr
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.max_epochs is not None:
        config.MAX_EPOCHS = args.max_epochs
    if args.patience is not None:
        config.PATIENCE = args.patience
    if args.pretrained_embeddings:
        config.PRETRAINED_EMBEDDINGS_PATH = args.pretrained_embeddings
    if args.pretrained_vocab:
        config.PRETRAINED_EMBEDDINGS_VOCAB_PATH = args.pretrained_vocab


def format_metric(value):
    return "nan" if value is None else f"{value:.3f}"


class RunLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, line=""):
        with open(self.path, "a") as handle:
            handle.write(f"{line}\n")
        print(line)


def masked_mse_loss(preds, targets, loss_mask):
    squared_error = (preds - targets) ** 2
    mask = loss_mask.float()
    valid = mask.sum()
    if valid.item() <= 0:
        return squared_error.mean()
    return (squared_error * mask).sum() / valid


def build_optimizer(model):
    optimizer_name = config.OPTIMIZER_NAME.lower()
    train_embedding = config.EMBEDDING_TRAINING_MODE == "fine_tune"
    model.embedding.weight.requires_grad = train_embedding

    param_groups = []
    if train_embedding:
        param_groups.append({"params": [model.embedding.weight], "lr": config.EMBEDDING_LR})

    other_params = [param for name, param in model.named_parameters() if name != "embedding.weight" and param.requires_grad]
    param_groups.append({"params": other_params, "lr": config.LEARNING_RATE})

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    if optimizer_name == "adam":
        return torch.optim.Adam(param_groups, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER_NAME}")


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    tree_ids, horizon_tags = [], []

    with torch.no_grad():
        for batch in loader:
            if not batch:
                continue
            preds = model(batch)
            targets = torch.stack([tree["targets"] for tree in batch]).to(device)
            masks = torch.stack([tree["loss_mask"] for tree in batch]).to(device)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            tree_ids.extend([tree["tree_id"] for tree in batch])
            horizon_tags.extend([tree["horizon_tag"] for tree in batch])

    return {
        "preds": np.vstack(all_preds),
        "targets": np.vstack(all_targets),
        "masks": np.vstack(all_masks).astype(bool),
        "tree_ids": tree_ids,
        "horizon_tags": horizon_tags,
    }


def save_predictions_csv(out_dir, target_names, prediction_bundle):
    y_true = prediction_bundle["targets"]
    y_pred = prediction_bundle["preds"]
    mask = prediction_bundle["masks"]

    rows = {
        "tree_id": prediction_bundle["tree_ids"],
        "horizon_tag": prediction_bundle["horizon_tags"],
    }
    for idx, target_name in enumerate(target_names):
        gt_log_values = y_true[:, idx].copy()
        pred_log_values = y_pred[:, idx].copy()
        gt_raw_values = np.expm1(gt_log_values)
        pred_raw_values = np.expm1(pred_log_values)

        if target_name == "root_score":
            gt_log_values = np.where(mask[:, idx], gt_log_values, np.nan)
            pred_log_values = np.where(mask[:, idx], pred_log_values, np.nan)
            gt_raw_values = np.where(mask[:, idx], gt_raw_values, np.nan)
            pred_raw_values = np.where(mask[:, idx], pred_raw_values, np.nan)

        rows[f"{target_name}_gt_log1p"] = gt_log_values
        rows[f"{target_name}_pred_log1p"] = pred_log_values
        rows[f"{target_name}_gt_raw"] = gt_raw_values
        rows[f"{target_name}_pred_raw"] = pred_raw_values

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)


def log_metrics(logger, dataset_name, window_label, metrics, grouped_metrics=None):
    for target_name, target_metrics in metrics.items():
        logger.log(
            f"{dataset_name}\t{window_label}\tfinal\t{target_name}\t"
            f"MSE={format_metric(target_metrics['MSE'])}\t"
            f"Spearman={format_metric(target_metrics['Spearman'])}\t"
            f"R2={format_metric(target_metrics['R2'])}"
        )
    if grouped_metrics:
        for horizon_tag, horizon_metrics in grouped_metrics.items():
            for target_name, target_metrics in horizon_metrics.items():
                logger.log(
                    f"{dataset_name}\t{window_label}\t{horizon_tag}\t{target_name}\t"
                    f"MSE={format_metric(target_metrics['MSE'])}\t"
                    f"Spearman={format_metric(target_metrics['Spearman'])}\t"
                    f"R2={format_metric(target_metrics['R2'])}"
                )


def train_window(dataset_name, window_tag, data, dataset_logger, combined_logger):
    window_label = config.get_window_label(window_tag)
    out_dir = os.path.join(config.get_output_root(dataset_name), window_label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Graph-LSTM | {dataset_name.upper()} | {window_label} | {config.MODE.upper()}")
    print(f"{'=' * 60}")

    train_set = GraphLSTMDataset(data["train_samples"], window_tag, dataset_name)
    val_set = GraphLSTMDataset(data["val_samples"], window_tag, dataset_name)
    test_set = GraphLSTMDataset(data["test_samples"], window_tag, dataset_name)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    device = config.DEVICE
    model = GraphLSTMModel(
        struct_dim=train_set.struct_dim,
        vocab_size=train_set.vocab_size,
        embed_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        n_targets=train_set.n_targets,
        dropout=config.DROPOUT,
        mlp_hidden=config.FINAL_MLP_HIDDEN,
        use_horizon_conditioning=config.is_future_horizon_mode(),
    ).to(device)

    vocab_path = os.path.join(config.get_output_root(dataset_name, config.MODE_STANDARD), "graph_lstm_cache", "vocab.json")
    if config.PRETRAINED_EMBEDDINGS_PATH:
        loaded = model.load_pretrained_embeddings(
            vocab_path,
            config.PRETRAINED_EMBEDDINGS_PATH,
            config.PRETRAINED_EMBEDDINGS_VOCAB_PATH,
        )
        print(f"Loaded pretrained embeddings: {loaded}")

    optimizer = build_optimizer(model)
    model_path = os.path.join(out_dir, "best_model.pt")
    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            if not batch:
                continue
            preds = model(batch)
            targets = torch.stack([tree["targets"] for tree in batch]).to(device)
            loss_mask = torch.stack([tree["loss_mask"] for tree in batch]).to(device)
            loss = masked_mse_loss(preds, targets, loss_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if not batch:
                    continue
                preds = model(batch)
                targets = torch.stack([tree["targets"] for tree in batch]).to(device)
                loss_mask = torch.stack([tree["loss_mask"] for tree in batch]).to(device)
                val_losses.append(masked_mse_loss(preds, targets, loss_mask).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:3d}/{config.MAX_EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    prediction_bundle = get_predictions(model, test_loader, device)
    overall_metrics = evaluate_predictions(
        prediction_bundle["targets"],
        prediction_bundle["preds"],
        data["target_names"],
        prediction_bundle["masks"],
    )

    final_keep = np.asarray(prediction_bundle["horizon_tags"]) == "final"
    final_metrics = evaluate_predictions(
        prediction_bundle["targets"][final_keep],
        prediction_bundle["preds"][final_keep],
        data["target_names"],
        prediction_bundle["masks"][final_keep],
    )

    grouped_metrics = None
    if config.is_future_horizon_mode():
        grouped_metrics = evaluate_grouped_predictions(
            prediction_bundle["targets"],
            prediction_bundle["preds"],
            data["target_names"],
            prediction_bundle["horizon_tags"],
            prediction_bundle["masks"],
        )

    metrics_payload = {
        "dataset": dataset_name,
        "window": window_label,
        "mode": config.MODE,
        "metric_space": "log1p",
        "metrics_final": final_metrics,
        "metrics_all_samples": overall_metrics,
        "metrics_by_horizon": grouped_metrics,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as handle:
        json.dump(metrics_payload, handle, indent=2)

    save_predictions_csv(out_dir, data["target_names"], prediction_bundle)
    plot_training_curves(history, os.path.join(out_dir, "training_curves.png"), f"GraphLSTM_{dataset_name}_{window_label}")

    if np.any(final_keep):
        plot_prediction_scatter(
            prediction_bundle["targets"][final_keep],
            prediction_bundle["preds"][final_keep],
            data["target_names"],
            out_dir,
            f"GraphLSTM_{dataset_name}_{window_label}_final",
        )

    log_metrics(dataset_logger, dataset_name, window_label, final_metrics, grouped_metrics)
    log_metrics(combined_logger, dataset_name, window_label, final_metrics, grouped_metrics)

    return metrics_payload


def run_dataset(dataset_name, combined_logger):
    config.set_active_dataset(dataset_name)
    output_root = config.get_output_root(dataset_name)
    os.makedirs(output_root, exist_ok=True)
    dataset_logger = RunLogger(os.path.join(output_root, "dataset_results.log"))
    dataset_logger.log(f"Run started: {datetime.now().isoformat()}")
    dataset_logger.log(f"Mode: {config.MODE}")

    windows = config.get_requested_windows(dataset_name)
    from data_loader import load_full_data

    full_metadata, full_posts, _ = load_full_data(dataset_name)
    dataset_logger.log("Precomputing caches...")
    for window_tag in windows:
        precompute_window(window_tag, full_posts, full_metadata, dataset_name)
    precompute_vocab_and_tokens(full_posts, windows, dataset_name)

    all_results = {}
    for window_tag in windows:
        data = load_dataset(window_tag, dataset_name, config.MODE)
        all_results[config.get_window_label(window_tag)] = train_window(dataset_name, window_tag, data, dataset_logger, combined_logger)

    with open(os.path.join(output_root, "all_results.json"), "w") as handle:
        json.dump(all_results, handle, indent=2)
    return all_results


def main():
    args = parse_args()
    apply_runtime_overrides(args)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    top_level_root = os.path.join(config.RESULTS_ROOT, f"run_{config.MODE}_{run_stamp}")
    config.RUN_OUTPUT_ROOT = top_level_root
    os.makedirs(top_level_root, exist_ok=True)
    combined_logger = RunLogger(os.path.join(top_level_root, "combined_results.log"))
    combined_logger.log(f"Run started: {datetime.now().isoformat()}")
    combined_logger.log(f"Datasets: {config.get_requested_datasets()}")
    combined_logger.log(f"Mode: {config.MODE}")

    combined_results = {}
    for dataset_name in config.get_requested_datasets():
        try:
            combined_results[dataset_name] = run_dataset(dataset_name, combined_logger)
            combined_logger.log(f"Completed dataset: {dataset_name}")
        except Exception as exc:
            import traceback

            combined_logger.log(f"ERROR dataset={dataset_name}: {exc}")
            combined_logger.log(traceback.format_exc())

    with open(os.path.join(top_level_root, "combined_results.json"), "w") as handle:
        json.dump(combined_results, handle, indent=2)
    print(f"\nAll results saved to {top_level_root}")


if __name__ == "__main__":
    main()
