from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def create_run_output_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_dataset_output_dir(run_dir: Path, dataset_name: str) -> Path:
    output_dir = run_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_logging(output_dir: Path, log_filename: str = "training.log") -> logging.Logger:
    log_path = output_dir / log_filename
    logger = logging.getLogger(f"deephawkes.{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def plot_training_curves(train_losses: List[float], val_losses: List[float], output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss (log1p space)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        plt.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.5)
        plt.plot(best_epoch, best_val_loss, "g*", markersize=15)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_checkpoint(model: torch.nn.Module, run_config: Dict, epoch: int, val_loss: float, output_path: Path) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": run_config,
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, output_path)


def _fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.3f}"


def format_metrics_for_logging(metrics: Dict, targets: List[str], metric_names: List[str]) -> str:
    lines = []
    ordered_keys = [target for target in targets if target in metrics]
    if "avg" in metrics:
        ordered_keys.append("avg")

    for key in ordered_keys:
        label = "avg" if key == "avg" else key
        parts = [f"{metric_name}={_fmt(metrics[key].get(metric_name, float('nan')))}" for metric_name in metric_names]
        lines.append(f"{label}: " + ", ".join(parts))
    return "\n".join(lines)


def append_summary_log(
    output_dir: Path,
    dataset_name: str,
    window_label: str,
    setting_name: str,
    metrics_by_scope: Dict[str, Dict],
    targets: List[str],
    log_filename: str = "run_summary.log",
) -> None:
    summary_path = output_dir / log_filename
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"dataset={dataset_name} window={window_label} setting={setting_name}\n")
        for scope, metrics in metrics_by_scope.items():
            f.write(f"[{scope}]\n")
            for target in targets:
                target_metrics = metrics.get(target, {})
                f.write(
                    f"{target}: "
                    f"MSE={_fmt(target_metrics.get('MSE', float('nan')))}, "
                    f"Spearman={_fmt(target_metrics.get('Spearman', float('nan')))}, "
                    f"R2={_fmt(target_metrics.get('R2', float('nan')))}\n"
                )
        f.write("\n")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
