"""
Training loop for CasSeqGCN with masked MSE supervision.
"""

import numpy as np
import torch

import config


def masked_mse_loss(preds, targets, target_mask):
    """Mean squared error over valid masked positions only."""
    squared_error = (preds - targets) ** 2
    valid_mask = target_mask.bool()
    if valid_mask.any():
        return squared_error[valid_mask].mean()
    return squared_error.mean()


def _forward(model, batch, device):
    return model(
        batch["snapshots"].to(device),
        batch["L_sn"].to(device),
        batch["node_mask"].to(device),
        batch["snap_mask"].to(device),
        batch["horizon_onehot"].to(device),
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        targets = batch["targets"].to(device)
        target_mask = batch["target_mask"].to(device)

        optimizer.zero_grad()
        preds = _forward(model, batch, device)
        loss = masked_mse_loss(preds, targets, target_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_n += len(targets)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        targets = batch["targets"].to(device)
        target_mask = batch["target_mask"].to(device)
        preds = _forward(model, batch, device)
        loss = masked_mse_loss(preds, targets, target_mask)
        total_loss += loss.item() * len(targets)
        total_n += len(targets)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    all_horizons = []
    all_tree_ids = []

    for batch in loader:
        preds = _forward(model, batch, device)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch["targets"].numpy())
        all_masks.append(batch["target_mask"].numpy())
        all_horizons.extend(batch["horizon_tags"])
        all_tree_ids.extend(batch["tree_ids"])

    return {
        "preds": np.vstack(all_preds) if all_preds else np.zeros((0, len(config.TARGET_NAMES))),
        "targets": np.vstack(all_targets) if all_targets else np.zeros((0, len(config.TARGET_NAMES))),
        "target_mask": np.vstack(all_masks) if all_masks else np.zeros((0, len(config.TARGET_NAMES)), dtype=bool),
        "horizon_tags": all_horizons,
        "tree_ids": all_tree_ids,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    max_epochs=None,
    patience=None,
    scheduler=None,
):
    max_epochs = max_epochs or config.MAX_EPOCHS
    patience = patience or config.PATIENCE

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{max_epochs} | "
                f"Train: {train_loss:.4f}  Val: {val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (best val: {best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history
