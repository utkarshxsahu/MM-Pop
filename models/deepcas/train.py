"""
Training loop for DeepCas.
"""

import torch

import config


def masked_mse_loss(predictions, targets, target_mask):
    """Mean squared error over masked target positions."""
    squared_error = (predictions - targets) ** 2
    mask = target_mask.bool()

    if mask.any():
        return squared_error[mask].mean()
    return squared_error.mean()


def train_epoch(model, data_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        paths = batch["paths"].to(device)
        global_features = batch["global_features"].to(device)
        targets = batch["targets"].to(device)
        target_mask = batch["target_mask"].to(device)
        tree_sizes = batch["tree_sizes"].to(device)

        optimizer.zero_grad()
        predictions = model(paths, global_features, tree_sizes)
        loss = masked_mse_loss(predictions, targets, target_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_samples += len(targets)

    return total_loss / max(total_samples, 1)


def evaluate(model, data_loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            paths = batch["paths"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"].to(device)
            target_mask = batch["target_mask"].to(device)
            tree_sizes = batch["tree_sizes"].to(device)

            predictions = model(paths, global_features, tree_sizes)
            loss = masked_mse_loss(predictions, targets, target_mask)

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

    return total_loss / max(total_samples, 1)


def train_model(model, train_loader, val_loader, optimizer, device, max_epochs=None, patience=None):
    """
    Full training loop with early stopping.
    """
    max_epochs = max_epochs or config.MAX_EPOCHS
    patience = patience or config.PATIENCE

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state)
    return model, history
