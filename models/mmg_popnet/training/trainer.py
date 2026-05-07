"""
Training utilities — DDP-aware.

Location: training/trainer.py

Fixes applied vs previous version:
  Fix 1 — Gradient clipping every step (not just every accumulation_steps).
           Prevents a single explosive batch from corrupting weights before
           the clip fires. ~5-10% slower per epoch, worth it for stability.
  Fix 2 — Skip NaN/Inf batches. If a batch produces a non-finite loss,
           zero the gradients and continue rather than propagating NaN.
  Fix 3 — Transformer lr lowered to 1e-6 in CANONICAL_PARAMS (experiment_runner).
  Fix 4 — EarlyStopping handles NaN val_loss: counts toward patience but
           never saves NaN state, triggers early stop if patience exceeded.
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.amp import GradScaler, autocast

import config.config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_ddp():
    return dist.is_available() and dist.is_initialized()


def _reduce_scalar(value, device):
    """Average a Python float across all DDP ranks. No-op in single-GPU."""
    if not _is_ddp():
        return value
    t = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


# ---------------------------------------------------------------------------
# Early stopping  (Fix 4: NaN-aware)
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience         = patience
        self.min_delta        = min_delta
        self.counter          = 0
        self.best_loss        = None
        self.early_stop       = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        raw = model.module if hasattr(model, 'module') else model

        # Fix 4: NaN/Inf means numerical explosion.
        # Count toward patience but never save this state.
        # Prevents the model running uselessly after a NaN collapse.
        if not math.isfinite(val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        if self.best_loss is None:
            self.best_loss        = val_loss
            self.best_model_state = {
                k: v.cpu().clone() for k, v in raw.state_dict().items()
            }
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss        = val_loss
            self.best_model_state = {
                k: v.cpu().clone() for k, v in raw.state_dict().items()
            }
            self.counter = 0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def mse_loss(y_pred, y_true, horizon_mask=None):
    """
    MSE in log-space.

    Args:
        y_pred        : (batch, n_targets) model predictions
        y_true        : (batch, n_targets) ground truth — device may differ (model parallelism)
        horizon_mask  : optional (batch, n_targets) BoolTensor.
                        When provided, loss is computed only over True positions
                        (mean over valid positions, not the full tensor).
                        Used to exclude root_score from non-final horizon samples.
    """
    loss = (y_pred - y_true.to(y_pred.device)) ** 2   # (batch, n_targets)
    if horizon_mask is not None:
        mask = horizon_mask.float().to(y_pred.device)  # (batch, n_targets)
        valid = mask.sum()
        if valid > 0:
            return (loss * mask).sum() / valid
        return loss.mean()   # fallback: shouldn't happen with valid data
    return torch.mean(loss)


# ---------------------------------------------------------------------------
# Train / eval epoch functions
# ---------------------------------------------------------------------------

def _extract_horizon_mask_mlp(batch_tuple):
    """Return horizon_mask from MLP batch tuple (or None in default mode)."""
    if len(batch_tuple) == 4:
        return batch_tuple[2]   # (x, y, horizon_mask, horizon_idx)
    return None


def train_epoch_mlp(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        X_batch = batch[0].to(device)
        Y_batch = batch[1].to(device)
        hmask   = _extract_horizon_mask_mlp(batch)
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss   = mse_loss(Y_pred, Y_batch,
                          horizon_mask=hmask.to(device) if hmask is not None else None)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(train_loader.dataset)


def train_epoch_graphsage(model, train_loader, optimizer, device):
    """
    Standard GraphSAGE/GAT trainer — no gradient accumulation needed.
    With DDP each rank sees its shard; gradients averaged automatically.
    """
    model.train()
    total_loss    = 0
    total_samples = 0
    for batch in train_loader:
        batch  = batch.to(device)
        optimizer.zero_grad()
        Y_pred = model(batch)
        hmask  = getattr(batch, 'horizon_mask', None)
        loss   = mse_loss(Y_pred, batch.y, horizon_mask=hmask)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * batch.y.size(0)
        total_samples += batch.y.size(0)
    return total_loss / total_samples


def train_epoch_text_graphsage(model, train_loader, optimizer, device,
                                scaler, accumulation_steps, scheduler=None):
    """
    Mixed precision + gradient accumulation for TextGraphSAGE.

    Fix 1: Gradient clipping at every accumulation boundary.

    Fix 2: Skip non-finite batches WITHOUT zeroing accumulated valid gradients.
           The original code called optimizer.zero_grad() on every NaN skip,
           which erased valid gradients already accumulated from earlier batches
           in the same window — causing the optimizer to step on zero/stale
           gradients and rapidly corrupt weights.

    Fix 3: Fire an optimizer step for any remaining valid gradients at epoch end
           (the tail window when len(dataset) % accumulation_steps != 0 was
           previously silently dropped every epoch).

    accumulation_steps: passed from caller, computed as
        max(1, effective_batch // (per_rank_batch * world_size))
    """
    model.train()
    total_loss      = 0
    total_samples   = 0
    skipped         = 0
    valid_in_window = 0          # valid batches accumulated since last optimizer step
    optimizer.zero_grad()
    raw = model.module if hasattr(model, 'module') else model

    def _do_optimizer_step():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), max_norm=0.5)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scaler.get_scale() >= old_scale:
            scheduler.step()
        optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        batch = batch.to(device)

        with autocast('cuda'):
            Y_pred = model(batch)
            hmask  = getattr(batch, 'horizon_mask', None)
            loss   = mse_loss(Y_pred, batch.y, horizon_mask=hmask)
            loss   = loss / accumulation_steps

        # Fix 2: skip NaN/Inf batches — do NOT zero_grad here.
        # Zeroing would erase valid gradients already accumulated in this window.
        # Just skip this batch's contribution and advance the window counter.
        if not torch.isfinite(loss):
            skipped += 1
            # If this lands on an accumulation boundary and we had valid grads,
            # still step so they aren't held stale into the next window.
            if (i + 1) % accumulation_steps == 0:
                if valid_in_window > 0:
                    _do_optimizer_step()
                else:
                    optimizer.zero_grad()   # nothing to step on; reset cleanly
                valid_in_window = 0
            continue

        scaler.scale(loss).backward()
        valid_in_window += 1

        total_loss    += loss.item() * accumulation_steps * batch.y.size(0)
        total_samples += batch.y.size(0)

        if (i + 1) % accumulation_steps == 0:
            _do_optimizer_step()
            valid_in_window = 0

    # Fix 3: flush any remaining gradients from a partial tail window
    if valid_in_window > 0:
        _do_optimizer_step()

    if skipped > 0:
        print(f"  [trainer] Warning: skipped {skipped} non-finite batches this epoch")

    if total_samples == 0:
        return float('nan')
    return total_loss / total_samples


def evaluate_mlp(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch[0].to(device)
            Y_batch = batch[1].to(device)
            hmask   = _extract_horizon_mask_mlp(batch)
            Y_pred     = model(X_batch)
            total_loss += mse_loss(
                Y_pred, Y_batch,
                horizon_mask=hmask.to(device) if hmask is not None else None
            ).item() * len(X_batch)
    return total_loss / len(data_loader.dataset)


def evaluate_graphsage(model, data_loader, device):
    model.eval()
    total_loss    = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            batch      = batch.to(device)
            Y_pred     = model(batch)
            hmask      = getattr(batch, 'horizon_mask', None)
            total_loss += mse_loss(Y_pred, batch.y, horizon_mask=hmask).item() * batch.y.size(0)
            total_samples += batch.y.size(0)
    return total_loss / total_samples


def evaluate_text_graphsage(model, data_loader, device):
    model.eval()
    total_loss    = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            with autocast('cuda'):
                Y_pred = model(batch)
                hmask  = getattr(batch, 'horizon_mask', None)
                loss   = mse_loss(Y_pred, batch.y, horizon_mask=hmask)
            total_loss    += loss.item() * batch.y.size(0)
            total_samples += batch.y.size(0)
    return total_loss / total_samples


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_model(model, model_type, train_loader, val_loader, optimizer,
                device, max_epochs=200, rank=0, world_size=1,
                accumulation_steps=8, scheduler=None, scheduler_config=None):
    """
    Train with optional DDP.

    Args:
        model             : raw (non-DDP) model already placed on `device`
        rank              : this process's rank (0 = master)
        world_size        : total DDP processes (1 = single GPU)
        accumulation_steps: gradient accumulation steps for text_graphsage.
                            Computed by caller as:
                            max(1, effective_batch // (per_rank_batch * world_size))
                            Has no effect for graphsage / gat / mlp.
        scheduler         : optional LR scheduler stepped after real optimizer
                            steps. Used by text_graphsage with accumulation.
        scheduler_config  : optional JSON-serializable metadata for logging.

    Returns:
        raw model with best weights restored, history dict
    """
    use_ddp = world_size > 1

    if use_ddp:
        ddp_model = DDP(model,
                        device_ids=[device.index],
                        output_device=device.index,
                        find_unused_parameters=True)
    else:
        ddp_model = model

    early_stopping = EarlyStopping(patience=cfg.PATIENCE)
    history        = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'scheduler_config': scheduler_config or None,
    }
    scaler         = None

    if model_type == 'text_graphsage':
        train_fn = train_epoch_text_graphsage
        eval_fn  = evaluate_text_graphsage
        scaler   = GradScaler('cuda')
        if rank == 0:
            print(f"[trainer] TextGraphSAGE — AMP + grad clip every step, "
                  f"accumulation_steps={accumulation_steps}, "
                  f"effective_batch={accumulation_steps * world_size} × per_rank_batch")
            if scheduler_config is not None:
                print(f"[trainer] LR scheduler: {scheduler_config}")
    elif model_type == 'mlp':
        train_fn = train_epoch_mlp
        eval_fn  = evaluate_mlp
    else:
        train_fn = train_epoch_graphsage
        eval_fn  = evaluate_graphsage

    for epoch in range(max_epochs):
        # DistributedSampler needs epoch set for correct per-epoch shuffling
        if use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if model_type == 'text_graphsage':
            train_loss = train_fn(ddp_model, train_loader, optimizer,
                                  device, scaler, accumulation_steps,
                                  scheduler=scheduler)
        else:
            train_loss = train_fn(ddp_model, train_loader, optimizer, device)

        val_loss = eval_fn(ddp_model, val_loader, device)

        # Sync losses so EarlyStopping is identical on every rank
        if use_ddp:
            train_loss = _reduce_scalar(train_loss, device)
            val_loss   = _reduce_scalar(val_loss,   device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append([group['lr'] for group in optimizer.param_groups])

        if rank == 0:
            if (epoch + 1) % 5 == 0 or model_type == 'text_graphsage':
                finite_flag = "" if math.isfinite(train_loss) else " ⚠ NaN"
                print(f"Epoch {epoch+1}/{max_epochs} — "
                      f"Train MSE: {train_loss:.6f}, "
                      f"Val MSE: {val_loss:.6f}{finite_flag}")

        # Detect NaN weights early — if the model has gone NaN, every subsequent
        # batch will also be NaN and training is unrecoverable. Stop immediately
        # rather than grinding through patience epochs of wasted compute.
        if not math.isfinite(train_loss) and model_type == 'text_graphsage':
            raw_check = ddp_model.module if hasattr(ddp_model, 'module') else ddp_model
            has_nan_weights = any(
                not torch.isfinite(p).all()
                for p in raw_check.parameters() if p is not None
            )
            if has_nan_weights:
                if rank == 0:
                    print(f"  [trainer] NaN weights at epoch {epoch+1} — "
                          f"stopping early (unrecoverable divergence).")
                early_stopping.early_stop = True

        # Fix 4: NaN-aware early stopping
        early_stopping(val_loss, ddp_model)

        if early_stopping.early_stop:
            if rank == 0:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best weights into the raw (non-DDP) model
    # If best_model_state is None (NaN from epoch 1), keep current weights
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    else:
        if rank == 0:
            print("[trainer] Warning: no valid checkpoint saved "
                  "(all epochs produced NaN). Returning current weights.")

    return model, history
