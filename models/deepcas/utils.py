"""
Utility functions.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, save_path, title):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train MSE', linewidth=2)
    plt.plot(history['val_loss'], label='Val MSE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE (log-space)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(y_true, y_pred, target_names, save_path, title):
    """Plot predictions vs true values."""
    n_targets = len(target_names)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        
        # Convert from log-space
        y_t_orig = np.expm1(y_true[:, i])
        y_p_orig = np.expm1(y_pred[:, i])
        
        ax.scatter(y_t_orig, y_p_orig, alpha=0.3, s=10)
        
        # Diagonal line
        min_val = max(1, min(y_t_orig.min(), y_p_orig.min()))
        max_val = max(y_t_orig.max(), y_p_orig.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('True Value (log scale)')
        ax.set_ylabel('Predicted Value (log scale)')
        ax.set_title(target)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()