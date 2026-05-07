"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves(history, output_path, model_name):
    """
    Plot training and validation loss curves
    
    Args:
        history: dict with 'train_loss' and 'val_loss' lists
        output_path: path to save plot
        model_name: name for title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train MSE', linewidth=2)
    plt.plot(history['val_loss'], label='Validation MSE', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE (in log-space)', fontsize=12)
    plt.title(f'Training Progress - {model_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_scatter(y_true, y_pred, target_names, output_dir, model_name):
    """
    Create scatter plots of predictions vs true values for each target
    Uses log-scale on both axes to handle outliers
    
    Args:
        y_true: true values (n_samples, n_targets) in log-space
        y_pred: predicted values (n_samples, n_targets) in log-space
        target_names: list of target names
        output_dir: directory to save plots
        model_name: name for title
    """
    n_targets = len(target_names)
    n_rows = (n_targets + 2) // 3  # Ceiling division
    n_cols = min(3, n_targets)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        
        # Convert back to original scale for plotting
        y_t_orig = np.expm1(y_true[:, i])
        y_p_orig = np.expm1(y_pred[:, i])
        
        # Plot on log-log scale
        ax.scatter(y_t_orig, y_p_orig, alpha=0.3, s=10)
        
        # Add diagonal line
        min_val = max(1, min(y_t_orig.min(), y_p_orig.min()))
        max_val = max(y_t_orig.max(), y_p_orig.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel('True Value (log scale)', fontsize=10)
        ax.set_ylabel('Predicted Value (log scale)', fontsize=10)
        ax.set_title(target, fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
    
    # Remove extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Predictions vs True Values - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_scatter_{model_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_scatter_linear(y_true, y_pred, target_names, output_dir, model_name):
    """
    Create scatter plots of predictions vs true values on LINEAR scale (actual values)
    
    Args:
        y_true: true values (n_samples, n_targets) in log-space
        y_pred: predicted values (n_samples, n_targets) in log-space
        target_names: list of target names
        output_dir: directory to save plots
        model_name: name for title
    """
    n_targets = len(target_names)
    n_rows = (n_targets + 2) // 3  # Ceiling division
    n_cols = min(3, n_targets)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        
        # Convert back to original scale for plotting
        y_t_orig = np.expm1(y_true[:, i])
        y_p_orig = np.expm1(y_pred[:, i])
        
        # Plot on LINEAR scale (not log)
        ax.scatter(y_t_orig, y_p_orig, alpha=0.3, s=10)
        
        # Add diagonal line
        min_val = min(y_t_orig.min(), y_p_orig.min())
        max_val = max(y_t_orig.max(), y_p_orig.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # LINEAR scale (not log)
        ax.set_xlabel('True Value (actual scale)', fontsize=10)
        ax.set_ylabel('Predicted Value (actual scale)', fontsize=10)
        ax.set_title(target, fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Predictions vs True Values (Linear Scale) - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_scatter_{model_name}_linear.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true, y_pred, target_names, output_dir, model_name):
    """
    Plot residuals for each target
    
    Args:
        y_true: true values in log-space
        y_pred: predicted values in log-space
        target_names: list of target names
        output_dir: directory to save plots
        model_name: name for title
    """
    n_targets = len(target_names)
    n_rows = (n_targets + 2) // 3
    n_cols = min(3, n_targets)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        
        residuals = y_pred[:, i] - y_true[:, i]
        
        ax.scatter(y_true[:, i], residuals, alpha=0.3, s=10)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel('True Value (log-space)', fontsize=10)
        ax.set_ylabel('Residual (log-space)', fontsize=10)
        ax.set_title(target, fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Residual Plots - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'residuals_{model_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()