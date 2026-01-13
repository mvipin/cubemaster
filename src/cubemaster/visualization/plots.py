"""Plotting utilities for training visualization and evaluation metrics.

This module provides functions for generating training curves and confusion matrices.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np


def plot_training_curves(
    history: Dict[str, list],
    output_path: Path,
    title: str = "Training Curves",
) -> None:
    """Generate and save training curves plot.

    Args:
        history: Training history dictionary with keys:
            - train_loss: List of training losses per epoch
            - val_loss: List of validation losses per epoch
            - train_acc: List of training accuracies per epoch
            - val_acc: List of validation accuracies per epoch
        output_path: Path to save the plot
        title: Plot title
    """
    import matplotlib.pyplot as plt

    if not history or not history.get('train_loss'):
        print("  Warning: No training history available, skipping training curves plot")
        return

    epochs = range(len(history['train_loss']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, len(epochs) - 0.5)

    # Plot Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc', marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc', marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, len(epochs) - 0.5)

    # Add best validation accuracy annotation
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc)
    ax2.annotate(
        f'Best: {best_val_acc:.2f}%',
        xy=(best_epoch, best_val_acc),
        xytext=(best_epoch + 0.5, best_val_acc - 5),
        fontsize=10,
        fontweight='bold',
        color='green',
        arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
    )

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix as numpy array
        class_names: List of class names for axis labels
        output_path: Path to save the plot
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))

    # Normalize confusion matrix for color intensity
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Create heatmap with annotations showing counts
    ax = sns.heatmap(
        cm_normalized,
        annot=cm,  # Show raw counts
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Proportion'},
        annot_kws={'size': 14, 'weight': 'bold'},
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to: {output_path}")

