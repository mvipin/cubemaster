#!/usr/bin/env python3
"""Evaluation script for CubeMaster color classification models.

This script evaluates trained models on the test dataset and generates
comprehensive metrics, confusion matrix visualizations, and training curves.

Usage:
    python scripts/evaluate_model.py --model shallow_cnn --checkpoint models/shallow_cnn/best.pt
    python scripts/evaluate_model.py --model mlp --checkpoint models/mlp/best.pt
    python scripts/evaluate_model.py --model mobilenet --checkpoint models/mobilenet/best.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cubemaster import COLOR_CLASSES
from cubemaster.utils.config import load_config, get_device
from cubemaster.models.shallow_cnn import ShallowCNNClassifier
from cubemaster.models.mlp import MLPClassifier
from cubemaster.models.mobilenet import MobileNetV3Classifier
from cubemaster.training.dataset import CubeColorDataset
from cubemaster.training.augmentations import get_val_transforms
from cubemaster.evaluation.metrics import evaluate_model, compute_per_class_metrics


MODEL_REGISTRY = {
    "shallow_cnn": ShallowCNNClassifier,
    "mlp": MLPClassifier,
    "mobilenet": MobileNetV3Classifier,
}

DEFAULT_CONFIGS = {
    "shallow_cnn": "configs/shallow_cnn.yaml",
    "mlp": "configs/mlp.yaml",
    "mobilenet": "configs/mobilenet.yaml",
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate CubeMaster color classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_model.py --model shallow_cnn
  python scripts/evaluate_model.py --model shallow_cnn --checkpoint models/shallow_cnn/best.pt
  python scripts/evaluate_model.py --model mlp --checkpoint models/mlp/last.pt --no-plots
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: models/{model}/best.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: configs/{model}.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/{model})",
    )
    return parser.parse_args()


def build_model(model_name: str, cfg: dict) -> nn.Module:
    """Build model from config."""
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {model_name}")

    kwargs = {
        "num_classes": cfg["data"].get("num_classes", 6),
        "input_size": tuple(cfg["data"].get("image_size", [40, 40])),
    }

    model_cfg = cfg.get("model", {})
    if "dropout_rate" in model_cfg:
        kwargs["dropout_rate"] = model_cfg["dropout_rate"]
    if "hidden_dims" in model_cfg:
        kwargs["hidden_dims"] = tuple(model_cfg["hidden_dims"])
    if "pretrained" in model_cfg:
        kwargs["pretrained"] = model_cfg["pretrained"]
    if "freeze_backbone" in model_cfg:
        kwargs["freeze_backbone"] = model_cfg["freeze_backbone"]

    return model_cls(**kwargs)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: str) -> Dict[str, Any]:
    """Load model checkpoint and return checkpoint data."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def print_evaluation_results(results: Dict[str, Any], class_names: list) -> None:
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Overall Metrics':^60}")
    print("-" * 60)
    print(f"  Accuracy:        {results['accuracy']:>8.2f}%")
    print(f"  Macro Precision: {results['macro_precision']:>8.2f}%")
    print(f"  Macro Recall:    {results['macro_recall']:>8.2f}%")
    print(f"  Macro F1:        {results['macro_f1']:>8.2f}%")
    
    print(f"\n{'Per-Class Metrics':^60}")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Support':>10}")
    print("-" * 60)
    for cls_name in class_names:
        m = results['per_class'][cls_name]
        print(f"{cls_name:<10} {m['precision']:>11.2f}% {m['recall']:>11.2f}% {m['f1']:>11.2f}% {m['support']:>10}")

    print(f"\n{'Confusion Matrix':^60}")
    print("-" * 60)
    cm = results['confusion_matrix']
    header = "Predicted:  " + "  ".join(f"{c:>5}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"Actual {class_names[i]}:  " + "  ".join(f"{v:>5}" for v in row)
        print(row_str)
    print("=" * 60)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save confusion matrix heatmap."""
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


def plot_training_curves(
    history: Dict[str, list],
    output_path: Path,
    title: str = "Training Curves",
) -> None:
    """Generate and save training curves plot."""
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


def save_evaluation_json(
    results: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    output_path: Path,
    model_name: str,
) -> None:
    """Save evaluation metrics to JSON file."""
    # Prepare JSON-serializable results
    json_results = {
        "model_name": model_name,
        "checkpoint_epoch": checkpoint_info.get("epoch", "unknown"),
        "best_val_acc": checkpoint_info.get("best_val_acc", None),
        "test_metrics": {
            "accuracy": float(results["accuracy"]),
            "macro_precision": float(results["macro_precision"]),
            "macro_recall": float(results["macro_recall"]),
            "macro_f1": float(results["macro_f1"]),
        },
        "per_class_metrics": {
            cls: {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                  for k, v in metrics.items()}
            for cls, metrics in results["per_class"].items()
        },
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "total_samples": int(len(results["labels"])),
        "correct_predictions": int(np.sum(results["predictions"] == results["labels"])),
    }

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Evaluation metrics saved to: {output_path}")


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Set up paths
    config_path = Path(args.config) if args.config else Path(DEFAULT_CONFIGS[args.model])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(f"models/{args.model}/best.pt")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/{args.model}")

    # Validate paths
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print(f"  Available checkpoints in models/{args.model}/:")
        model_dir = Path(f"models/{args.model}")
        if model_dir.exists():
            for f in model_dir.glob("*.pt"):
                print(f"    - {f.name}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"CubeMaster Model Evaluation")
    print(f"{'=' * 60}")
    print(f"  Model:      {args.model}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config:     {config_path}")
    print(f"  Output:     {output_dir}")

    # Setup device
    device = get_device(args.device)
    print(f"  Device:     {device}")

    # Load config
    cfg = load_config(config_path)

    # Build and load model
    print(f"\nLoading model...")
    model = build_model(args.model, cfg)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")

    # Load test dataset
    print(f"\nLoading test dataset...")
    data_cfg = cfg["data"]
    image_size = tuple(data_cfg.get("image_size", [40, 40]))
    test_dir = Path(data_cfg["root_dir"]) / data_cfg["test_dir"]

    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)

    transform = get_val_transforms(image_size, cfg.get("augmentation", {}).get("val"))
    test_dataset = CubeColorDataset(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"].get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
    )
    print(f"  Test samples: {len(test_dataset)}")

    # Evaluate model
    print(f"\nEvaluating model...")
    results = evaluate_model(model, test_loader, device=device, class_names=COLOR_CLASSES)

    # Print results
    print_evaluation_results(results, COLOR_CLASSES)

    # Generate visualizations
    if not args.no_plots:
        print(f"\nGenerating visualizations...")

        # Confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            results["confusion_matrix"],
            COLOR_CLASSES,
            cm_path,
            title=f"{args.model.upper()} - Confusion Matrix (Test Set)",
        )

        # Training curves (if history available in checkpoint)
        history = checkpoint.get("history", {})
        if history and history.get("train_loss"):
            curves_path = output_dir / "training_curves.png"
            plot_training_curves(
                history,
                curves_path,
                title=f"{args.model.upper()} - Training Curves",
            )
        else:
            print("  Warning: No training history in checkpoint, skipping training curves")

    # Save evaluation JSON
    print(f"\nSaving evaluation results...")
    json_path = output_dir / "test_evaluation.json"
    save_evaluation_json(results, checkpoint, json_path, args.model)

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

