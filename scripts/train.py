#!/usr/bin/env python3
"""Training script for CubeMaster color classification models.

Usage:
    python scripts/train.py --config configs/shallow_cnn.yaml
    python scripts/train.py --config configs/mlp.yaml
    python scripts/train.py --config configs/mobilenet.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cubemaster import COLOR_CLASSES
from cubemaster.utils.config import load_config, get_device, set_seed
from cubemaster.models.shallow_cnn import ShallowCNNClassifier
from cubemaster.models.mlp import MLPClassifier
from cubemaster.models.mobilenet import MobileNetV3Classifier
from cubemaster.training.dataset import CubeColorDataset
from cubemaster.training.augmentations import get_train_transforms, get_val_transforms
from cubemaster.training.trainer import Trainer, EarlyStopping
from cubemaster.evaluation.metrics import evaluate_model


MODEL_REGISTRY = {
    "shallow_cnn": ShallowCNNClassifier,
    "mlp": MLPClassifier,
    "mobilenet": MobileNetV3Classifier,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CubeMaster color classification model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto, cuda, cpu)",
    )
    return parser.parse_args()


def build_model(cfg: dict) -> nn.Module:
    """Build model from config."""
    model_name = cfg["model"]["name"]
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Build model kwargs from config
    kwargs = {
        "num_classes": cfg["data"].get("num_classes", 6),
        "input_size": tuple(cfg["data"].get("image_size", [40, 40])),
    }

    # Add model-specific args
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


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 0.001)
    weight_decay = opt_cfg.get("weight_decay", 0.0001)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    """Build learning rate scheduler from config."""
    sched_cfg = cfg.get("scheduler", {})
    name = sched_cfg.get("name", "cosine").lower()

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", 100),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 30),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def main():
    """Main training entry point."""
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    print(f"Training model: {cfg['model']['name']}")

    # Setup device and seed
    device = get_device(args.device or cfg.get("device", "auto"))
    print(f"Using device: {device}")

    set_seed(cfg.get("seed", 42), cfg.get("deterministic", True))

    # Build data loaders
    data_cfg = cfg["data"]
    image_size = tuple(data_cfg.get("image_size", [40, 40]))
    root_dir = Path(data_cfg["root_dir"])

    train_transform = get_train_transforms(image_size, cfg.get("augmentation", {}).get("train"))
    val_transform = get_val_transforms(image_size, cfg.get("augmentation", {}).get("val"))

    train_dataset = CubeColorDataset(root_dir / data_cfg["train_dir"], transform=train_transform)
    val_dataset = CubeColorDataset(root_dir / data_cfg["val_dir"], transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"].get("batch_size", 32),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"].get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Build model
    model = build_model(cfg)
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Loss function
    loss_cfg = cfg.get("loss", {})
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Early stopping
    patience = cfg["training"].get("early_stopping_patience", 15)
    early_stopping = EarlyStopping(patience=patience) if patience > 0 else None

    # Checkpoint directory
    output_cfg = cfg.get("output", {})
    checkpoint_dir = Path(output_cfg.get("model_dir", f"models/{cfg['model']['name']}"))

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint_dir=checkpoint_dir,
        log_interval=cfg.get("logging", {}).get("log_interval", 10),
    )

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Train
    epochs = cfg["training"].get("epochs", 100)
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 80)

    history = trainer.fit(train_loader, val_loader, epochs=epochs)

    print("=" * 80)
    print(f"Training complete. Best val accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

