#!/usr/bin/env python3
"""Training script for CubeMaster color classification models.

Usage:
    python scripts/train.py --config configs/shallow_cnn.yaml
    python scripts/train.py --config configs/mlp.yaml
    python scripts/train.py --config configs/mobilenet.yaml

    # With wandb logging:
    python scripts/train.py --config configs/shallow_cnn.yaml --wandb

    # For sweep agent (called automatically by wandb):
    python scripts/train.py --config configs/shallow_cnn.yaml --wandb --sweep
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cubemaster.utils.config import load_config, get_device, set_seed
from cubemaster.models.shallow_cnn import ShallowCNNClassifier
from cubemaster.models.mlp import MLPClassifier
from cubemaster.models.mobilenet import MobileNetV3Classifier
from cubemaster.training.dataset import CubeColorDataset
from cubemaster.training.augmentations import get_train_transforms, get_val_transforms
from cubemaster.training.trainer import Trainer, EarlyStopping
from cubemaster.visualization import plot_training_curves

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
    # Wandb arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Running as wandb sweep agent (uses wandb.config for hyperparams)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name (overrides config)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity/team name (overrides config)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name (overrides config)",
    )
    return parser.parse_args()


def apply_sweep_config(cfg: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply wandb sweep parameters to the config.

    Args:
        cfg: Base configuration dictionary
        sweep_config: Wandb sweep config (from wandb.config)

    Returns:
        Updated configuration dictionary
    """
    # Learning rate
    if "lr" in sweep_config:
        cfg["optimizer"]["lr"] = sweep_config["lr"]

    # Batch size
    if "batch_size" in sweep_config:
        cfg["training"]["batch_size"] = sweep_config["batch_size"]

    # Dropout rate
    if "dropout_rate" in sweep_config:
        cfg["model"]["dropout_rate"] = sweep_config["dropout_rate"]

    # Hidden dimensions (MLP)
    if "hidden_dims" in sweep_config:
        cfg["model"]["hidden_dims"] = sweep_config["hidden_dims"]

    # Weight decay
    if "weight_decay" in sweep_config:
        cfg["optimizer"]["weight_decay"] = sweep_config["weight_decay"]

    # Optimizer
    if "optimizer" in sweep_config:
        cfg["optimizer"]["name"] = sweep_config["optimizer"]

    # Label smoothing
    if "label_smoothing" in sweep_config:
        cfg["loss"]["label_smoothing"] = sweep_config["label_smoothing"]

    # Freeze backbone (MobileNet)
    if "freeze_backbone" in sweep_config:
        cfg["model"]["freeze_backbone"] = sweep_config["freeze_backbone"]

    # Scheduler
    if "scheduler" in sweep_config:
        cfg["scheduler"]["name"] = sweep_config["scheduler"]

    # Augmentation settings
    if "rotation_limit" in sweep_config:
        if "augmentation" not in cfg:
            cfg["augmentation"] = {"train": {}}
        cfg["augmentation"]["train"]["rotation_limit"] = sweep_config["rotation_limit"]

    if "brightness_limit" in sweep_config:
        if "augmentation" not in cfg:
            cfg["augmentation"] = {"train": {}}
        cfg["augmentation"]["train"]["brightness_limit"] = sweep_config["brightness_limit"]

    return cfg


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


def init_wandb(args, cfg: Dict[str, Any]) -> bool:
    """Initialize wandb if enabled.

    Args:
        args: Command line arguments
        cfg: Configuration dictionary

    Returns:
        True if wandb is initialized, False otherwise
    """
    # Check if wandb should be enabled
    use_wandb = args.wandb or cfg.get("wandb", {}).get("enabled", False)

    if not use_wandb:
        return False

    if not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Install with: pip install wandb")
        return False

    # Get wandb config from args or config file
    wandb_cfg = cfg.get("wandb", {})
    project = args.wandb_project or wandb_cfg.get("project", "cubemaster")
    entity = args.wandb_entity or wandb_cfg.get("entity")
    name = args.wandb_name or wandb_cfg.get("name")
    tags = wandb_cfg.get("tags", [])
    notes = wandb_cfg.get("notes")

    # Add model name to tags
    model_name = cfg.get("model", {}).get("name", "unknown")
    if model_name not in tags:
        tags = tags + [model_name]

    # Initialize wandb
    wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        notes=notes,
        config=cfg,  # Log full config
        settings=wandb.Settings(reinit="finish_previous"),  # Allow reinit for sweeps
    )

    print(f"Wandb initialized: {wandb.run.url}")
    return True


def main():
    """Main training entry point."""
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Initialize wandb (before applying sweep config)
    use_wandb = init_wandb(args, cfg)

    # Apply sweep config if running as sweep agent
    if args.sweep and use_wandb and WANDB_AVAILABLE:
        sweep_config = dict(wandb.config)
        cfg = apply_sweep_config(cfg, sweep_config)
        print(f"Applied sweep config: {sweep_config}")

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

    # Wandb config for trainer
    wandb_cfg = cfg.get("wandb", {})

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
        use_wandb=use_wandb,
        wandb_config=wandb_cfg,
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

    # Generate training curves
    model_name = cfg["model"]["name"]
    results_dir = Path(f"results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    curves_path = results_dir / "training_curves.png"
    print(f"\nGenerating training curves...")
    plot_training_curves(
        history,
        curves_path,
        title=f"{model_name.upper()} - Training Curves",
    )

    # Finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()

