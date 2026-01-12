"""Training infrastructure for color classification models."""

from pathlib import Path
from typing import Dict, Optional, Any
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


class Trainer:
    """Training loop manager for color classification models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        scheduler: Optional[_LRScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 10,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            early_stopping: Early stopping instance
            checkpoint_dir: Directory to save checkpoints
            log_interval: Log every N batches
            use_wandb: Whether to log to Weights & Biases
            wandb_config: Wandb configuration dict (log_model, etc.)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_interval = log_interval
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_config = wandb_config or {}

        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        self.current_epoch = 0

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Watch model with wandb for gradient logging
        if self.use_wandb and wandb.run is not None:
            wandb.watch(self.model, log="gradients", log_freq=100)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        return {"loss": total_loss / total, "acc": 100. * correct / total}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return {"loss": total_loss / total, "acc": 100. * correct / total}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_best: bool = True,
    ) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_best: Whether to save best model checkpoint

        Returns:
            Training history dictionary
        """
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["lr"].append(current_lr)

            # Log to wandb
            if self.use_wandb and wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "learning_rate": current_lr,
                    "best_val_acc": self.best_val_acc,
                })

            # Print epoch summary
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.2f}% | "
                f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
            )

            # Save best model
            if save_best and val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                self.save_checkpoint("best.pt")
                print(f"  → New best model saved (val_acc: {self.best_val_acc:.2f}%)")

                # Log best model to wandb if configured
                if self.use_wandb and self.wandb_config.get("log_model") and wandb.run is not None:
                    wandb.save(str(self.checkpoint_dir / "best.pt"))

            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_metrics["loss"]):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Save final model
        self.save_checkpoint("last.pt")
        self.save_history()

        # Log final summary to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.run.summary["best_val_acc"] = self.best_val_acc
            wandb.run.summary["final_train_loss"] = self.history["train_loss"][-1]
            wandb.run.summary["final_val_loss"] = self.history["val_loss"][-1]
            wandb.run.summary["total_epochs"] = self.current_epoch + 1

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filepath: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.history = checkpoint.get("history", self.history)
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def save_history(self) -> None:
        """Save training history to JSON."""
        if not self.checkpoint_dir:
            return
        with open(self.checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
