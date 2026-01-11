"""Evaluation metrics for color classification models."""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute overall accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy as percentage (0-100)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return 100.0 * np.mean(y_true == y_pred)


def compute_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    num_classes: int = 6,
) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        Row i, Column j = count of samples with true label i predicted as j
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_per_class_metrics(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1 score.
    
    Args:
        confusion_matrix: Confusion matrix (num_classes x num_classes)
        class_names: Optional list of class names
        
    Returns:
        Dictionary with per-class metrics
    """
    num_classes = confusion_matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    metrics = {}
    
    for i, name in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[name] = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "support": int(confusion_matrix[i, :].sum()),
        }
    
    return metrics


def compute_macro_metrics(per_class_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-averaged metrics.
    
    Args:
        per_class_metrics: Per-class metrics dictionary
        
    Returns:
        Dictionary with macro precision, recall, F1
    """
    precisions = [m["precision"] for m in per_class_metrics.values()]
    recalls = [m["recall"] for m in per_class_metrics.values()]
    f1s = [m["f1"] for m in per_class_metrics.values()]
    
    return {
        "macro_precision": np.mean(precisions),
        "macro_recall": np.mean(recalls),
        "macro_f1": np.mean(f1s),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    class_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Comprehensive model evaluation.
    
    Args:
        model: Trained PyTorch model
        dataloader: Test/validation data loader
        device: Device to use
        class_names: Optional list of class names
        
    Returns:
        Dictionary containing all metrics
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu()
        
        all_preds.append(preds)
        all_labels.append(labels)
    
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    
    # Compute metrics
    accuracy = compute_accuracy(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names) if class_names else 6)
    per_class = compute_per_class_metrics(cm, class_names)
    macro = compute_macro_metrics(per_class)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class": per_class,
        **macro,
        "predictions": y_pred,
        "labels": y_true,
    }

