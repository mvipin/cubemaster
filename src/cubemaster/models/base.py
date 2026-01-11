"""Base class for all color classification models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn


class BaseColorClassifier(nn.Module, ABC):
    """Abstract base class for Rubik's Cube color classifiers.
    
    All model architectures must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, num_classes: int = 6, input_size: Tuple[int, int] = (50, 50)):
        """Initialize the base classifier.
        
        Args:
            num_classes: Number of color classes (default: 6 for BGORWY)
            input_size: Expected input image size (height, width)
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self._model_name = self.__class__.__name__
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability distribution over classes
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.
        
        Returns:
            Dictionary with total and trainable parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "parameters": self.count_parameters(),
        }

