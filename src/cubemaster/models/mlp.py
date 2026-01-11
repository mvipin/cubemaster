"""MLP (Multi-Layer Perceptron) classifier for color detection."""

from typing import Tuple

import torch
import torch.nn as nn

from .base import BaseColorClassifier


class MLPClassifier(BaseColorClassifier):
    """Simple MLP baseline for Rubik's Cube color classification.

    Architecture:
        - Flatten input (40x40x3 = 4800)
        - FC: 4800 -> 256 -> ReLU -> Dropout(0.3)
        - FC: 256 -> 128 -> ReLU -> Dropout(0.3)
        - FC: 128 -> 6 (output)

    Expected Parameters: ~1.3M (mostly in first layer)
    Target Accuracy: 85-92%
    """

    def __init__(
        self,
        num_classes: int = 6,
        input_size: Tuple[int, int] = (40, 40),
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout_rate: float = 0.3,
    ):
        """Initialize MLP classifier.
        
        Args:
            num_classes: Number of output classes
            input_size: Input image size (H, W)
            hidden_dims: Tuple of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__(num_classes=num_classes, input_size=input_size)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Calculate flattened input size (3 channels)
        input_dim = input_size[0] * input_size[1] * 3
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Flatten: (batch, 3, H, W) -> (batch, 3*H*W)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

