"""Shallow CNN classifier for color detection."""

from typing import Tuple

import torch
import torch.nn as nn

from .base import BaseColorClassifier


class ShallowCNNClassifier(BaseColorClassifier):
    """Shallow CNN for Rubik's Cube color classification.
    
    Architecture (mirrors existing Keras model):
        - Conv2D: 3 -> 32, 3x3, ReLU
        - MaxPool: 2x2
        - Conv2D: 32 -> 64, 3x3, ReLU  
        - MaxPool: 2x2
        - Conv2D: 64 -> 64, 3x3, ReLU
        - Flatten
        - FC: -> 64, ReLU, Dropout(0.5)
        - FC: -> 6 (output)
    
    Expected Parameters: ~50K-100K
    Target Accuracy: 95-98%
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        input_size: Tuple[int, int] = (40, 40),
        dropout_rate: float = 0.5,
    ):
        """Initialize Shallow CNN classifier.
        
        Args:
            num_classes: Number of output classes
            input_size: Input image size (H, W)
            dropout_rate: Dropout probability for FC layer
        """
        super().__init__(num_classes=num_classes, input_size=input_size)
        
        self.dropout_rate = dropout_rate
        
        # Feature extractor (convolutional layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Calculate flattened size after convolutions
        # Input: 40x40 -> after pool1: 20x20 -> after pool2: 10x10
        # With 64 channels: 64 * 10 * 10 = 6400
        self._feat_size = self._calculate_feat_size(input_size)
        
        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feat_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )
    
    def _calculate_feat_size(self, input_size: Tuple[int, int]) -> int:
        """Calculate feature map size after convolutions."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size[0], input_size[1])
            out = self.features(dummy)
            return out.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

