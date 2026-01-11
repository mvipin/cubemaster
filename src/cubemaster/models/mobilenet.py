"""MobileNetV3 transfer learning classifier for color detection."""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

from .base import BaseColorClassifier


class MobileNetV3Classifier(BaseColorClassifier):
    """MobileNetV3-Small with transfer learning for color classification.
    
    Architecture:
        - MobileNetV3-Small backbone (pretrained on ImageNet)
        - Replace classifier head with custom head for 6 classes
        - Optional: freeze backbone for initial training
    
    Expected Parameters: ~1.5M (mostly frozen initially)
    Target Accuracy: 98-99.5%
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        input_size: Tuple[int, int] = (40, 40),
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.2,
    ):
        """Initialize MobileNetV3 classifier.

        Args:
            num_classes: Number of output classes
            input_size: Input image size (H, W) - resized to 224x224 in dataloader
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze backbone weights initially
            dropout_rate: Dropout probability for classifier head
        """
        super().__init__(num_classes=num_classes, input_size=input_size)
        
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v3_small(weights=weights)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Get the number of features from the original classifier
        in_features = self.backbone.classifier[0].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters except classifier."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self, layers: Optional[int] = None) -> None:
        """Unfreeze backbone for fine-tuning.
        
        Args:
            layers: Number of layers to unfreeze from the end.
                   If None, unfreeze all layers.
        """
        if layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Get all named parameters as a list
            params = list(self.backbone.named_parameters())
            # Unfreeze last N layers
            for name, param in params[-layers:]:
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
               Note: Input will be resized to 224x224 internally by the model
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.backbone(x)

