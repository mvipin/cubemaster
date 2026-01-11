"""Data augmentation pipelines using Albumentations."""

from typing import Dict, Any, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: Tuple[int, int] = (50, 50),
    config: Dict[str, Any] = None,
) -> A.Compose:
    """Get training augmentation pipeline.
    
    Args:
        image_size: Target image size (H, W)
        config: Augmentation config from YAML
        
    Returns:
        Albumentations Compose object
    """
    config = config or {}
    
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
    ]
    
    # Geometric transforms
    if config.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))
    
    if config.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.5))
    
    rotation_limit = config.get("rotation_limit", 15)
    if rotation_limit > 0:
        transforms.append(A.Rotate(limit=rotation_limit, p=0.5))
    
    # Color transforms
    brightness = config.get("brightness_limit", 0.2)
    contrast = config.get("contrast_limit", 0.2)
    if brightness > 0 or contrast > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=contrast,
                p=0.5,
            )
        )
    
    hue_shift = config.get("hue_shift_limit", 10)
    sat_shift = config.get("saturation_limit", 20)
    if hue_shift > 0 or sat_shift > 0:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=hue_shift,
                sat_shift_limit=sat_shift,
                val_shift_limit=20,
                p=0.5,
            )
        )
    
    # Noise
    if config.get("gaussian_noise", True):
        transforms.append(A.GaussNoise(std_range=(0.01, 0.05), p=0.3))
    
    # Normalize and convert to tensor
    if config.get("normalize", True):
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_val_transforms(
    image_size: Tuple[int, int] = (50, 50),
    config: Dict[str, Any] = None,
) -> A.Compose:
    """Get validation/test augmentation pipeline (no augmentation).
    
    Args:
        image_size: Target image size (H, W)
        config: Augmentation config from YAML
        
    Returns:
        Albumentations Compose object
    """
    config = config or {}
    
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
    ]
    
    if config.get("normalize", True):
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)

