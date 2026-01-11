"""Dataset classes for Rubik's Cube color classification."""

from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from cubemaster import COLOR_CLASSES


class CubeColorDataset(Dataset):
    """PyTorch Dataset for Rubik's Cube color patches.
    
    Expects directory structure:
        root/
            B/  (Blue images)
            G/  (Green images)
            O/  (Orange images)
            R/  (Red images)
            W/  (White images)
            Y/  (Yellow images)
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: List[str] = COLOR_CLASSES,
    ):
        """Initialize the dataset.
        
        Args:
            root_dir: Root directory containing class subdirectories
            transform: Optional transform to apply to images
            class_names: List of class names (subdirectory names)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Collect all image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
    
    def _load_samples(self) -> None:
        """Load all image paths and their labels."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def get_class_counts(self) -> dict:
        """Get count of samples per class."""
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts

