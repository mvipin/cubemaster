#!/usr/bin/env python3
"""Dataset preparation script for CubeMaster.

This script:
1. Organizes images into color subdirectories
2. Creates train/val/test splits
3. Applies data augmentation to increase dataset size

Usage:
    python scripts/prepare_dataset.py --val-split 0.2 --augment-factor 10
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import albumentations as A


def parse_filename(filename: str) -> Tuple[str, str]:
    """Parse color label from filename.
    
    Format: {color}.{image_id}-{session}-{position}.jpg
    Returns: (color_label, base_name)
    """
    parts = filename.split('.')
    if len(parts) >= 2:
        return parts[0], filename
    return None, filename


def organize_into_subdirs(data_dir: Path) -> Dict[str, List[Path]]:
    """Move images from root directory into color subdirectories."""
    color_images = defaultdict(list)
    
    # Find all images in root
    for img_file in data_dir.glob("*.jpg"):
        color, _ = parse_filename(img_file.name)
        if color in ["B", "G", "O", "R", "W", "Y"]:
            color_images[color].append(img_file)
    
    # Move to subdirectories
    for color, images in color_images.items():
        color_dir = data_dir / color
        color_dir.mkdir(exist_ok=True)
        for img_path in images:
            dest = color_dir / img_path.name
            if img_path.exists() and not dest.exists():
                shutil.move(str(img_path), str(dest))
    
    # Return updated paths
    result = defaultdict(list)
    for color in ["B", "G", "O", "R", "W", "Y"]:
        color_dir = data_dir / color
        if color_dir.exists():
            result[color] = list(color_dir.glob("*.jpg"))
    
    return result


def create_val_split(
    train_dir: Path,
    val_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, int]:
    """Create validation split from training data."""
    random.seed(seed)
    moved_counts = {}
    
    for color in ["B", "G", "O", "R", "W", "Y"]:
        train_color_dir = train_dir / color
        val_color_dir = val_dir / color
        val_color_dir.mkdir(parents=True, exist_ok=True)
        
        if not train_color_dir.exists():
            continue
        
        images = list(train_color_dir.glob("*.jpg"))
        random.shuffle(images)
        
        n_val = int(len(images) * val_ratio)
        val_images = images[:n_val]
        
        for img_path in val_images:
            dest = val_color_dir / img_path.name
            shutil.move(str(img_path), str(dest))
        
        moved_counts[color] = n_val
    
    return moved_counts


def get_augmentation_pipeline(target_size: Tuple[int, int] = (50, 50)) -> A.Compose:
    """Get augmentation pipeline based on base.yaml settings."""
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),  # Updated API for albumentations 2.x
    ])


def augment_dataset(
    data_dir: Path,
    augment_factor: int = 10,
    target_size: Tuple[int, int] = (50, 50),
    seed: int = 42
) -> Dict[str, int]:
    """Apply augmentation to increase dataset size."""
    random.seed(seed)
    np.random.seed(seed)
    
    transform = get_augmentation_pipeline(target_size)
    aug_counts = {}
    
    for color in ["B", "G", "O", "R", "W", "Y"]:
        color_dir = data_dir / color
        if not color_dir.exists():
            continue
        
        # Get original images (not augmented ones)
        original_images = [f for f in color_dir.glob("*.jpg") if "_aug" not in f.name]
        aug_count = 0
        
        for img_path in original_images:
            image = np.array(Image.open(img_path).convert("RGB"))
            
            # Generate augmented versions
            for i in range(augment_factor - 1):  # -1 because original counts as 1
                augmented = transform(image=image)["image"]
                aug_name = f"{img_path.stem}_aug{i+1}.jpg"
                aug_path = color_dir / aug_name
                
                # Convert back to PIL and save
                Image.fromarray(augmented).save(aug_path, quality=95)
                aug_count += 1
        
        aug_counts[color] = aug_count
    
    return aug_counts


def count_images(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Count images per color class."""
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        counts[split] = {}
        for color in ["B", "G", "O", "R", "W", "Y"]:
            color_dir = split_dir / color
            if color_dir.exists():
                counts[split][color] = len(list(color_dir.glob("*.jpg")))
            else:
                counts[split][color] = 0
    return counts


def main():
    parser = argparse.ArgumentParser(description="Prepare CubeMaster dataset")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--augment-factor", type=int, default=10,
                        help="Augmentation factor (default: 10x)")
    parser.add_argument("--target-size", type=int, nargs=2, default=[50, 50],
                        help="Target image size (default: 50 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-organize", action="store_true",
                        help="Skip organizing into subdirectories")
    parser.add_argument("--skip-val-split", action="store_true",
                        help="Skip creating validation split")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Skip data augmentation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    print("=" * 60)
    print("CubeMaster Dataset Preparation")
    print("=" * 60)

    # Step 1: Organize into subdirectories
    if not args.skip_organize:
        print("\n[Step 1] Organizing images into color subdirectories...")
        for split_dir in [train_dir, test_dir]:
            if split_dir.exists():
                organized = organize_into_subdirs(split_dir)
                total = sum(len(v) for v in organized.values())
                print(f"  {split_dir.name}: {total} images organized")

    # Step 2: Create validation split
    if not args.skip_val_split:
        print(f"\n[Step 2] Creating validation split ({args.val_split*100:.0f}%)...")
        moved = create_val_split(train_dir, val_dir, args.val_split, args.seed)
        print(f"  Moved to val: {moved}")

    # Step 3: Apply augmentation
    if not args.skip_augment:
        print(f"\n[Step 3] Applying {args.augment_factor}x augmentation...")
        target_size = tuple(args.target_size)

        # Augment train only (not val/test)
        print("  Augmenting training set...")
        train_aug = augment_dataset(train_dir, args.augment_factor, target_size, args.seed)
        print(f"  Train augmented: {train_aug}")

    # Final counts
    print("\n" + "=" * 60)
    print("Final Dataset Summary")
    print("=" * 60)
    counts = count_images(data_dir)

    for split in ["train", "val", "test"]:
        total = sum(counts[split].values())
        print(f"\n{split.upper()}: {total} images")
        for color, count in counts[split].items():
            print(f"  {color}: {count}")

    grand_total = sum(sum(c.values()) for c in counts.values())
    print(f"\nGRAND TOTAL: {grand_total} images")


if __name__ == "__main__":
    main()

