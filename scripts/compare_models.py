#!/usr/bin/env python3
"""Model comparison script for CubeMaster.

Compares all trained models and generates comparison reports.

Usage:
    python scripts/compare_models.py --output results/comparison/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison",
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "shallow_cnn", "mobilenet"],
        help="Models to compare",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Comparing models: {args.models}")
    print(f"Output directory: {args.output}")
    
    # TODO: Implement comparison
    # 1. Load all trained models
    # 2. Evaluate on test set
    # 3. Generate comparison table
    # 4. Plot comparison charts
    # 5. Recommend best model
    
    print("\n[TODO] Comparison pipeline not yet implemented.")


if __name__ == "__main__":
    main()

