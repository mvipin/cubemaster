#!/usr/bin/env python3
"""Wandb sweep launcher for CubeMaster hyperparameter optimization.

Usage:
    # Create a new sweep and run an agent:
    python scripts/run_sweep.py --sweep-config configs/sweeps/mlp_sweep.yaml
    
    # Join an existing sweep:
    python scripts/run_sweep.py --sweep-id <sweep_id>
    
    # Run multiple agents in parallel:
    python scripts/run_sweep.py --sweep-config configs/sweeps/mlp_sweep.yaml --count 5
"""

import argparse
import sys
import os
from pathlib import Path

import yaml

try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Install with: pip install wandb")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch wandb hyperparameter sweeps for CubeMaster"
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        help="Path to sweep configuration YAML file",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Existing sweep ID to join (format: entity/project/sweep_id or just sweep_id)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="cubemaster",
        help="Wandb project name (default: cubemaster)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Wandb entity/team name",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs to execute (default: unlimited until sweep completes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sweep config without creating/running sweep",
    )
    return parser.parse_args()


def load_sweep_config(config_path: str) -> dict:
    """Load sweep configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_sweep(config: dict, project: str, entity: str = None) -> str:
    """Create a new wandb sweep.
    
    Args:
        config: Sweep configuration dictionary
        project: Wandb project name
        entity: Wandb entity name
        
    Returns:
        Sweep ID
    """
    sweep_id = wandb.sweep(
        sweep=config,
        project=project,
        entity=entity,
    )
    return sweep_id


def run_agent(sweep_id: str, project: str = None, entity: str = None, count: int = None):
    """Run a wandb sweep agent.
    
    Args:
        sweep_id: Sweep ID or full path (entity/project/sweep_id)
        project: Wandb project name (optional if sweep_id is full path)
        entity: Wandb entity name (optional if sweep_id is full path)
        count: Maximum number of runs to execute
    """
    # Build function that will be called for each run
    def train_fn():
        """Training function for sweep agent."""
        # Get the base config from sweep config
        base_config = wandb.config.get("config", "configs/mlp.yaml")
        
        # Build command to run training script
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config", base_config,
            "--wandb",
            "--sweep",
        ]
        
        # Execute training
        os.system(" ".join(cmd))
    
    # Construct sweep path
    if "/" in sweep_id:
        sweep_path = sweep_id
    else:
        if entity:
            sweep_path = f"{entity}/{project}/{sweep_id}"
        else:
            sweep_path = f"{project}/{sweep_id}"
    
    print(f"Starting sweep agent for: {sweep_path}")
    print(f"Max runs: {count if count else 'unlimited'}")
    
    wandb.agent(
        sweep_id=sweep_path,
        function=train_fn,
        count=count,
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if not args.sweep_config and not args.sweep_id:
        print("Error: Must provide either --sweep-config or --sweep-id")
        sys.exit(1)
    
    if args.sweep_config and args.sweep_id:
        print("Error: Cannot provide both --sweep-config and --sweep-id")
        sys.exit(1)
    
    # Load sweep config if provided
    if args.sweep_config:
        config = load_sweep_config(args.sweep_config)
        print(f"Loaded sweep config from: {args.sweep_config}")
        
        if args.dry_run:
            print("\n--- Sweep Configuration ---")
            print(yaml.dump(config, default_flow_style=False))
            return
        
        # Create new sweep
        sweep_id = create_sweep(config, args.project, args.entity)
        print(f"Created sweep: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"Joining existing sweep: {sweep_id}")
    
    # Run agent
    run_agent(sweep_id, args.project, args.entity, args.count)


if __name__ == "__main__":
    main()

