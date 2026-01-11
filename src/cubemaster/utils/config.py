"""Configuration management with YAML inheritance support."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import copy

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_config(
    config_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load YAML config with inheritance support.
    
    Supports `_base_` key for inheriting from another config file.
    Child config values override parent config values.
    
    Args:
        config_path: Path to YAML config file
        base_dir: Base directory for resolving relative paths.
                  If None, uses config file's parent directory.
                  
    Returns:
        Merged configuration dictionary
        
    Example:
        # child.yaml
        _base_: "base.yaml"
        model:
          name: "shallow_cnn"
          
        # Loads base.yaml first, then merges child.yaml on top
    """
    config_path = Path(config_path)
    
    if base_dir is None:
        base_dir = config_path.parent
    else:
        base_dir = Path(base_dir)
    
    # Load the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    # Check for base config
    if "_base_" in config:
        base_config_name = config.pop("_base_")
        base_config_path = base_dir / base_config_name
        
        # Recursively load base config (supports multi-level inheritance)
        base_config = load_config(base_config_path, base_dir)
        
        # Merge: base values + child overrides
        config = deep_merge(base_config, config)
    
    return config


class Config:
    """Configuration object with attribute-style access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
        self._config_dict = config_dict
    
    def __repr__(self) -> str:
        return f"Config({self._config_dict})"
    
    def __getitem__(self, key: str) -> Any:
        return self._config_dict[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        return self._config_dict.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        return copy.deepcopy(self._config_dict)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load config from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Config object
        """
        config_dict = load_config(config_path)
        return cls(config_dict)


def get_device(device_config: str = "auto") -> str:
    """Determine the device to use for training.
    
    Args:
        device_config: Device config string ("auto", "cuda", "cpu")
        
    Returns:
        Device string ("cuda" or "cpu")
    """
    import torch
    
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

