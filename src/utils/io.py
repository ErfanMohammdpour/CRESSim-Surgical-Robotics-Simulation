"""
IO utilities for configuration and file handling.
"""

import yaml
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    logger.debug(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML or JSON file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    logger.debug(f"Saved config to {config_path}")


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    
    logger.debug(f"Loaded pickle from {file_path}")
    return obj


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """Save object to pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.debug(f"Saved pickle to {file_path}")


def load_paths_config() -> Dict[str, str]:
    """Load paths configuration."""
    try:
        return load_config("configs/paths.yaml")
    except FileNotFoundError:
        # Return default paths if config doesn't exist
        return {
            "data_dir": "data",
            "sim_dir": "sim",
            "unity_build": "sim/CRESSim/Builds/SuctionEnv.x86_64",
            "kvasir_seg_dir": "data/kvasir_seg",
            "demos_dir": "data/demos",
            "checkpoints_dir": "data/checkpoints",
            "logs_dir": "data/logs",
            "videos_dir": "data/videos",
            "benchmarks_dir": "data/benchmarks"
        }
