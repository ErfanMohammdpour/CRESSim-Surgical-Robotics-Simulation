"""
I/O utilities for configuration and file handling.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.debug(f"Loaded config from {config_path}")
        return config or {}
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.debug(f"Saved config to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving config {config_path}: {e}")
        raise


def load_json(json_path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    json_file = Path(json_path)
    
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON from {json_path}")
        return data
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON {json_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON {json_path}: {e}")
        raise


def save_json(data: Dict[str, Any], json_path: str) -> None:
    """Save data to JSON file."""
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved JSON to {json_path}")
    
    except Exception as e:
        logger.error(f"Error saving JSON {json_path}: {e}")
        raise


def ensure_dir(dir_path: str) -> Path:
    """Ensure directory exists, create if it doesn't."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def get_file_extension(file_path: str) -> str:
    """Get file extension."""
    return Path(file_path).suffix.lower()


def is_image_file(file_path: str) -> bool:
    """Check if file is an image."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return get_file_extension(file_path) in image_extensions


def is_video_file(file_path: str) -> bool:
    """Check if file is a video."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    return get_file_extension(file_path) in video_extensions


def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> list:
    """Find files matching pattern in directory."""
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter out directories
    files = [f for f in files if f.is_file()]
    
    logger.debug(f"Found {len(files)} files matching '{pattern}' in {directory}")
    return files


def copy_file(src: str, dst: str) -> None:
    """Copy file from source to destination."""
    import shutil
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    # Create destination directory if it doesn't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.copy2(src_path, dst_path)
        logger.debug(f"Copied {src} to {dst}")
    
    except Exception as e:
        logger.error(f"Error copying file {src} to {dst}: {e}")
        raise


def move_file(src: str, dst: str) -> None:
    """Move file from source to destination."""
    import shutil
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    # Create destination directory if it doesn't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.move(str(src_path), str(dst_path))
        logger.debug(f"Moved {src} to {dst}")
    
    except Exception as e:
        logger.error(f"Error moving file {src} to {dst}: {e}")
        raise


if __name__ == "__main__":
    # Test utilities
    test_config = {
        "test": True,
        "value": 42,
        "nested": {"key": "value"}
    }
    
    # Test config loading/saving
    save_config(test_config, "test_config.yaml")
    loaded_config = load_config("test_config.yaml")
    print(f"Config test: {loaded_config == test_config}")
    
    # Test JSON loading/saving
    save_json(test_config, "test_config.json")
    loaded_json = load_json("test_config.json")
    print(f"JSON test: {loaded_json == test_config}")
    
    # Clean up
    Path("test_config.yaml").unlink()
    Path("test_config.json").unlink()