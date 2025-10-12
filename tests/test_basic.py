"""
Basic tests for the project.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that basic imports work."""
    try:
        from utils.io import load_paths_config
        from utils.data import download_kvasir_seg
        from utils.sim import clone_cressim
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_config_loading():
    """Test configuration loading."""
    from utils.io import load_paths_config
    config = load_paths_config()
    assert "data_dir" in config
    assert "sim_dir" in config
    assert "unity_build" in config

def test_directory_structure():
    """Test that required directories exist."""
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        "src",
        "configs", 
        "data",
        "tests"
    ]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        assert dir_path.exists(), f"Directory {dir_name} not found"

def test_config_files():
    """Test that config files exist."""
    base_dir = Path(__file__).parent.parent
    
    config_files = [
        "configs/paths.yaml",
        "configs/env.yaml", 
        "configs/train.yaml",
        "configs/safety.yaml",
        "configs/demos.yaml"
    ]
    
    for config_file in config_files:
        file_path = base_dir / config_file
        assert file_path.exists(), f"Config file {config_file} not found"

if __name__ == "__main__":
    pytest.main([__file__])
