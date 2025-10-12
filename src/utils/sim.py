"""
Simulator utilities for cloning and managing CRESSim.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def clone_cressim(sim_dir: str, force: bool = False) -> None:
    """Clone CRESSim repository."""
    sim_dir = Path(sim_dir)
    
    if sim_dir.exists() and not force:
        logger.info(f"CRESSim already exists at {sim_dir}")
        return
    
    # Remove existing directory if force
    if sim_dir.exists() and force:
        logger.info(f"Removing existing CRESSim directory: {sim_dir}")
        shutil.rmtree(sim_dir)
    
    # Clone repository
    repo_url = "https://github.com/tbs-ualberta/CRESSim.git"
    logger.info(f"Cloning CRESSim from {repo_url}")
    
    try:
        subprocess.run([
            "git", "clone", repo_url, str(sim_dir)
        ], check=True, capture_output=True)
        logger.info(f"Successfully cloned CRESSim to {sim_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone CRESSim: {e}")
        raise


def check_unity_build(unity_build_path: str) -> bool:
    """Check if Unity build exists and is valid."""
    build_path = Path(unity_build_path)
    
    if not build_path.exists():
        return False
    
    # Check if it's a valid executable
    if not build_path.is_file():
        return False
    
    # On Windows, check for .exe extension
    if build_path.suffix.lower() != '.exe':
        return False
    
    return True


def get_unity_build_info(unity_build_path: str) -> dict:
    """Get information about Unity build."""
    build_path = Path(unity_build_path)
    
    if not build_path.exists():
        return {
            "status": "not_found",
            "path": str(build_path),
            "message": "Unity build not found. Please build the SuctionEnv scene."
        }
    
    # Get file size and modification time
    stat = build_path.stat()
    
    return {
        "status": "found",
        "path": str(build_path),
        "size_mb": stat.st_size / (1024 * 1024),
        "modified": stat.st_mtime,
        "message": "Unity build ready"
    }


def validate_sim_setup(sim_dir: str, unity_build_path: str) -> dict:
    """Validate complete simulator setup."""
    sim_dir = Path(sim_dir)
    unity_build_path = Path(unity_build_path)
    
    result = {
        "sim_dir_exists": sim_dir.exists(),
        "unity_build_exists": unity_build_path.exists(),
        "ready": False,
        "issues": []
    }
    
    if not result["sim_dir_exists"]:
        result["issues"].append(f"Simulator directory not found: {sim_dir}")
    
    if not result["unity_build_exists"]:
        result["issues"].append(f"Unity build not found: {unity_build_path}")
    
    # Check for required Unity files
    if result["sim_dir_exists"]:
        required_files = [
            "Assets/Scenes/SuctionEnv.unity",
            "Assets/Scripts/SuctionAgent.cs",
            "ProjectSettings/ProjectVersion.txt"
        ]
        
        for file_path in required_files:
            full_path = sim_dir / file_path
            if not full_path.exists():
                result["issues"].append(f"Required file missing: {file_path}")
    
    result["ready"] = len(result["issues"]) == 0
    
    return result


def get_build_instructions() -> str:
    """Get Unity build instructions."""
    return """
Unity Build Instructions:

1. Open Unity Hub
2. Click "Open" and select the CRESSim project folder
3. Wait for Unity to load the project
4. Open the scene: Assets/Scenes/SuctionEnv.unity
5. Go to File → Build Settings
6. Select "PC, Mac & Linux Standalone" → "Windows"
7. Click "Build" and save as: sim/CRESSim/Build/SuctionEnv.exe
8. Wait for the build to complete

The executable should appear at: sim/CRESSim/Build/SuctionEnv.exe
"""
