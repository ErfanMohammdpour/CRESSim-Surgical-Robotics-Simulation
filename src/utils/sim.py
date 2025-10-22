"""
Simulation utilities for CRESSim integration.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def clone_cressim(output_dir: Path, force: bool = False) -> None:
    """Clone CRESSim repository."""
    output_dir = Path(output_dir)
    
    # Check if already cloned
    if not force and output_dir.exists():
        logger.info("CRESSim already cloned")
        return
    
    # Remove existing directory if force
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Clone repository
    url = "https://github.com/tbs-ualberta/CRESSim.git"
    logger.info(f"Cloning CRESSim from {url}")
    
    try:
        subprocess.run([
            "git", "clone", "--depth", "1", url, str(output_dir)
        ], check=True, capture_output=True)
        
        logger.info(f"Cloned CRESSim to {output_dir}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone CRESSim: {e}")
        raise
    except FileNotFoundError:
        logger.error("Git not found. Please install Git to clone CRESSim")
        raise


def check_unity_build(build_path: Path) -> bool:
    """Check if Unity build exists."""
    build_path = Path(build_path)
    
    if build_path.exists():
        logger.info(f"Unity build found: {build_path}")
        return True
    else:
        logger.warning(f"Unity build not found: {build_path}")
        return False


def get_unity_build_instructions() -> str:
    """Get instructions for building Unity environment."""
    instructions = """
Unity Build Instructions:

1. Open Unity Hub
2. Open the CRESSim project from sim/CRESSim/
3. Go to File -> Build Settings
4. Select the SuctionEnv scene
5. Choose Platform: Linux (for headless) or Windows
6. Set Build Settings:
   - Target Platform: Linux x86_64 (for headless) or Windows x86_64
   - Architecture: x86_64
   - Build Type: Headless (for Linux)
7. Click Build and save to sim/CRESSim/Builds/
8. The executable will be named SuctionEnv.x86_64

For headless server deployment:
- Build as Linux x86_64 headless
- Transfer the build folder to your server
- Run with: ./SuctionEnv.x86_64 -batchmode -nographics -logFile /tmp/cressim.log
"""
    return instructions


def launch_unity_headless(
    build_path: Path,
    log_file: Optional[Path] = None,
    port: int = 5005
) -> subprocess.Popen:
    """Launch Unity headless build."""
    build_path = Path(build_path)
    
    if not build_path.exists():
        raise FileNotFoundError(f"Unity build not found: {build_path}")
    
    # Prepare command
    cmd = [str(build_path)]
    
    # Add headless flags
    cmd.extend([
        "-batchmode",
        "-nographics",
        "-port", str(port)
    ])
    
    # Add log file if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-logFile", str(log_file)])
    
    logger.info(f"Launching Unity headless: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Unity process started with PID: {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"Failed to launch Unity: {e}")
        raise


def check_unity_connection(port: int = 5005) -> bool:
    """Check if Unity is accepting connections."""
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            result = s.connect_ex(('localhost', port))
            return result == 0
    except Exception:
        return False


def wait_for_unity_connection(port: int = 5005, timeout: int = 60) -> bool:
    """Wait for Unity to accept connections."""
    import time
    
    logger.info(f"Waiting for Unity connection on port {port}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_unity_connection(port):
            logger.info("Unity connection established")
            return True
        time.sleep(1)
    
    logger.error(f"Unity connection timeout after {timeout}s")
    return False
