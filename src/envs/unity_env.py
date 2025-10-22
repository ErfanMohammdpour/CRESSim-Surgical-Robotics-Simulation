"""
Unity environment wrapper and curriculum learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
import logging
import subprocess
import time
import socket
from pathlib import Path

from .mock_env import MockSuctionEnv
from ..utils.seeding import set_seed

logger = logging.getLogger(__name__)


class UnityEnvWrapper(gym.Env):
    """Wrapper for Unity ML-Agents environment."""
    
    def __init__(
        self,
        unity_build_path: Path,
        port: int = 5005,
        image_size: Tuple[int, int] = (128, 128),
        max_steps: int = 1000
    ):
        super().__init__()
        
        self.unity_build_path = Path(unity_build_path)
        self.port = port
        self.image_size = image_size
        self.max_steps = max_steps
        
        # Action space: [dx, dy, dz, dyaw, suction_toggle]
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1, -0.1, -0.2, 0.0], dtype=np.float32),
            high=np.array([0.1, 0.1, 0.1, 0.2, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: RGB image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(*image_size, 3),
            dtype=np.uint8
        )
        
        # Environment state
        self.step_count = 0
        self.unity_process = None
        self.socket = None
        
        logger.info(f"UnityEnvWrapper initialized: {unity_build_path}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # Set seed
        if seed is not None:
            set_seed(seed)
        
        # Launch Unity if not running
        if self.unity_process is None:
            self._launch_unity()
        
        # Reset step count
        self.step_count = 0
        
        # Send reset command to Unity
        self._send_command("reset")
        
        # Receive initial observation
        observation, info = self._receive_observation()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment."""
        self.step_count += 1
        
        # Send action to Unity
        self._send_action(action)
        
        # Receive observation and reward
        observation, reward, terminated, truncated, info = self._receive_observation()
        
        # Check truncation
        if self.step_count >= self.max_steps:
            truncated = True
        
        return observation, reward, terminated, truncated, info
    
    def _launch_unity(self):
        """Launch Unity headless build."""
        if not self.unity_build_path.exists():
            raise FileNotFoundError(f"Unity build not found: {self.unity_build_path}")
        
        # Launch Unity process
        cmd = [
            str(self.unity_build_path),
            "-batchmode",
            "-nographics",
            "-port", str(self.port)
        ]
        
        self.unity_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Unity to start
        time.sleep(5)
        
        # Connect to Unity
        self._connect_to_unity()
        
        logger.info(f"Unity launched with PID: {self.unity_process.pid}")
    
    def _connect_to_unity(self):
        """Connect to Unity via socket."""
        max_retries = 10
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect(('localhost', self.port))
                logger.info(f"Connected to Unity on port {self.port}")
                return
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    raise ConnectionError(f"Failed to connect to Unity after {max_retries} attempts")
    
    def _send_command(self, command: str):
        """Send command to Unity."""
        if self.socket is None:
            raise RuntimeError("Not connected to Unity")
        
        message = f"{command}\n".encode('utf-8')
        self.socket.send(message)
    
    def _send_action(self, action: np.ndarray):
        """Send action to Unity."""
        if self.socket is None:
            raise RuntimeError("Not connected to Unity")
        
        action_str = ",".join(map(str, action))
        message = f"action:{action_str}\n".encode('utf-8')
        self.socket.send(message)
    
    def _receive_observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Receive observation from Unity."""
        if self.socket is None:
            raise RuntimeError("Not connected to Unity")
        
        # Receive data from Unity
        data = self.socket.recv(1024).decode('utf-8')
        
        # Parse data (simplified - in practice, you'd need proper protocol)
        parts = data.strip().split(',')
        
        # Extract observation (placeholder)
        observation = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        
        # Extract reward and info
        reward = float(parts[0]) if len(parts) > 0 else 0.0
        terminated = bool(int(parts[1])) if len(parts) > 1 else False
        truncated = bool(int(parts[2])) if len(parts) > 2 else False
        
        info = {
            'step_count': self.step_count,
            'liquid_reduction': float(parts[3]) if len(parts) > 3 else 0.0,
            'contaminant_reduction': float(parts[4]) if len(parts) > 4 else 0.0,
            'safety_violations': int(parts[5]) if len(parts) > 5 else 0,
            'collision_count': int(parts[6]) if len(parts) > 6 else 0
        }
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Close environment."""
        if self.socket:
            self.socket.close()
            self.socket = None
        
        if self.unity_process:
            self.unity_process.terminate()
            self.unity_process.wait()
            self.unity_process = None
        
        logger.info("Unity environment closed")


class CurriculumEnv(gym.Env):
    """Environment with curriculum learning."""
    
    def __init__(
        self,
        base_env: gym.Env,
        curriculum_config: Dict[str, Any]
    ):
        super().__init__()
        
        self.base_env = base_env
        self.curriculum_config = curriculum_config
        
        # Copy action and observation spaces
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        
        # Curriculum parameters
        self.min_difficulty = curriculum_config.get('min_difficulty', 0.3)
        self.max_difficulty = curriculum_config.get('max_difficulty', 1.0)
        self.success_threshold = curriculum_config.get('success_threshold', 0.7)
        self.difficulty_increase = curriculum_config.get('difficulty_increase', 0.1)
        self.evaluation_episodes = curriculum_config.get('evaluation_episodes', 10)
        
        # Curriculum state
        self.current_difficulty = self.min_difficulty
        self.episode_count = 0
        self.recent_episodes = []
        
        logger.info(f"CurriculumEnv initialized: {self.min_difficulty} -> {self.max_difficulty}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with curriculum parameters."""
        # Update curriculum
        self._update_curriculum()
        
        # Set curriculum parameters
        curriculum_params = self._get_curriculum_params()
        
        # Reset base environment
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # Add curriculum info
        info.update(curriculum_params)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment."""
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Add curriculum info
        info['current_difficulty'] = self.current_difficulty
        
        return obs, reward, terminated, truncated, info
    
    def _update_curriculum(self):
        """Update curriculum based on recent performance."""
        if len(self.recent_episodes) >= self.evaluation_episodes:
            success_rate = sum(self.recent_episodes) / len(self.recent_episodes)
            
            if success_rate >= self.success_threshold:
                self.current_difficulty = min(
                    self.max_difficulty,
                    self.current_difficulty + self.difficulty_increase
                )
                logger.info(f"Increased difficulty to {self.current_difficulty:.2f}")
    
    def _get_curriculum_params(self) -> Dict[str, Any]:
        """Get curriculum parameters for current difficulty."""
        return {
            'liquid_mass': 100.0 * (0.5 + 0.5 * self.current_difficulty),
            'contaminant_mass': 50.0 * (0.5 + 0.5 * self.current_difficulty),
            'noise_level': 0.1 * (1.0 - self.current_difficulty),
            'time_limit': 1000 * (0.5 + 0.5 * self.current_difficulty),
            'precision_required': self.current_difficulty
        }
    
    def update_episode_result(self, success: bool):
        """Update curriculum with episode result."""
        self.episode_count += 1
        self.recent_episodes.append(success)
        
        # Keep only recent episodes
        if len(self.recent_episodes) > self.evaluation_episodes:
            self.recent_episodes.pop(0)
    
    def close(self):
        """Close environment."""
        self.base_env.close()


def create_environment(
    config: Dict[str, Any],
    mock: bool = True
) -> gym.Env:
    """Create environment based on configuration."""
    if mock:
        # Create mock environment
        env = MockSuctionEnv(
            image_size=tuple(config.get('image_size', [128, 128])),
            max_steps=config.get('max_episode_steps', 1000)
        )
    else:
        # Create Unity environment
        unity_build_path = Path(config.get('unity_build_path', 'sim/CRESSim/Builds/SuctionEnv.x86_64'))
        env = UnityEnvWrapper(
            unity_build_path=unity_build_path,
            port=config.get('unity_port', 5005),
            image_size=tuple(config.get('image_size', [128, 128])),
            max_steps=config.get('max_episode_steps', 1000)
        )
    
    # Add curriculum learning if enabled
    curriculum_config = config.get('curriculum', {})
    if curriculum_config.get('enabled', False):
        env = CurriculumEnv(env, curriculum_config)
    
    return env
