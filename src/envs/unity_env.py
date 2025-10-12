"""
Unity environment wrapper for CRESSim using ML-Agents.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union
import logging
import subprocess
import time
import json
import socket
from pathlib import Path

logger = logging.getLogger(__name__)


class UnityEnv(gym.Env):
    """
    Gymnasium-style wrapper for Unity ML-Agents environment.
    """
    
    def __init__(
        self,
        unity_build_path: str,
        port: int = 5005,
        timeout: int = 30,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.unity_build_path = Path(unity_build_path)
        self.port = port
        self.timeout = timeout
        self.config = config or {}
        
        # Check if Unity build exists
        if not self.unity_build_path.exists():
            raise FileNotFoundError(f"Unity build not found: {unity_build_path}")
        
        # Environment state
        self.unity_process = None
        self.socket = None
        self.connected = False
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )  # [dx, dy, dz, dyaw, suction_toggle]
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255, 
                shape=(3, 84, 84),  # RGB channels first
                dtype=np.uint8
            ),
            'aux': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4,),  # [suction_state, liquid_mass, contaminant_mass, collisions]
                dtype=np.float32
            )
        })
        
        # Episode state
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_info = {}
        
        # Start Unity process
        self._start_unity()
    
    def _start_unity(self) -> None:
        """Start Unity executable."""
        try:
            logger.info(f"Starting Unity executable: {self.unity_build_path}")
            
            # Start Unity process
            self.unity_process = subprocess.Popen([
                str(self.unity_build_path),
                "--mlagents-port", str(self.port),
                "--mlagents-quit-on-load-failure", "false"
            ])
            
            # Wait for Unity to start
            time.sleep(5)
            
            # Connect to Unity
            self._connect_to_unity()
            
        except Exception as e:
            logger.error(f"Failed to start Unity: {e}")
            raise
    
    def _connect_to_unity(self) -> None:
        """Connect to Unity via socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect(('localhost', self.port))
            self.connected = True
            logger.info("Connected to Unity")
        except Exception as e:
            logger.error(f"Failed to connect to Unity: {e}")
            raise
    
    def _send_message(self, message: Dict[str, Any]) -> None:
        """Send message to Unity."""
        if not self.connected:
            raise RuntimeError("Not connected to Unity")
        
        message_str = json.dumps(message) + '\n'
        self.socket.send(message_str.encode())
    
    def _receive_message(self) -> Dict[str, Any]:
        """Receive message from Unity."""
        if not self.connected:
            raise RuntimeError("Not connected to Unity")
        
        data = b''
        while b'\n' not in data:
            chunk = self.socket.recv(1024)
            if not chunk:
                raise ConnectionError("Unity disconnected")
            data += chunk
        
        message_str = data.split(b'\n')[0].decode()
        return json.loads(message_str)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Send reset message
        self._send_message({
            "command": "reset",
            "seed": seed,
            "options": options or {}
        })
        
        # Receive initial observation
        response = self._receive_message()
        
        if response["status"] != "success":
            raise RuntimeError(f"Reset failed: {response.get('error', 'Unknown error')}")
        
        # Parse observation
        obs = self._parse_observation(response["observation"])
        
        # Reset episode state
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_info = {}
        
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step environment."""
        # Ensure action is in correct format
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Send action
        self._send_message({
            "command": "step",
            "action": action.tolist()
        })
        
        # Receive response
        response = self._receive_message()
        
        if response["status"] != "success":
            raise RuntimeError(f"Step failed: {response.get('error', 'Unknown error')}")
        
        # Parse response
        obs = self._parse_observation(response["observation"])
        reward = float(response["reward"])
        terminated = bool(response["terminated"])
        truncated = bool(response["truncated"])
        info = response.get("info", {})
        
        # Update episode state
        self.episode_step += 1
        self.episode_reward += reward
        self.episode_info.update(info)
        
        return obs, reward, terminated, truncated, info
    
    def _parse_observation(self, obs_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse observation from Unity."""
        # Parse image (convert from Unity format to PyTorch format)
        image = np.array(obs_data["image"], dtype=np.uint8)
        if image.shape[0] != 3:  # Convert from HWC to CHW
            image = np.transpose(image, (2, 0, 1))
        
        # Parse auxiliary data
        aux = np.array([
            obs_data.get("suction_state", 0.0),
            obs_data.get("liquid_mass", 0.0),
            obs_data.get("contaminant_mass", 0.0),
            obs_data.get("collisions", 0.0)
        ], dtype=np.float32)
        
        return {
            "image": image,
            "aux": aux
        }
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render environment."""
        if mode == "rgb_array":
            # Request current image from Unity
            self._send_message({"command": "render"})
            response = self._receive_message()
            
            if response["status"] == "success":
                image = np.array(response["image"], dtype=np.uint8)
                return image
            else:
                return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self) -> None:
        """Close environment."""
        if self.connected:
            try:
                self._send_message({"command": "close"})
                self.socket.close()
            except:
                pass
            self.connected = False
        
        if self.unity_process is not None:
            try:
                self.unity_process.terminate()
                self.unity_process.wait(timeout=5)
            except:
                try:
                    self.unity_process.kill()
                except:
                    pass
            self.unity_process = None
    
    def __del__(self):
        """Destructor."""
        self.close()


class MockUnityEnv(UnityEnv):
    """
    Mock Unity environment for testing without Unity build.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Don't call super().__init__ to avoid Unity startup
        gym.Env.__init__(self)
        
        self.config = config or {}
        self.connected = True
        
        # Action and observation spaces (same as Unity)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255, 
                shape=(3, 84, 84),
                dtype=np.uint8
            ),
            'aux': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
        })
        
        # Mock state
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_info = {}
        self.max_steps = 1000
        
        # Mock liquid state
        self.initial_liquid_mass = 1.0
        self.current_liquid_mass = 1.0
        self.initial_contaminant_mass = 0.5
        self.current_contaminant_mass = 0.5
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset mock environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_info = {}
        self.current_liquid_mass = self.initial_liquid_mass
        self.current_contaminant_mass = self.initial_contaminant_mass
        
        # Generate mock observation
        obs = self._generate_mock_observation()
        
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step mock environment."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Simulate suction effect
        suction_active = action[4] > 0.5
        if suction_active:
            # Reduce liquid mass based on action magnitude
            suction_strength = np.linalg.norm(action[:4])
            liquid_reduction = suction_strength * 0.01
            self.current_liquid_mass = max(0, self.current_liquid_mass - liquid_reduction)
            
            # Also reduce contaminant mass
            contaminant_reduction = liquid_reduction * 0.5
            self.current_contaminant_mass = max(0, self.current_contaminant_mass - contaminant_reduction)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination
        terminated = self.current_liquid_mass < 0.01
        truncated = self.episode_step >= self.max_steps
        
        # Generate observation
        obs = self._generate_mock_observation()
        
        # Update episode state
        self.episode_step += 1
        self.episode_reward += reward
        
        info = {
            "liquid_mass_remaining": self.current_liquid_mass,
            "contaminant_mass_remaining": self.current_contaminant_mass,
            "collisions": 0.0,
            "suction_state": float(suction_active)
        }
        
        return obs, reward, terminated, truncated, info
    
    def _generate_mock_observation(self) -> Dict[str, np.ndarray]:
        """Generate mock observation."""
        # Generate random image
        image = np.random.randint(0, 256, (3, 84, 84), dtype=np.uint8)
        
        # Generate auxiliary data
        aux = np.array([
            float(self.current_liquid_mass > 0.01),  # suction_state
            self.current_liquid_mass,
            self.current_contaminant_mass,
            0.0  # collisions
        ], dtype=np.float32)
        
        return {
            "image": image,
            "aux": aux
        }
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for mock environment."""
        # Liquid reduction reward
        liquid_reward = (self.initial_liquid_mass - self.current_liquid_mass) * 10.0
        
        # Contaminant reduction reward
        contaminant_reward = (self.initial_contaminant_mass - self.current_contaminant_mass) * 5.0
        
        # Action smoothness penalty
        action_penalty = -np.linalg.norm(action) * 0.01
        
        # Time penalty
        time_penalty = -0.01
        
        return liquid_reward + contaminant_reward + action_penalty + time_penalty
    
    def close(self) -> None:
        """Close mock environment."""
        self.connected = False
