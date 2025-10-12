"""
Mock environment for testing without Unity.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MockSuctionEnv(gym.Env):
    """Mock suction environment for testing."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        max_steps: int = 1000,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.image_size = image_size
        self.max_steps = max_steps
        self.reward_weights = reward_weights or {
            "alpha": 2.0,  # liquid mass reduction
            "beta": 1.5,   # contaminant mass reduction
            "lambda_time": -0.005,  # time penalty
            "lambda_action": -0.0005,  # action smoothness
            "lambda_collision": -2.0,  # collision penalty
            "lambda_safety": -5.0  # safety violation penalty
        }
        
        # Action space: [dx, dy, dz, dyaw, suction_toggle]
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1, -0.1, -0.2, 0.0]),
            high=np.array([0.1, 0.1, 0.1, 0.2, 1.0]),
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
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = 0.0  # yaw
        self.suction_active = False
        
        # Task state
        self.liquid_mass = 100.0
        self.contaminant_mass = 50.0
        self.collision_count = 0
        self.safety_violations = 0
        
        # Performance tracking
        self.initial_liquid_mass = self.liquid_mass
        self.initial_contaminant_mass = self.contaminant_mass
        
        logger.info(f"MockSuctionEnv initialized: {image_size}, max_steps={max_steps}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.step_count = 0
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = 0.0
        self.suction_active = False
        
        # Reset task state
        self.liquid_mass = 100.0
        self.contaminant_mass = 50.0
        self.collision_count = 0
        self.safety_violations = 0
        
        # Generate initial observation
        observation = self._generate_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Parse action
        dx, dy, dz, dyaw, suction_toggle = action
        
        # Update position and orientation
        self.position += np.array([dx, dy, dz])
        self.orientation += dyaw
        
        # Update suction state
        if suction_toggle > 0.5:
            self.suction_active = True
        else:
            self.suction_active = False
        
        # Simulate task dynamics
        self._simulate_task_dynamics()
        
        # Generate observation
        observation = self._generate_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _simulate_task_dynamics(self):
        """Simulate task-specific dynamics."""
        # Simulate liquid removal based on suction
        if self.suction_active:
            # Liquid removal rate depends on position and suction strength
            removal_rate = 0.1 * np.exp(-np.linalg.norm(self.position) * 0.1)
            self.liquid_mass = max(0, self.liquid_mass - removal_rate)
            
            # Contaminant removal (slower than liquid)
            contaminant_removal = removal_rate * 0.5
            self.contaminant_mass = max(0, self.contaminant_mass - contaminant_removal)
        
        # Simulate collisions (random for mock)
        if np.random.random() < 0.01:  # 1% chance per step
            self.collision_count += 1
        
        # Simulate safety violations (random for mock)
        if np.random.random() < 0.005:  # 0.5% chance per step
            self.safety_violations += 1
    
    def _generate_observation(self) -> np.ndarray:
        """Generate observation image."""
        # Create synthetic surgical scene
        img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # Background (dark surgical field)
        img[:, :] = [20, 20, 30]
        
        # Add some structure (simulate surgical instruments)
        center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
        
        # Draw suction tool
        tool_x = int(center_x + self.position[0] * 100)
        tool_y = int(center_y + self.position[1] * 100)
        
        if 0 <= tool_x < self.image_size[0] and 0 <= tool_y < self.image_size[1]:
            # Tool body
            cv2.circle(img, (tool_x, tool_y), 5, (255, 255, 255), -1)
            
            # Suction tip
            if self.suction_active:
                cv2.circle(img, (tool_x, tool_y), 3, (0, 255, 0), -1)
            else:
                cv2.circle(img, (tool_x, tool_y), 3, (255, 0, 0), -1)
        
        # Add liquid visualization
        liquid_alpha = self.liquid_mass / self.initial_liquid_mass
        if liquid_alpha > 0:
            liquid_color = [0, 100, 200]  # Blue
            liquid_region = img[center_x-20:center_x+20, center_y-20:center_y+20]
            liquid_region[:] = liquid_region * (1 - liquid_alpha) + np.array(liquid_color) * liquid_alpha
        
        # Add contaminant visualization
        contaminant_alpha = self.contaminant_mass / self.initial_contaminant_mass
        if contaminant_alpha > 0:
            contaminant_color = [200, 100, 0]  # Orange
            contaminant_region = img[center_x-15:center_x+15, center_y-15:center_y+15]
            contaminant_region[:] = contaminant_region * (1 - contaminant_alpha) + np.array(contaminant_color) * contaminant_alpha
        
        # Add noise for realism
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        # Liquid mass reduction reward
        liquid_reduction = (self.initial_liquid_mass - self.liquid_mass) / self.initial_liquid_mass
        liquid_reward = self.reward_weights["alpha"] * liquid_reduction
        
        # Contaminant mass reduction reward
        contaminant_reduction = (self.initial_contaminant_mass - self.contaminant_mass) / self.initial_contaminant_mass
        contaminant_reward = self.reward_weights["beta"] * contaminant_reduction
        
        # Time penalty
        time_penalty = self.reward_weights["lambda_time"] * self.step_count
        
        # Action smoothness penalty (simplified)
        action_penalty = self.reward_weights["lambda_action"] * np.linalg.norm(self.position)
        
        # Collision penalty
        collision_penalty = self.reward_weights["lambda_collision"] * self.collision_count
        
        # Safety violation penalty
        safety_penalty = self.reward_weights["lambda_safety"] * self.safety_violations
        
        # Total reward
        total_reward = (
            liquid_reward + 
            contaminant_reward + 
            time_penalty + 
            action_penalty + 
            collision_penalty + 
            safety_penalty
        )
        
        return total_reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if task is complete
        liquid_complete = self.liquid_mass < 5.0  # 95% liquid removed
        contaminant_complete = self.contaminant_mass < 2.5  # 95% contaminant removed
        
        return liquid_complete and contaminant_complete
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            "step_count": self.step_count,
            "position": self.position.copy(),
            "orientation": self.orientation,
            "suction_active": self.suction_active,
            "liquid_mass": self.liquid_mass,
            "contaminant_mass": self.contaminant_mass,
            "collision_count": self.collision_count,
            "safety_violations": self.safety_violations,
            "liquid_reduction": (self.initial_liquid_mass - self.liquid_mass) / self.initial_liquid_mass,
            "contaminant_reduction": (self.initial_contaminant_mass - self.contaminant_mass) / self.initial_contaminant_mass
        }
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "rgb_array":
            return self._generate_observation()
        return None
    
    def close(self):
        """Close the environment."""
        pass


# Add cv2 import for drawing functions
try:
    import cv2
except ImportError:
    # Fallback if cv2 is not available
    def cv2_circle(img, center, radius, color, thickness):
        y, x = center
        h, w = img.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        img[ny, nx] = color
    
    cv2 = type('cv2', (), {'circle': cv2_circle})()


if __name__ == "__main__":
    # Test mock environment
    env = MockSuctionEnv(image_size=(128, 128), max_steps=100)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward={reward:.3f}, Liquid={info['liquid_mass']:.1f}")
        
        if terminated or truncated:
            break
    
    env.close()

