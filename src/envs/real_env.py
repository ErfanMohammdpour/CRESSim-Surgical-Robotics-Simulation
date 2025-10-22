#!/usr/bin/env python3
"""
Real Surgical Environment Wrapper
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class RealSurgicalEnv(gym.Env):
    """Real surgical environment wrapper"""
    
    def __init__(self, data_dir: str, episode_id: str = None):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.episode_id = episode_id
        
        # Load episode data
        self.episode_data = self._load_episode_data()
        
        # Action space: [dx, dy, dz, dyaw, suction_toggle]
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1, -0.1, -0.2, 0.0], dtype=np.float32),
            high=np.array([0.1, 0.1, 0.1, 0.2, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )
        
        # Episode state
        self.current_step = 0
        self.max_steps = len(self.episode_data.get('images', []))
        self.total_reward = 0.0
        
        # Task state
        self.initial_liquid_mass = 100.0
        self.current_liquid_mass = 100.0
        self.initial_contaminant_mass = 50.0
        self.current_contaminant_mass = 50.0
        
        # Performance tracking
        self.collision_count = 0
        self.safety_violations = 0
        
        logger.info(f"RealSurgicalEnv initialized for episode: {self.episode_id}")
        logger.info(f"Max steps: {self.max_steps}")
    
    def _load_episode_data(self) -> Dict[str, Any]:
        """Load episode data from metadata"""
        try:
            metadata_file = self.data_dir / "metadata.json"
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found: {metadata_file}")
                return self._create_sample_episode_data()
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                episodes = metadata.get('episodes', [])
            
            if not episodes:
                logger.warning("No episodes found in metadata")
                return self._create_sample_episode_data()
            
            # Use specified episode or first available
            if self.episode_id is None:
                self.episode_id = episodes[0]['episode_id']
            
            # Find the episode
            episode_data = next(
                (ep for ep in episodes if ep['episode_id'] == self.episode_id), 
                None
            )
            
            if episode_data is None:
                logger.warning(f"Episode {self.episode_id} not found, using first episode")
                episode_data = episodes[0]
                self.episode_id = episode_data['episode_id']
            
            return episode_data
            
        except Exception as e:
            logger.error(f"Error loading episode data: {e}")
            return self._create_sample_episode_data()
    
    def _create_sample_episode_data(self) -> Dict[str, Any]:
        """Create sample episode data for testing"""
        logger.info("Creating sample episode data...")
        
        # Create sample images
        images_dir = self.data_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        sample_images = []
        sample_actions = []
        
        for i in range(10):  # Create 10 sample steps
            # Create synthetic image
            img = self._create_synthetic_image(i)
            img_path = f"sample_{i:03d}.jpg"
            img.save(images_dir / img_path)
            sample_images.append(img_path)
            
            # Create sample action
            action = [
                np.random.uniform(-0.05, 0.05),  # dx
                np.random.uniform(-0.05, 0.05),  # dy
                np.random.uniform(-0.02, 0.02),  # dz
                np.random.uniform(-0.1, 0.1),    # dyaw
                np.random.choice([0.0, 1.0])     # suction_toggle
            ]
            sample_actions.append(action)
        
        episode_data = {
            'episode_id': 'sample_episode',
            'images': sample_images,
            'actions': sample_actions,
            'rewards': [np.random.uniform(0.1, 1.0) for _ in range(10)],
            'success': True,
            'liquid_reduction': 0.85,
            'contaminant_reduction': 0.78
        }
        
        # Save metadata
        metadata = {
            'episodes': [episode_data]
        }
        
        with open(self.data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.episode_id = 'sample_episode'
        return episode_data
    
    def _create_synthetic_image(self, step: int) -> Image.Image:
        """Create synthetic surgical image"""
        # Create a synthetic surgical scene
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Background (dark surgical field)
        img[:, :] = [20, 20, 30]
        
        # Add some structure based on step
        center_x, center_y = 64, 64
        
        # Draw surgical instruments with some variation
        tool_x = int(center_x + np.sin(step * 0.3) * 20)
        tool_y = int(center_y + np.cos(step * 0.3) * 15)
        
        # Ensure coordinates are within bounds
        tool_x = max(10, min(118, tool_x))
        tool_y = max(10, min(118, tool_y))
        
        # Draw tool
        cv2.circle(img, (tool_x, tool_y), 6, (255, 255, 255), -1)
        cv2.circle(img, (tool_x, tool_y), 4, (0, 255, 0), -1)
        
        # Add some liquid visualization
        liquid_alpha = max(0, 1.0 - step * 0.1)
        if liquid_alpha > 0:
            liquid_color = [0, 100, 200]
            liquid_region = img[center_x-15:center_x+15, center_y-15:center_y+15]
            liquid_region[:] = liquid_region * (1 - liquid_alpha) + np.array(liquid_color) * liquid_alpha
        
        # Add noise for realism
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_reward = 0.0
        self.current_liquid_mass = self.initial_liquid_mass
        self.current_contaminant_mass = self.initial_contaminant_mass
        self.collision_count = 0
        self.safety_violations = 0
        
        # Load first image
        obs = self._load_image(self.current_step)
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step"""
        self.current_step += 1
        
        # Load next image
        obs = self._load_image(self.current_step)
        
        # Calculate reward based on action effectiveness
        reward = self._calculate_reward(action)
        self.total_reward += reward
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _load_image(self, step: int) -> np.ndarray:
        """Load image for current step"""
        images = self.episode_data.get('images', [])
        
        if step < len(images):
            img_path = self.data_dir / 'images' / images[step]
            try:
                if img_path.exists():
                    image = Image.open(img_path).convert('RGB')
                else:
                    # Create synthetic image if file doesn't exist
                    image = self._create_synthetic_image(step)
                
                image = image.resize((128, 128))
                return np.array(image, dtype=np.uint8)
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")
                # Create synthetic image as fallback
                image = self._create_synthetic_image(step)
                return np.array(image, dtype=np.uint8)
        else:
            # Return last image if step exceeds available images
            return self._load_image(len(images) - 1)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on action"""
        dx, dy, dz, dyaw, suction_toggle = action
        
        # Liquid removal simulation
        if suction_toggle > 0.5:
            # Base removal rate
            base_removal = 0.15
            
            # Position factor (better positioning = more removal)
            position_factor = 1.0 - (abs(dx) + abs(dy) + abs(dz)) * 2.0
            position_factor = max(0.1, position_factor)
            
            # Suction strength factor
            suction_strength = 1.0
            
            # Final removal rate
            removal_rate = base_removal * position_factor * suction_strength
            self.current_liquid_mass = max(0, self.current_liquid_mass - removal_rate)
            
            # Contaminant removal (slower than liquid)
            contaminant_removal = removal_rate * 0.6
            self.current_contaminant_mass = max(0, self.current_contaminant_mass - contaminant_removal)
        
        # Simulate collisions (random for now)
        if np.random.random() < 0.02:  # 2% chance per step
            self.collision_count += 1
        
        # Simulate safety violations (random for now)
        if np.random.random() < 0.01:  # 1% chance per step
            self.safety_violations += 1
        
        # Calculate reward components
        liquid_reduction = (self.initial_liquid_mass - self.current_liquid_mass) / self.initial_liquid_mass
        contaminant_reduction = (self.initial_contaminant_mass - self.current_contaminant_mass) / self.initial_contaminant_mass
        
        # Reward calculation
        liquid_reward = 2.0 * liquid_reduction
        contaminant_reward = 1.5 * contaminant_reduction
        action_penalty = -0.001 * np.linalg.norm(action)
        time_penalty = -0.005
        collision_penalty = -2.0 * self.collision_count
        safety_penalty = -5.0 * self.safety_violations
        
        return liquid_reward + contaminant_reward + action_penalty + time_penalty + collision_penalty + safety_penalty
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        liquid_complete = self.current_liquid_mass < 5.0  # 95% liquid removed
        contaminant_complete = self.current_contaminant_mass < 2.5  # 95% contaminant removed
        return liquid_complete and contaminant_complete
    
    def _get_info(self) -> Dict[str, Any]:
        """Get episode information"""
        return {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'liquid_mass': self.current_liquid_mass,
            'contaminant_mass': self.current_contaminant_mass,
            'liquid_reduction': (self.initial_liquid_mass - self.current_liquid_mass) / self.initial_liquid_mass,
            'contaminant_reduction': (self.initial_contaminant_mass - self.current_contaminant_mass) / self.initial_contaminant_mass,
            'collision_count': self.collision_count,
            'safety_violations': self.safety_violations,
            'episode_id': self.episode_id
        }
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == "rgb_array":
            return self._load_image(self.current_step)
        return None
    
    def close(self) -> None:
        """Close the environment"""
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
    # Test real environment
    env = RealSurgicalEnv("data/real_dataset")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward={reward:.3f}, Liquid={info['liquid_mass']:.1f}")
        
        if terminated or truncated:
            break
    
    env.close()
