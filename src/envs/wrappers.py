"""
Environment wrappers for frame stacking, augmentation, and reward shaping.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union
import cv2
from collections import deque
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FrameStackWrapper(gym.Wrapper):
    """Wrapper for stacking frames."""
    
    def __init__(self, env: gym.Env, num_frames: int = 4):
        super().__init__(env)
        self.num_frames = num_frames
        
        # Update observation space
        old_obs_space = env.observation_space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(num_frames * 3, 84, 84),  # Stacked frames
                dtype=np.uint8
            ),
            'aux': old_obs_space['aux']
        })
        
        # Frame buffer
        self.frames = deque(maxlen=num_frames)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Initialize frame buffer
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(obs['image'])
        
        return self._get_stacked_obs(), info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add new frame
        self.frames.append(obs['image'])
        
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self) -> Dict[str, np.ndarray]:
        """Get stacked observation."""
        stacked_frames = np.concatenate(list(self.frames), axis=0)
        return {
            'image': stacked_frames,
            'aux': self.env.observation_space['aux'].sample()  # Keep aux as is
        }


class ImageAugmentationWrapper(gym.Wrapper):
    """Wrapper for image augmentation during training."""
    
    def __init__(self, env: gym.Env, augment_prob: float = 0.5):
        super().__init__(env)
        self.augment_prob = augment_prob
        
        # Define augmentation pipeline
        self.augment = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.3
            ),
            A.Blur(
                blur_limit=3,
                p=0.3
            ),
        ])
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply augmentation with probability
        if np.random.random() < self.augment_prob:
            obs['image'] = self._augment_image(obs['image'])
        
        return obs, reward, terminated, truncated, info
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image."""
        # Convert from CHW to HWC for albumentations
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Apply augmentation
        augmented = self.augment(image=image)['image']
        
        # Convert back to CHW
        if augmented.shape[2] == 3:
            augmented = np.transpose(augmented, (2, 0, 1))
        
        return augmented.astype(np.uint8)


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper for reward shaping."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env)
        self.config = config
        
        # Reward weights
        self.weights = config.get('reward_weights', {})
        self.alpha = self.weights.get('alpha', 1.0)  # liquid mass reduction
        self.beta = self.weights.get('beta', 0.5)    # contaminant mass reduction
        self.lambda_time = self.weights.get('lambda_time', -0.01)  # time penalty
        self.lambda_action = self.weights.get('lambda_action', -0.001)  # action smoothness
        self.lambda_collision = self.weights.get('lambda_collision', -1.0)  # collision penalty
        self.lambda_safety = self.weights.get('lambda_safety', -2.0)  # safety violation penalty
        
        # Episode state
        self.prev_liquid_mass = None
        self.prev_contaminant_mass = None
        self.prev_action = None
        self.episode_collisions = 0
        self.episode_safety_violations = 0
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset episode state
        self.prev_liquid_mass = info.get('liquid_mass_remaining', 0.0)
        self.prev_contaminant_mass = info.get('contaminant_mass_remaining', 0.0)
        self.prev_action = None
        self.episode_collisions = 0
        self.episode_safety_violations = 0
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate shaped reward
        shaped_reward = self._calculate_shaped_reward(action, info)
        
        # Update episode state
        self.prev_liquid_mass = info.get('liquid_mass_remaining', 0.0)
        self.prev_contaminant_mass = info.get('contaminant_mass_remaining', 0.0)
        self.prev_action = action.copy()
        
        # Track violations
        if info.get('collisions', 0) > 0:
            self.episode_collisions += 1
        if info.get('safety_violation', False):
            self.episode_safety_violations += 1
        
        # Add episode info
        info.update({
            'episode_collisions': self.episode_collisions,
            'episode_safety_violations': self.episode_safety_violations,
            'shaped_reward': shaped_reward
        })
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _calculate_shaped_reward(self, action: np.ndarray, info: Dict[str, Any]) -> float:
        """Calculate shaped reward."""
        reward = 0.0
        
        # Liquid mass reduction reward
        current_liquid_mass = info.get('liquid_mass_remaining', 0.0)
        if self.prev_liquid_mass is not None:
            liquid_reduction = self.prev_liquid_mass - current_liquid_mass
            reward += self.alpha * liquid_reduction
        
        # Contaminant mass reduction reward
        current_contaminant_mass = info.get('contaminant_mass_remaining', 0.0)
        if self.prev_contaminant_mass is not None:
            contaminant_reduction = self.prev_contaminant_mass - current_contaminant_mass
            reward += self.beta * contaminant_reduction
        
        # Time penalty
        reward += self.lambda_time
        
        # Action smoothness penalty
        if self.prev_action is not None:
            action_diff = np.linalg.norm(action - self.prev_action)
            reward += self.lambda_action * action_diff
        
        # Collision penalty
        if info.get('collisions', 0) > 0:
            reward += self.lambda_collision
        
        # Safety violation penalty
        if info.get('safety_violation', False):
            reward += self.lambda_safety
        
        return reward


class DomainRandomizationWrapper(gym.Wrapper):
    """Wrapper for domain randomization."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env)
        self.config = config
        self.randomization_config = config.get('domain_randomization', {})
        
        # Randomization state
        self.current_lighting = None
        self.current_camera_noise = None
        self.current_texture_scale = None
        self.current_fluid_mass = None
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Apply domain randomization
        self._apply_randomization()
        
        return obs, info
    
    def _apply_randomization(self) -> None:
        """Apply domain randomization."""
        # Lighting randomization
        lighting_config = self.randomization_config.get('lighting', {})
        if lighting_config:
            intensity_range = lighting_config.get('intensity_range', [0.7, 1.3])
            color_temp_range = lighting_config.get('color_temp_range', [3000, 7000])
            
            self.current_lighting = {
                'intensity': np.random.uniform(*intensity_range),
                'color_temp': np.random.uniform(*color_temp_range)
            }
        
        # Camera noise randomization
        camera_config = self.randomization_config.get('camera', {})
        if camera_config:
            noise_std = camera_config.get('noise_std', 0.01)
            self.current_camera_noise = np.random.normal(0, noise_std)
        
        # Texture randomization
        texture_config = self.randomization_config.get('textures', {})
        if texture_config:
            scale_range = texture_config.get('scale_range', [0.8, 1.2])
            self.current_texture_scale = np.random.uniform(*scale_range)
        
        # Fluid randomization
        fluid_config = self.randomization_config.get('fluid', {})
        if fluid_config:
            mass_range = fluid_config.get('initial_mass_range', [0.8, 1.2])
            self.current_fluid_mass = np.random.uniform(*mass_range)


class CurriculumWrapper(gym.Wrapper):
    """Wrapper for curriculum learning."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env)
        self.config = config
        self.curriculum_config = config.get('curriculum', {})
        
        # Curriculum state
        self.current_difficulty = self.curriculum_config.get('min_difficulty', 0.3)
        self.success_rate = 0.0
        self.episode_count = 0
        self.evaluation_episodes = self.curriculum_config.get('evaluation_episodes', 10)
        self.update_frequency = self.curriculum_config.get('update_frequency', 1000)
        self.success_threshold = self.curriculum_config.get('success_threshold', 0.7)
        self.difficulty_increase = self.curriculum_config.get('difficulty_increase', 0.1)
        
        # Episode tracking
        self.recent_episodes = []
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track episode completion
        if terminated or truncated:
            success = self._is_episode_successful(info)
            self.recent_episodes.append(success)
            self.episode_count += 1
            
            # Update curriculum
            if self.episode_count % self.update_frequency == 0:
                self._update_curriculum()
        
        return obs, reward, terminated, truncated, info
    
    def _is_episode_successful(self, info: Dict[str, Any]) -> bool:
        """Check if episode was successful."""
        liquid_mass = info.get('liquid_mass_remaining', 1.0)
        contaminant_mass = info.get('contaminant_mass_remaining', 1.0)
        
        # Success if most liquid and contaminant is removed
        liquid_success = liquid_mass < 0.1
        contaminant_success = contaminant_mass < 0.05
        
        return liquid_success and contaminant_success
    
    def _update_curriculum(self) -> None:
        """Update curriculum difficulty."""
        if len(self.recent_episodes) >= self.evaluation_episodes:
            # Calculate success rate
            self.success_rate = np.mean(self.recent_episodes[-self.evaluation_episodes:])
            
            # Increase difficulty if success rate is high enough
            if self.success_rate >= self.success_threshold:
                self.current_difficulty = min(
                    self.curriculum_config.get('max_difficulty', 1.0),
                    self.current_difficulty + self.difficulty_increase
                )
            
            # Clear recent episodes
            self.recent_episodes = []
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get current curriculum information."""
        return {
            'current_difficulty': self.current_difficulty,
            'success_rate': self.success_rate,
            'episode_count': self.episode_count
        }
