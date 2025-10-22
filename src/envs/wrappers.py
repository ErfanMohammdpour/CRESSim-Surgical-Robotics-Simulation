"""
Environment wrappers for preprocessing observations and actions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Tuple, Union
import cv2


class FrameStack(gym.ObservationWrapper):
    """Stack consecutive frames for temporal information."""
    
    def __init__(self, env: gym.Env, num_frames: int = 4):
        super().__init__(env)
        self.num_frames = num_frames
        
        # Update observation space
        low = np.repeat(self.observation_space.low, num_frames, axis=0)
        high = np.repeat(self.observation_space.high, num_frames, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
        
        self.frames = []
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.num_frames
        return self._get_obs(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        self.frames.pop(0)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        return np.concatenate(self.frames, axis=0)


class ResizeObservation(gym.ObservationWrapper):
    """Resize observation to specified dimensions."""
    
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        return cv2.resize(obs, (self.width, self.height))


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to [0, 1] range."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


class ActionScale(gym.ActionWrapper):
    """Scale continuous actions to specified range."""
    
    def __init__(self, env: gym.Env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high
        
        # Update action space
        self.action_space = spaces.Box(
            low=low, high=high, shape=env.action_space.shape, dtype=np.float32
        )
    
    def action(self, action: np.ndarray) -> np.ndarray:
        # Scale from [-1, 1] to original range
        original_low = self.env.action_space.low
        original_high = self.env.action_space.high
        
        scaled_action = (action - self.low) / (self.high - self.low)
        scaled_action = scaled_action * (original_high - original_low) + original_low
        
        return np.clip(scaled_action, original_low, original_high)


class DomainRandomization(gym.Wrapper):
    """Apply domain randomization to observations."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env)
        self.config = config
        self.rng = np.random.RandomState()
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._apply_randomization(obs), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._apply_randomization(obs), reward, terminated, truncated, info
    
    def _apply_randomization(self, obs: np.ndarray) -> np.ndarray:
        """Apply domain randomization to observation."""
        if 'brightness' in self.config:
            brightness_range = self.config['brightness']
            brightness_factor = self.rng.uniform(
                1.0 - brightness_range, 1.0 + brightness_range
            )
            obs = np.clip(obs * brightness_factor, 0, 255)
        
        if 'contrast' in self.config:
            contrast_range = self.config['contrast']
            contrast_factor = self.rng.uniform(
                1.0 - contrast_range, 1.0 + contrast_range
            )
            mean = np.mean(obs)
            obs = np.clip((obs - mean) * contrast_factor + mean, 0, 255)
        
        if 'hue' in self.config:
            hue_shift = self.rng.uniform(-self.config['hue'], self.config['hue'])
            # Convert to HSV, shift hue, convert back
            hsv = cv2.cvtColor(obs.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            obs = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        if 'blur' in self.config and self.rng.random() < 0.3:
            blur_sigma = self.rng.uniform(0, self.config['blur'])
            obs = cv2.GaussianBlur(obs.astype(np.uint8), (0, 0), blur_sigma)
        
        return obs.astype(np.float32)


class SafetyWrapper(gym.Wrapper):
    """Wrapper that applies safety constraints to actions."""
    
    def __init__(self, env: gym.Env, safety_config: Dict[str, Any]):
        super().__init__(env)
        self.safety_config = safety_config
        self.violation_count = 0
        self.max_violations = safety_config.get('max_violations_per_episode', 10)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Check safety constraints
        safe_action = self._apply_safety_constraints(action)
        
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        
        # Add safety information to info
        info['safety_violation'] = not np.array_equal(action, safe_action)
        info['violation_count'] = self.violation_count
        
        # Check for emergency stop
        if self.violation_count >= self.max_violations:
            terminated = True
            info['emergency_stop'] = True
        
        return obs, reward, terminated, truncated, info
    
    def _apply_safety_constraints(self, action: np.ndarray) -> np.ndarray:
        """Apply safety constraints to action."""
        # This is a placeholder - actual implementation would use
        # safety shield predictions
        return action
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.violation_count = 0
        return self.env.reset(**kwargs)


def make_env(
    env_id: str,
    config: Dict[str, Any],
    safety_config: Dict[str, Any] = None
) -> gym.Env:
    """Create and configure environment with wrappers."""
    
    # Create base environment
    if env_id == "mock":
        from .mock_env import MockSuctionEnv
        env = MockSuctionEnv(config)
    else:
        from .unity_env import UnitySuctionEnv
        env = UnitySuctionEnv(config)
    
    # Apply wrappers
    env = ResizeObservation(env, config.get('image_width', 84), config.get('image_height', 84))
    env = NormalizeObservation(env)
    env = FrameStack(env, config.get('frame_stack', 4))
    env = ActionScale(env, config.get('action_low', -1.0), config.get('action_high', 1.0))
    
    # Apply domain randomization
    if 'domain_randomization' in config:
        env = DomainRandomization(env, config['domain_randomization'])
    
    # Apply safety wrapper
    if safety_config:
        env = SafetyWrapper(env, safety_config)
    
    return env
