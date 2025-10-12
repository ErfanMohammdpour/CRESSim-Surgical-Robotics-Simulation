"""
Tests for environment API and wrappers.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from src.envs.unity_env import UnityEnv, MockUnityEnv
from src.envs.wrappers import FrameStackWrapper, RewardShapingWrapper, DomainRandomizationWrapper


class TestMockUnityEnv:
    """Test MockUnityEnv functionality."""
    
    def test_env_creation(self):
        """Test environment creation."""
        env = MockUnityEnv()
        assert env is not None
        assert env.action_space.shape == (5,)
        assert 'image' in env.observation_space.spaces
        assert 'aux' in env.observation_space.spaces
    
    def test_reset(self):
        """Test environment reset."""
        env = MockUnityEnv()
        obs, info = env.reset()
        
        assert isinstance(obs, dict)
        assert 'image' in obs
        assert 'aux' in obs
        assert obs['image'].shape == (3, 84, 84)
        assert obs['aux'].shape == (4,)
        assert isinstance(info, dict)
    
    def test_step(self):
        """Test environment step."""
        env = MockUnityEnv()
        obs, info = env.reset()
        
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(next_obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_render(self):
        """Test environment render."""
        env = MockUnityEnv()
        obs, info = env.reset()
        
        image = env.render()
        assert image is not None
        assert image.shape == (84, 84, 3)
    
    def test_close(self):
        """Test environment close."""
        env = MockUnityEnv()
        env.close()
        assert not env.connected


class TestFrameStackWrapper:
    """Test FrameStackWrapper functionality."""
    
    def test_wrapper_creation(self):
        """Test wrapper creation."""
        env = MockUnityEnv()
        wrapped_env = FrameStackWrapper(env, num_frames=4)
        
        assert wrapped_env.num_frames == 4
        assert wrapped_env.observation_space['image'].shape == (12, 84, 84)  # 4 * 3 channels
    
    def test_reset_with_wrapper(self):
        """Test reset with frame stacking."""
        env = MockUnityEnv()
        wrapped_env = FrameStackWrapper(env, num_frames=4)
        
        obs, info = wrapped_env.reset()
        assert obs['image'].shape == (12, 84, 84)
        assert len(wrapped_env.frames) == 4
    
    def test_step_with_wrapper(self):
        """Test step with frame stacking."""
        env = MockUnityEnv()
        wrapped_env = FrameStackWrapper(env, num_frames=4)
        
        obs, info = wrapped_env.reset()
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        assert next_obs['image'].shape == (12, 84, 84)
        assert len(wrapped_env.frames) == 4


class TestRewardShapingWrapper:
    """Test RewardShapingWrapper functionality."""
    
    def test_wrapper_creation(self):
        """Test wrapper creation."""
        env = MockUnityEnv()
        config = {
            'reward_weights': {
                'alpha': 1.0,
                'beta': 0.5,
                'lambda_time': -0.01,
                'lambda_action': -0.001,
                'lambda_collision': -1.0,
                'lambda_safety': -2.0
            }
        }
        wrapped_env = RewardShapingWrapper(env, config)
        
        assert wrapped_env.alpha == 1.0
        assert wrapped_env.beta == 0.5
    
    def test_reward_shaping(self):
        """Test reward shaping."""
        env = MockUnityEnv()
        config = {
            'reward_weights': {
                'alpha': 1.0,
                'beta': 0.5,
                'lambda_time': -0.01,
                'lambda_action': -0.001,
                'lambda_collision': -1.0,
                'lambda_safety': -2.0
            }
        }
        wrapped_env = RewardShapingWrapper(env, config)
        
        obs, info = wrapped_env.reset()
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        assert isinstance(reward, float)
        assert 'shaped_reward' in info


class TestDomainRandomizationWrapper:
    """Test DomainRandomizationWrapper functionality."""
    
    def test_wrapper_creation(self):
        """Test wrapper creation."""
        env = MockUnityEnv()
        config = {
            'domain_randomization': {
                'lighting': {
                    'intensity_range': [0.7, 1.3],
                    'color_temp_range': [3000, 7000]
                },
                'camera': {
                    'noise_std': 0.01
                }
            }
        }
        wrapped_env = DomainRandomizationWrapper(env, config)
        
        assert wrapped_env.randomization_config is not None
    
    def test_randomization_application(self):
        """Test domain randomization application."""
        env = MockUnityEnv()
        config = {
            'domain_randomization': {
                'lighting': {
                    'intensity_range': [0.7, 1.3],
                    'color_temp_range': [3000, 7000]
                }
            }
        }
        wrapped_env = DomainRandomizationWrapper(env, config)
        
        obs, info = wrapped_env.reset()
        assert wrapped_env.current_lighting is not None
        assert 'intensity' in wrapped_env.current_lighting
        assert 'color_temp' in wrapped_env.current_lighting


class TestEnvironmentIntegration:
    """Test environment integration."""
    
    def test_full_wrapper_chain(self):
        """Test full wrapper chain."""
        env = MockUnityEnv()
        
        # Apply all wrappers
        env = FrameStackWrapper(env, num_frames=4)
        env = RewardShapingWrapper(env, {'reward_weights': {}})
        env = DomainRandomizationWrapper(env, {'domain_randomization': {}})
        
        # Test reset
        obs, info = env.reset()
        assert obs['image'].shape == (12, 84, 84)
        assert 'aux' in obs
        
        # Test step
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert next_obs['image'].shape == (12, 84, 84)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_episode_completion(self):
        """Test episode completion."""
        env = MockUnityEnv()
        env = FrameStackWrapper(env, num_frames=4)
        
        obs, info = env.reset()
        episode_length = 0
        
        for _ in range(100):  # Max steps
            action = np.array([0.1, 0.1, 0.1, 0.1, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            if terminated or truncated:
                break
        
        assert episode_length > 0
        assert episode_length <= 100


@pytest.mark.parametrize("num_frames", [2, 4, 8])
def test_frame_stack_variations(num_frames):
    """Test different frame stack sizes."""
    env = MockUnityEnv()
    wrapped_env = FrameStackWrapper(env, num_frames=num_frames)
    
    obs, info = wrapped_env.reset()
    expected_channels = num_frames * 3
    assert obs['image'].shape == (expected_channels, 84, 84)


@pytest.mark.parametrize("action", [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
])
def test_action_variations(action):
    """Test different action types."""
    env = MockUnityEnv()
    obs, info = env.reset()
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
