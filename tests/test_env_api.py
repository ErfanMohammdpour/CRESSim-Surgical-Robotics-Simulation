"""
Tests for environment API and functionality.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.envs.mock_env import MockSuctionEnv
from src.envs.unity_env import UnityEnvWrapper, CurriculumEnv, create_environment


class TestMockSuctionEnv:
    """Test mock suction environment."""
    
    def test_env_creation(self):
        """Test environment creation."""
        env = MockSuctionEnv(image_size=(128, 128), max_steps=1000)
        
        assert env.image_size == (128, 128)
        assert env.max_steps == 1000
        assert env.action_space.shape == (5,)
        assert env.observation_space.shape == (128, 128, 3)
    
    def test_env_reset(self):
        """Test environment reset."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        assert 'step_count' in info
        assert 'position' in info
        assert 'liquid_mass' in info
    
    def test_env_step(self):
        """Test environment step."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        obs, info = env.reset(seed=42)
        
        # Test valid action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        assert next_obs.shape == (64, 64, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(next_info, dict)
    
    def test_env_determinism(self):
        """Test environment determinism with fixed seed."""
        env1 = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        env2 = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        # Reset with same seed
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)
        
        # Should be identical
        assert np.array_equal(obs1, obs2)
        assert info1['step_count'] == info2['step_count']
        assert np.array_equal(info1['position'], info2['position'])
    
    def test_env_termination(self):
        """Test environment termination conditions."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=10)
        
        obs, info = env.reset(seed=42)
        
        # Run until termination
        for step in range(15):  # More than max_steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        assert step < 15  # Should terminate before max steps
        assert terminated or truncated
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        obs, info = env.reset(seed=42)
        
        # Test reward calculation
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])  # Suction on
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        assert reward >= -10.0  # Should not be extremely negative
        assert reward <= 10.0   # Should not be extremely positive
    
    def test_info_consistency(self):
        """Test info dictionary consistency."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        obs, info = env.reset(seed=42)
        
        # Check initial info
        assert 'step_count' in info
        assert 'position' in info
        assert 'orientation' in info
        assert 'suction_active' in info
        assert 'liquid_mass' in info
        assert 'contaminant_mass' in info
        assert 'collision_count' in info
        assert 'safety_violations' in info
        assert 'liquid_reduction' in info
        assert 'contaminant_reduction' in info
        
        # Check types
        assert isinstance(info['step_count'], int)
        assert isinstance(info['position'], np.ndarray)
        assert isinstance(info['orientation'], float)
        assert isinstance(info['suction_active'], bool)
        assert isinstance(info['liquid_mass'], float)
        assert isinstance(info['contaminant_mass'], float)
        assert isinstance(info['collision_count'], int)
        assert isinstance(info['safety_violations'], int)
        assert isinstance(info['liquid_reduction'], float)
        assert isinstance(info['contaminant_reduction'], float)


class TestUnityEnvWrapper:
    """Test Unity environment wrapper."""
    
    def test_wrapper_creation(self):
        """Test wrapper creation."""
        build_path = Path("test_build.x86_64")
        
        with patch.object(Path, 'exists', return_value=True):
            wrapper = UnityEnvWrapper(
                unity_build_path=build_path,
                port=5005,
                image_size=(128, 128),
                max_steps=1000
            )
            
            assert wrapper.unity_build_path == build_path
            assert wrapper.port == 5005
            assert wrapper.image_size == (128, 128)
            assert wrapper.max_steps == 1000
            assert wrapper.action_space.shape == (5,)
            assert wrapper.observation_space.shape == (128, 128, 3)
    
    def test_wrapper_missing_build(self):
        """Test wrapper with missing build."""
        build_path = Path("nonexistent_build.x86_64")
        
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                UnityEnvWrapper(
                    unity_build_path=build_path,
                    port=5005,
                    image_size=(128, 128),
                    max_steps=1000
                )


class TestCurriculumEnv:
    """Test curriculum environment."""
    
    def test_curriculum_creation(self):
        """Test curriculum environment creation."""
        base_env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        curriculum_config = {
            'min_difficulty': 0.3,
            'max_difficulty': 1.0,
            'success_threshold': 0.7,
            'difficulty_increase': 0.1,
            'evaluation_episodes': 10
        }
        
        env = CurriculumEnv(base_env, curriculum_config)
        
        assert env.min_difficulty == 0.3
        assert env.max_difficulty == 1.0
        assert env.success_threshold == 0.7
        assert env.difficulty_increase == 0.1
        assert env.current_difficulty == 0.3
    
    def test_curriculum_reset(self):
        """Test curriculum environment reset."""
        base_env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        curriculum_config = {
            'min_difficulty': 0.3,
            'max_difficulty': 1.0,
            'success_threshold': 0.7,
            'difficulty_increase': 0.1,
            'evaluation_episodes': 10
        }
        
        env = CurriculumEnv(base_env, curriculum_config)
        
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (64, 64, 3)
        assert 'current_difficulty' in info
        assert 'liquid_mass' in info
        assert 'contaminant_mass' in info
        assert 'noise_level' in info
        assert 'time_limit' in info
        assert 'precision_required' in info
    
    def test_curriculum_update(self):
        """Test curriculum update."""
        base_env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        curriculum_config = {
            'min_difficulty': 0.3,
            'max_difficulty': 1.0,
            'success_threshold': 0.7,
            'difficulty_increase': 0.1,
            'evaluation_episodes': 5
        }
        
        env = CurriculumEnv(base_env, curriculum_config)
        
        # Add successful episodes
        for _ in range(5):
            env.update_episode_result(True)
        
        # Should increase difficulty
        assert env.current_difficulty > env.min_difficulty
    
    def test_difficulty_params(self):
        """Test difficulty parameter generation."""
        base_env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        curriculum_config = {
            'min_difficulty': 0.3,
            'max_difficulty': 1.0,
            'success_threshold': 0.7,
            'difficulty_increase': 0.1,
            'evaluation_episodes': 10
        }
        
        env = CurriculumEnv(base_env, curriculum_config)
        env.current_difficulty = 0.5
        
        params = env._get_curriculum_params()
        
        assert 'liquid_mass' in params
        assert 'contaminant_mass' in params
        assert 'noise_level' in params
        assert 'time_limit' in params
        assert 'precision_required' in params
        
        # Check that parameters scale with difficulty
        assert params['liquid_mass'] > 0.5
        assert params['contaminant_mass'] > 0.3
        assert params['noise_level'] < 0.5


class TestCreateEnvironment:
    """Test environment creation function."""
    
    def test_create_mock_environment(self):
        """Test creating mock environment."""
        config = {
            'image_size': [64, 64],
            'max_episode_steps': 100
        }
        
        env = create_environment(config, mock=True)
        
        assert isinstance(env, MockSuctionEnv)
        assert env.image_size == (64, 64)
        assert env.max_steps == 100
    
    def test_create_environment_with_curriculum(self):
        """Test creating environment with curriculum."""
        config = {
            'image_size': [64, 64],
            'max_episode_steps': 100,
            'curriculum': {
                'enabled': True,
                'min_difficulty': 0.3,
                'max_difficulty': 1.0,
                'success_threshold': 0.7,
                'difficulty_increase': 0.1,
                'evaluation_episodes': 10
            }
        }
        
        env = create_environment(config, mock=True)
        
        assert isinstance(env, CurriculumEnv)
        assert env.min_difficulty == 0.3
        assert env.max_difficulty == 1.0
    
    def test_create_environment_without_curriculum(self):
        """Test creating environment without curriculum."""
        config = {
            'image_size': [64, 64],
            'max_episode_steps': 100,
            'curriculum': {
                'enabled': False
            }
        }
        
        env = create_environment(config, mock=True)
        
        assert isinstance(env, MockSuctionEnv)
        assert not isinstance(env, CurriculumEnv)


@pytest.mark.parametrize("image_size,max_steps", [
    ((64, 64), 100),
    ((128, 128), 500),
    ((256, 256), 1000)
])
def test_env_different_sizes(image_size, max_steps):
    """Test environment with different image sizes and max steps."""
    env = MockSuctionEnv(image_size=image_size, max_steps=max_steps)
    
    assert env.image_size == image_size
    assert env.max_steps == max_steps
    assert env.observation_space.shape == (*image_size, 3)
    
    obs, info = env.reset(seed=42)
    assert obs.shape == (*image_size, 3)
    
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    assert next_obs.shape == (*image_size, 3)


if __name__ == "__main__":
    pytest.main([__file__])