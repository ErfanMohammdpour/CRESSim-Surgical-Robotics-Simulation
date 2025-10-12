"""
Tests for reward calculation and shaping.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.envs.wrappers import RewardShapingWrapper
from src.rl.safety_shield import SafetyAwareReward


class TestRewardShaping:
    """Test reward shaping functionality."""
    
    def test_liquid_reduction_reward(self):
        """Test liquid mass reduction reward calculation."""
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
        
        # Create mock environment
        env = Mock()
        env.observation_space = Mock()
        env.action_space = Mock()
        
        wrapper = RewardShapingWrapper(env, config)
        
        # Test liquid reduction
        obs = {'aux': np.array([0.0, 1.0, 0.5, 0.0])}  # [suction_state, liquid_mass, contaminant_mass, collisions]
        next_obs = {'aux': np.array([0.0, 0.8, 0.5, 0.0])}  # Liquid reduced by 0.2
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
        info = {}
        
        wrapper.prev_liquid_mass = 1.0
        wrapper.prev_contaminant_mass = 0.5
        wrapper.prev_action = None
        
        reward = wrapper._calculate_shaped_reward(action, info)
        
        # Should have positive reward for liquid reduction
        assert reward > 0
        assert abs(reward - 0.2) < 0.01  # 1.0 * 0.2 liquid reduction
    
    def test_contaminant_reduction_reward(self):
        """Test contaminant mass reduction reward calculation."""
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
        
        env = Mock()
        env.observation_space = Mock()
        env.action_space = Mock()
        
        wrapper = RewardShapingWrapper(env, config)
        
        # Test contaminant reduction
        obs = {'aux': np.array([0.0, 1.0, 0.5, 0.0])}
        next_obs = {'aux': np.array([0.0, 1.0, 0.3, 0.0])}  # Contaminant reduced by 0.2
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
        info = {}
        
        wrapper.prev_liquid_mass = 1.0
        wrapper.prev_contaminant_mass = 0.5
        wrapper.prev_action = None
        
        reward = wrapper._calculate_shaped_reward(action, info)
        
        # Should have positive reward for contaminant reduction
        assert reward > 0
        assert abs(reward - 0.1) < 0.01  # 0.5 * 0.2 contaminant reduction
    
    def test_collision_penalty(self):
        """Test collision penalty."""
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
        
        env = Mock()
        env.observation_space = Mock()
        env.action_space = Mock()
        
        wrapper = RewardShapingWrapper(env, config)
        
        # Test collision penalty
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
        info = {'collisions': 1}
        
        wrapper.prev_liquid_mass = 1.0
        wrapper.prev_contaminant_mass = 0.5
        wrapper.prev_action = None
        
        reward = wrapper._calculate_shaped_reward(action, info)
        
        # Should have negative reward due to collision
        assert reward < 0
        assert abs(reward + 1.0) < 0.01  # -1.0 collision penalty
    
    def test_action_smoothness_penalty(self):
        """Test action smoothness penalty."""
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
        
        env = Mock()
        env.observation_space = Mock()
        env.action_space = Mock()
        
        wrapper = RewardShapingWrapper(env, config)
        
        # Test action smoothness penalty
        prev_action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
        action = np.array([0.5, 0.5, 0.5, 0.5, 0.0])  # Large change from previous
        info = {}
        
        wrapper.prev_liquid_mass = 1.0
        wrapper.prev_contaminant_mass = 0.5
        wrapper.prev_action = prev_action
        
        reward = wrapper._calculate_shaped_reward(action, info)
        
        # Should have additional penalty for large action change
        action_diff = np.linalg.norm(action - prev_action)
        expected_penalty = -0.001 * action_diff
        assert reward < expected_penalty


class TestSafetyAwareReward:
    """Test safety-aware reward calculation."""
    
    def test_safety_aware_reward_creation(self):
        """Test safety-aware reward creation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_time': -0.01,
            'lambda_action': -0.001,
            'lambda_collision': -1.0,
            'lambda_safety': -2.0
        }
        
        reward_calc = SafetyAwareReward(config)
        
        assert reward_calc.alpha == 1.0
        assert reward_calc.beta == 0.5
        assert reward_calc.lambda_safety == -2.0
    
    def test_liquid_reduction_calculation(self):
        """Test liquid reduction calculation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_time': -0.01,
            'lambda_action': -0.001,
            'lambda_collision': -1.0,
            'lambda_safety': -2.0
        }
        
        reward_calc = SafetyAwareReward(config)
        
        obs = {'aux': np.array([0.0, 1.0, 0.5, 0.0])}
        next_obs = {'aux': np.array([0.0, 0.8, 0.5, 0.0])}
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
        info = {}
        
        liquid_reward = reward_calc._calculate_liquid_reward(obs, next_obs)
        
        assert liquid_reward == 0.2  # 1.0 - 0.8
    
    def test_contaminant_reduction_calculation(self):
        """Test contaminant reduction calculation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_time': -0.01,
            'lambda_action': -0.001,
            'lambda_collision': -1.0,
            'lambda_safety': -2.0
        }
        
        reward_calc = SafetyAwareReward(config)
        
        obs = {'aux': np.array([0.0, 1.0, 0.5, 0.0])}
        next_obs = {'aux': np.array([0.0, 1.0, 0.3, 0.0])}
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
        info = {}
        
        contaminant_reward = reward_calc._calculate_contaminant_reward(obs, next_obs)
        
        assert contaminant_reward == 0.2  # 0.5 - 0.3
    
    def test_action_penalty_calculation(self):
        """Test action penalty calculation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_time': -0.01,
            'lambda_action': -0.001,
            'lambda_collision': -1.0,
            'lambda_safety': -2.0
        }
        
        reward_calc = SafetyAwareReward(config)
        
        action = np.array([0.1, 0.2, 0.3, 0.4, 1.0])
        action_penalty = reward_calc._calculate_action_penalty(action)
        
        action_magnitude = np.linalg.norm(action)
        expected_penalty = -0.001 * action_magnitude
        
        assert abs(action_penalty - expected_penalty) < 1e-6
    
    def test_collision_penalty_calculation(self):
        """Test collision penalty calculation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_time': -0.01,
            'lambda_action': -0.001,
            'lambda_collision': -1.0,
            'lambda_safety': -2.0
        }
        
        reward_calc = SafetyAwareReward(config)
        
        info = {'collisions': 2}
        collision_penalty = reward_calc._calculate_collision_penalty(info)
        
        assert collision_penalty == -2.0  # -1.0 * 2 collisions
    
    def test_safety_reward_calculation(self):
        """Test safety reward calculation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_time': -0.01,
            'lambda_action': -0.001,
            'lambda_collision': -1.0,
            'lambda_safety': -2.0,
            'safety_reward_weight': 0.1,
            'proximity_reward_weight': 0.05,
            'smooth_action_reward_weight': 0.02
        }
        
        reward_calc = SafetyAwareReward(config)
        
        # Test safe state
        safety_info = {
            'safety_level': 'safe',
            'distance': 0.3,
            'action_scale': 1.0
        }
        
        safety_reward = reward_calc._calculate_safety_reward(safety_info)
        
        assert safety_reward > 0  # Should be positive for safe state
        
        # Test warning state
        safety_info = {
            'safety_level': 'warning',
            'distance': 0.08,
            'action_scale': 0.5
        }
        
        safety_reward = reward_calc._calculate_safety_reward(safety_info)
        
        assert safety_reward < 0  # Should be negative for warning state
    
    def test_complete_reward_calculation(self):
        """Test complete reward calculation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_time': -0.01,
            'lambda_action': -0.001,
            'lambda_collision': -1.0,
            'lambda_safety': -2.0
        }
        
        reward_calc = SafetyAwareReward(config)
        
        obs = {'aux': np.array([0.0, 1.0, 0.5, 0.0])}
        next_obs = {'aux': np.array([0.0, 0.8, 0.3, 0.0])}
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
        info = {'collisions': 0}
        safety_info = {
            'safety_level': 'safe',
            'distance': 0.3,
            'action_scale': 1.0
        }
        
        total_reward = reward_calc.calculate_reward(obs, action, next_obs, info, safety_info)
        
        # Calculate expected components
        liquid_reward = 1.0 * 0.2  # alpha * liquid_reduction
        contaminant_reward = 0.5 * 0.2  # beta * contaminant_reduction
        time_penalty = -0.01
        action_penalty = -0.001 * np.linalg.norm(action)
        collision_penalty = 0  # no collisions
        safety_reward = 0.1  # safe state
        
        expected_reward = liquid_reward + contaminant_reward + time_penalty + action_penalty + collision_penalty + safety_reward
        
        assert abs(total_reward - expected_reward) < 1e-6


@pytest.mark.parametrize("liquid_reduction,expected_reward", [
    (0.0, 0.0),
    (0.1, 0.1),
    (0.5, 0.5),
    (1.0, 1.0)
])
def test_liquid_reward_variations(liquid_reduction, expected_reward):
    """Test liquid reward with different reduction amounts."""
    config = {'alpha': 1.0, 'beta': 0.5, 'lambda_time': -0.01, 'lambda_action': -0.001, 'lambda_collision': -1.0, 'lambda_safety': -2.0}
    reward_calc = SafetyAwareReward(config)
    
    obs = {'aux': np.array([0.0, 1.0, 0.5, 0.0])}
    next_obs = {'aux': np.array([0.0, 1.0 - liquid_reduction, 0.5, 0.0])}
    action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
    info = {}
    safety_info = {'safety_level': 'safe', 'distance': 0.3, 'action_scale': 1.0}
    
    total_reward = reward_calc.calculate_reward(obs, action, next_obs, info, safety_info)
    
    # Only liquid reward should change
    assert total_reward > 0
    assert abs(total_reward - expected_reward) < 0.1  # Allow for other components


@pytest.mark.parametrize("safety_level,expected_sign", [
    ('safe', 1),
    ('warning', -1),
    ('critical', -1),
    ('emergency_stop', -1)
])
def test_safety_level_impact(safety_level, expected_sign):
    """Test impact of different safety levels."""
    config = {
        'alpha': 1.0, 'beta': 0.5, 'lambda_time': -0.01, 'lambda_action': -0.001, 
        'lambda_collision': -1.0, 'lambda_safety': -2.0,
        'safety_reward_weight': 0.1
    }
    reward_calc = SafetyAwareReward(config)
    
    obs = {'aux': np.array([0.0, 1.0, 0.5, 0.0])}
    next_obs = {'aux': np.array([0.0, 0.9, 0.5, 0.0])}
    action = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
    info = {}
    safety_info = {'safety_level': safety_level, 'distance': 0.3, 'action_scale': 1.0}
    
    total_reward = reward_calc.calculate_reward(obs, action, next_obs, info, safety_info)
    
    if expected_sign > 0:
        assert total_reward > 0
    else:
        assert total_reward < 0
