"""
Tests for reward calculation and safety systems.
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
from src.rl.safety_shield import RLSafetyShield, SafetyAwareReward, SafetyCurriculum
from src.vision.safety_seg import SafetySegmentation, SafetyShield


class TestRewardCalculation:
    """Test reward calculation in mock environment."""
    
    def test_reward_components(self):
        """Test individual reward components."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        obs, info = env.reset(seed=42)
        
        # Test liquid reduction reward
        initial_liquid = env.initial_liquid_mass
        env.liquid_mass = initial_liquid * 0.5  # 50% reduction
        
        reward = env._calculate_reward()
        
        # Should be positive due to liquid reduction
        assert reward > 0
        
        # Test contaminant reduction reward
        initial_contaminant = env.initial_contaminant_mass
        env.contaminant_mass = initial_contaminant * 0.3  # 70% reduction
        
        reward = env._calculate_reward()
        
        # Should be even more positive
        assert reward > 0
    
    def test_penalty_components(self):
        """Test penalty components."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        obs, info = env.reset(seed=42)
        
        # Test collision penalty
        env.collision_count = 5
        reward = env._calculate_reward()
        
        # Should be negative due to collisions
        assert reward < 0
        
        # Test safety violation penalty
        env.safety_violations = 3
        reward = env._calculate_reward()
        
        # Should be even more negative
        assert reward < 0
    
    def test_time_penalty(self):
        """Test time penalty."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        obs, info = env.reset(seed=42)
        
        # Test time penalty increases with steps
        env.step_count = 50
        reward1 = env._calculate_reward()
        
        env.step_count = 100
        reward2 = env._calculate_reward()
        
        # Later reward should be lower due to time penalty
        assert reward2 < reward1
    
    def test_action_penalty(self):
        """Test action smoothness penalty."""
        env = MockSuctionEnv(image_size=(64, 64), max_steps=100)
        
        obs, info = env.reset(seed=42)
        
        # Test action penalty
        env.position = np.array([1.0, 1.0, 1.0])  # Large position
        reward = env._calculate_reward()
        
        # Should be negative due to action penalty
        assert reward < 0
    
    def test_reward_weights(self):
        """Test reward weight configuration."""
        custom_weights = {
            "alpha": 5.0,  # Higher liquid reward
            "beta": 2.0,   # Higher contaminant reward
            "lambda_time": -0.01,  # Higher time penalty
            "lambda_action": -0.001,  # Higher action penalty
            "lambda_collision": -10.0,  # Higher collision penalty
            "lambda_safety": -20.0  # Higher safety penalty
        }
        
        env = MockSuctionEnv(
            image_size=(64, 64),
            max_steps=100,
            reward_weights=custom_weights
        )
        
        obs, info = env.reset(seed=42)
        
        # Test liquid reduction with higher weight
        env.liquid_mass = env.initial_liquid_mass * 0.5
        reward = env._calculate_reward()
        
        # Should be more positive due to higher alpha
        assert reward > 0
        
        # Test collision with higher penalty
        env.collision_count = 1
        reward = env._calculate_reward()
        
        # Should be more negative due to higher lambda_collision
        assert reward < 0


class TestSafetyAwareReward:
    """Test safety-aware reward calculation."""
    
    def test_safety_aware_reward_creation(self):
        """Test safety-aware reward creation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_safety': -2.0,
            'safety_reward_weight': 0.1
        }
        
        reward_calc = SafetyAwareReward(config)
        
        assert reward_calc.alpha == 1.0
        assert reward_calc.beta == 0.5
        assert reward_calc.lambda_safety == -2.0
        assert reward_calc.safety_reward_weight == 0.1
    
    def test_safety_reward_calculation(self):
        """Test safety reward calculation."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_safety': -2.0,
            'safety_reward_weight': 0.1,
            'proximity_reward_weight': 0.05
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
        
        # Test critical state
        safety_info = {
            'safety_level': 'critical',
            'distance': 0.01,
            'action_scale': 0.1
        }
        
        safety_reward = reward_calc._calculate_safety_reward(safety_info)
        
        assert safety_reward < 0  # Should be negative for critical state
    
    def test_full_reward_calculation(self):
        """Test full reward calculation with safety."""
        config = {
            'alpha': 1.0,
            'beta': 0.5,
            'lambda_time': 0.01,
            'lambda_action': 1e-3,
            'lambda_collision': 5.0,
            'lambda_safety': 10.0,
            'safety_reward_weight': 0.1,
            'proximity_reward_weight': 0.05
        }
        
        reward_calc = SafetyAwareReward(config)
        
        # Test safe scenario
        safety_info = {
            'safety_level': 'safe',
            'distance': 0.3,
            'action_scale': 1.0
        }
        
        reward = reward_calc.calculate_reward(
            liquid_reduction=0.5,
            contaminant_reduction=0.3,
            step_count=10,
            action_norm=0.1,
            collision=False,
            safety_info=safety_info
        )
        
        assert isinstance(reward, float)
        assert reward > 0  # Should be positive for good performance
        
        # Test unsafe scenario
        safety_info = {
            'safety_level': 'critical',
            'distance': 0.01,
            'action_scale': 0.1
        }
        
        reward = reward_calc.calculate_reward(
            liquid_reduction=0.5,
            contaminant_reduction=0.3,
            step_count=10,
            action_norm=0.1,
            collision=False,
            safety_info=safety_info
        )
        
        assert reward < 0  # Should be negative due to safety violation


class TestSafetyCurriculum:
    """Test safety curriculum learning."""
    
    def test_curriculum_creation(self):
        """Test curriculum creation."""
        config = {
            'min_difficulty': 0.3,
            'max_difficulty': 1.0,
            'success_threshold': 0.7,
            'difficulty_increase': 0.1
        }
        
        curriculum = SafetyCurriculum(config)
        
        assert curriculum.min_difficulty == 0.3
        assert curriculum.max_difficulty == 1.0
        assert curriculum.success_threshold == 0.7
        assert curriculum.difficulty_increase == 0.1
        assert curriculum.current_difficulty == 0.3
    
    def test_curriculum_update(self):
        """Test curriculum update."""
        config = {
            'min_difficulty': 0.3,
            'max_difficulty': 1.0,
            'success_threshold': 0.7,
            'difficulty_increase': 0.1,
            'evaluation_episodes': 5
        }
        
        curriculum = SafetyCurriculum(config)
        
        # Add successful episodes
        for _ in range(5):
            curriculum.update(True)
        
        # Should increase difficulty
        assert curriculum.current_difficulty > curriculum.min_difficulty
    
    def test_difficulty_params(self):
        """Test difficulty parameter generation."""
        config = {
            'min_difficulty': 0.3,
            'max_difficulty': 1.0,
            'success_threshold': 0.7,
            'difficulty_increase': 0.1
        }
        
        curriculum = SafetyCurriculum(config)
        curriculum.current_difficulty = 0.5
        
        params = curriculum.get_difficulty_params()
        
        assert 'liquid_mass' in params
        assert 'contaminant_mass' in params
        assert 'noise_level' in params
        assert 'time_limit' in params
        assert 'precision_required' in params
        
        # Check that parameters scale with difficulty
        assert params['liquid_mass'] > 0.5
        assert params['contaminant_mass'] > 0.3
        assert params['noise_level'] < 0.5
    
    def test_curriculum_boundaries(self):
        """Test curriculum difficulty boundaries."""
        config = {
            'min_difficulty': 0.3,
            'max_difficulty': 1.0,
            'success_threshold': 0.7,
            'difficulty_increase': 0.1,
            'evaluation_episodes': 5
        }
        
        curriculum = SafetyCurriculum(config)
        
        # Test minimum difficulty
        curriculum.current_difficulty = 0.3
        curriculum.update(False)  # Failed episode
        assert curriculum.current_difficulty >= 0.3
        
        # Test maximum difficulty
        curriculum.current_difficulty = 1.0
        curriculum.update(True)  # Successful episode
        assert curriculum.current_difficulty <= 1.0


class TestSafetySegmentation:
    """Test safety segmentation network."""
    
    def test_safety_network_creation(self):
        """Test safety network creation."""
        network = SafetySegmentation(
            input_size=(64, 64),
            num_classes=3,
            base_channels=32
        )
        
        assert network is not None
        assert network.num_classes == 3
        assert network.input_size == (64, 64)
    
    def test_forward_pass(self):
        """Test forward pass of safety network."""
        network = SafetySegmentation(
            input_size=(64, 64),
            num_classes=3,
            base_channels=32
        )
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 64, 64)
        
        # Forward pass
        segmentation, distance = network(input_tensor)
        
        assert segmentation.shape == (batch_size, 3, 64, 64)
        assert distance.shape == (batch_size, 1)
        assert torch.all(distance >= 0) and torch.all(distance <= 1)
    
    def test_predict_safety(self):
        """Test safety prediction."""
        network = SafetySegmentation(
            input_size=(64, 64),
            num_classes=3,
            base_channels=32
        )
        
        # Create dummy input
        input_tensor = torch.randn(1, 3, 64, 64)
        
        # Predict safety
        safety_info = network.predict_safety(input_tensor)
        
        assert 'segmentation' in safety_info
        assert 'distance' in safety_info
        assert 'dangerous_mask' in safety_info
        assert 'dangerous_ratio' in safety_info
        assert 'is_safe' in safety_info
        
        assert safety_info['segmentation'].shape == (1, 3, 64, 64)
        assert safety_info['distance'].shape == (1, 1)
        assert safety_info['dangerous_mask'].shape == (1, 64, 64)
        assert safety_info['dangerous_ratio'].shape == (1,)
        assert safety_info['is_safe'].shape == (1,)


class TestSafetyShield:
    """Test safety shield functionality."""
    
    def test_safety_shield_creation(self):
        """Test safety shield creation."""
        # Create mock safety network
        safety_net = Mock()
        safety_net.predict_safety.return_value = {
            'distance': torch.tensor([0.1]),
            'dangerous_ratio': torch.tensor([0.05]),
            'is_safe': torch.tensor([True])
        }
        
        shield = SafetyShield(
            safety_net=safety_net,
            d_safe=0.05,
            d_warning=0.1,
            scaling_factor=0.5,
            max_scaling=0.1
        )
        
        assert shield.d_safe == 0.05
        assert shield.d_warning == 0.1
        assert shield.scaling_factor == 0.5
        assert shield.max_scaling == 0.1
    
    def test_safe_action_projection(self):
        """Test action projection for safe state."""
        # Create mock safety network
        safety_net = Mock()
        safety_net.predict_safety.return_value = {
            'distance': torch.tensor([0.2]),
            'dangerous_ratio': torch.tensor([0.05]),
            'is_safe': torch.tensor([True])
        }
        
        shield = SafetyShield(
            safety_net=safety_net,
            d_safe=0.05,
            d_warning=0.1,
            scaling_factor=0.5,
            max_scaling=0.1
        )
        
        # Test safe action
        image = torch.randn(1, 3, 64, 64)
        action = torch.tensor([[0.1, 0.1, 0.1, 0.1, 1.0]])
        
        projected_action, safety_info = shield(image, action)
        
        # Safe action should not be modified
        assert torch.allclose(projected_action, action)
        assert safety_info['safety_level'] == 'safe'
    
    def test_warning_action_projection(self):
        """Test action projection for warning state."""
        # Create mock safety network
        safety_net = Mock()
        safety_net.predict_safety.return_value = {
            'distance': torch.tensor([0.08]),
            'dangerous_ratio': torch.tensor([0.15]),
            'is_safe': torch.tensor([False])
        }
        
        shield = SafetyShield(
            safety_net=safety_net,
            d_safe=0.05,
            d_warning=0.1,
            scaling_factor=0.5,
            max_scaling=0.1
        )
        
        # Test warning action
        image = torch.randn(1, 3, 64, 64)
        action = torch.tensor([[0.1, 0.1, 0.1, 0.1, 1.0]])
        
        projected_action, safety_info = shield(image, action)
        
        # Warning action should be scaled down
        assert torch.allclose(projected_action, action * 0.5)
        assert safety_info['safety_level'] == 'warning'
    
    def test_critical_action_projection(self):
        """Test action projection for critical state."""
        # Create mock safety network
        safety_net = Mock()
        safety_net.predict_safety.return_value = {
            'distance': torch.tensor([0.02]),
            'dangerous_ratio': torch.tensor([0.25]),
            'is_safe': torch.tensor([False])
        }
        
        shield = SafetyShield(
            safety_net=safety_net,
            d_safe=0.05,
            d_warning=0.1,
            scaling_factor=0.5,
            max_scaling=0.1
        )
        
        # Test critical action
        image = torch.randn(1, 3, 64, 64)
        action = torch.tensor([[0.1, 0.1, 0.1, 0.1, 1.0]])
        
        projected_action, safety_info = shield(image, action)
        
        # Critical action should be zeroed
        assert torch.allclose(projected_action, torch.zeros_like(action))
        assert safety_info['safety_level'] == 'critical'
    
    def test_violation_tracking(self):
        """Test violation tracking."""
        # Create mock safety network
        safety_net = Mock()
        safety_net.predict_safety.return_value = {
            'distance': torch.tensor([0.02]),
            'dangerous_ratio': torch.tensor([0.25]),
            'is_safe': torch.tensor([False])
        }
        
        shield = SafetyShield(
            safety_net=safety_net,
            d_safe=0.05,
            d_warning=0.1,
            scaling_factor=0.5,
            max_scaling=0.1
        )
        
        # Test multiple violations
        image = torch.randn(1, 3, 64, 64)
        action = torch.tensor([[0.1, 0.1, 0.1, 0.1, 1.0]])
        
        for _ in range(3):
            projected_action, safety_info = shield(image, action)
        
        assert shield.violation_count == 3
        assert safety_info['violation_count'] == 3


@pytest.mark.parametrize("distance,expected_level", [
    (0.2, "safe"),
    (0.08, "warning"),
    (0.03, "critical")
])
def test_safety_level_determination(distance, expected_level):
    """Test safety level determination based on distance."""
    safety_net = Mock()
    safety_net.predict_safety.return_value = {
        'distance': torch.tensor([distance]),
        'dangerous_ratio': torch.tensor([0.05]),
        'is_safe': torch.tensor([distance > 0.1])
    }
    
    shield = SafetyShield(
        safety_net=safety_net,
        d_safe=0.05,
        d_warning=0.1,
        scaling_factor=0.5,
        max_scaling=0.1
    )
    
    image = torch.randn(1, 3, 64, 64)
    action = torch.tensor([[0.1, 0.1, 0.1, 0.1, 1.0]])
    
    projected_action, safety_info = shield(image, action)
    
    assert safety_info['safety_level'] == expected_level


if __name__ == "__main__":
    pytest.main([__file__])