"""
RL safety shield integration.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from ..vision.safety_seg import SafetySegmentation, SafetyShield

logger = logging.getLogger(__name__)


class RLSafetyShield:
    """Safety shield for RL training."""
    
    def __init__(
        self,
        safety_net: SafetySegmentation,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.safety_net = safety_net.to(device)
        self.device = device
        
        # Safety parameters
        self.d_safe = config.get('d_safe', 0.05)
        self.d_warning = config.get('d_warning', 0.1)
        self.d_critical = config.get('d_critical', 0.02)
        
        # Action projection
        action_proj = config.get('action_projection', {})
        self.scaling_factor = action_proj.get('scaling_factor', 0.5)
        self.max_scaling = action_proj.get('max_scaling', 0.1)
        
        # Monitoring
        monitoring = config.get('monitoring', {})
        self.violation_cooldown = monitoring.get('violation_cooldown', 10)
        self.max_violations_per_episode = monitoring.get('max_violations_per_episode', 5)
        self.violation_penalty = monitoring.get('violation_penalty', -1.0)
        self.safety_reward = monitoring.get('safety_reward', 0.1)
        
        # Emergency stop
        emergency_stop = config.get('emergency_stop', {})
        self.emergency_enabled = emergency_stop.get('enabled', True)
        self.critical_distance_threshold = emergency_stop.get('critical_distance_threshold', 0.01)
        self.max_consecutive_violations = emergency_stop.get('max_consecutive_violations', 3)
        self.reset_episode_on_stop = emergency_stop.get('reset_episode_on_stop', True)
        
        # Create safety shield
        self.shield = SafetyShield(
            safety_net=safety_net,
            d_safe=self.d_safe,
            d_warning=self.d_warning,
            scaling_factor=self.scaling_factor,
            max_scaling=self.max_scaling
        )
        
        # State tracking
        self.violation_count = 0
        self.consecutive_violations = 0
        self.cooldown_steps = 0
        
        logger.info(f"RLSafetyShield initialized: d_safe={self.d_safe}, d_warning={self.d_warning}")
    
    def __call__(
        self,
        image: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply safety shield to action."""
        # Check cooldown
        if self.cooldown_steps > 0:
            self.cooldown_steps -= 1
            return torch.zeros_like(action), {'safety_level': 'cooldown'}
        
        # Apply safety shield
        projected_action, safety_info = self.shield(image, action)
        
        # Update violation tracking
        if safety_info['safety_level'] in ['warning', 'critical']:
            self.violation_count += 1
            self.consecutive_violations += 1
            
            # Start cooldown
            self.cooldown_steps = self.violation_cooldown
        else:
            self.consecutive_violations = 0
        
        # Check for emergency stop
        if self.emergency_enabled and self._should_emergency_stop(safety_info):
            safety_info['emergency_stop'] = True
            if self.reset_episode_on_stop:
                safety_info['reset_episode'] = True
        
        return projected_action, safety_info
    
    def _should_emergency_stop(self, safety_info: Dict[str, Any]) -> bool:
        """Check if emergency stop should be triggered."""
        # Critical distance
        if safety_info.get('distance', 1.0) < self.critical_distance_threshold:
            return True
        
        # Consecutive violations
        if self.consecutive_violations >= self.max_consecutive_violations:
            return True
        
        return False
    
    def calculate_safety_reward(
        self,
        safety_info: Dict[str, Any],
        base_reward: float
    ) -> float:
        """Calculate safety-aware reward."""
        safety_level = safety_info.get('safety_level', 'safe')
        
        if safety_level == 'safe':
            return base_reward + self.safety_reward
        elif safety_level == 'warning':
            return base_reward + self.violation_penalty
        elif safety_level == 'critical':
            return base_reward + self.violation_penalty * 2
        else:
            return base_reward
    
    def reset_episode(self):
        """Reset episode state."""
        self.violation_count = 0
        self.consecutive_violations = 0
        self.cooldown_steps = 0
        self.shield.reset_violations()
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        return {
            'violation_count': self.violation_count,
            'consecutive_violations': self.consecutive_violations,
            'cooldown_steps': self.cooldown_steps
        }


class SafetyAwareReward:
    """Safety-aware reward calculation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 0.5)
        self.lambda_time = config.get('lambda_time', 0.01)
        self.lambda_action = config.get('lambda_action', 1e-3)
        self.lambda_collision = config.get('lambda_collision', 5.0)
        self.lambda_safety = config.get('lambda_safety', 10.0)
        self.safety_reward_weight = config.get('safety_reward_weight', 0.1)
        self.proximity_reward_weight = config.get('proximity_reward_weight', 0.05)
    
    def calculate_reward(
        self,
        liquid_reduction: float,
        contaminant_reduction: float,
        step_count: int,
        action_norm: float,
        collision: bool,
        safety_info: Dict[str, Any]
    ) -> float:
        """Calculate safety-aware reward."""
        # Base reward components
        liquid_reward = self.alpha * liquid_reduction
        contaminant_reward = self.beta * contaminant_reduction
        time_penalty = self.lambda_time * step_count
        action_penalty = self.lambda_action * action_norm
        collision_penalty = self.lambda_collision * int(collision)
        
        # Safety reward
        safety_reward = self._calculate_safety_reward(safety_info)
        
        # Total reward
        total_reward = (
            liquid_reward +
            contaminant_reward +
            time_penalty +
            action_penalty +
            collision_penalty +
            safety_reward
        )
        
        return total_reward
    
    def _calculate_safety_reward(self, safety_info: Dict[str, Any]) -> float:
        """Calculate safety component of reward."""
        safety_level = safety_info.get('safety_level', 'safe')
        distance = safety_info.get('distance', 1.0)
        action_scale = safety_info.get('action_scale', 1.0)
        
        if safety_level == 'safe':
            # Reward for staying safe
            proximity_reward = self.proximity_reward_weight * distance
            return self.safety_reward_weight + proximity_reward
        elif safety_level == 'warning':
            # Penalty for warning state
            return -self.lambda_safety * 0.5
        elif safety_level == 'critical':
            # Heavy penalty for critical state
            return -self.lambda_safety
        else:
            return 0.0


class SafetyCurriculum:
    """Safety curriculum learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_difficulty = config.get('min_difficulty', 0.3)
        self.max_difficulty = config.get('max_difficulty', 1.0)
        self.success_threshold = config.get('success_threshold', 0.7)
        self.difficulty_increase = config.get('difficulty_increase', 0.1)
        self.evaluation_episodes = config.get('evaluation_episodes', 10)
        
        self.current_difficulty = self.min_difficulty
        self.episode_count = 0
        self.success_count = 0
        self.recent_episodes = []
        
        logger.info(f"SafetyCurriculum initialized: {self.min_difficulty} -> {self.max_difficulty}")
    
    def update(self, success: bool):
        """Update curriculum with episode result."""
        self.episode_count += 1
        self.recent_episodes.append(success)
        
        # Keep only recent episodes
        if len(self.recent_episodes) > self.evaluation_episodes:
            self.recent_episodes.pop(0)
        
        # Check if we should increase difficulty
        if len(self.recent_episodes) >= self.evaluation_episodes:
            success_rate = sum(self.recent_episodes) / len(self.recent_episodes)
            
            if success_rate >= self.success_threshold:
                self.current_difficulty = min(
                    self.max_difficulty,
                    self.current_difficulty + self.difficulty_increase
                )
                logger.info(f"Increased difficulty to {self.current_difficulty:.2f}")
    
    def get_difficulty_params(self) -> Dict[str, Any]:
        """Get difficulty parameters for current level."""
        return {
            'liquid_mass': 100.0 * (0.5 + 0.5 * self.current_difficulty),
            'contaminant_mass': 50.0 * (0.5 + 0.5 * self.current_difficulty),
            'noise_level': 0.1 * (1.0 - self.current_difficulty),
            'time_limit': 1000 * (0.5 + 0.5 * self.current_difficulty),
            'precision_required': self.current_difficulty,
            'safety_threshold': 0.05 * (1.0 - self.current_difficulty * 0.5)
        }
