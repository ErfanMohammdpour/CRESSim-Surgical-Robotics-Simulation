"""
Safety shield for RL training and inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

from ..vision.safety_seg import SafetySegmentation, SafetyShield

logger = logging.getLogger(__name__)


class RLSafetyShield:
    """
    Safety shield for RL training that integrates with the training loop.
    """
    
    def __init__(
        self,
        safety_net: SafetySegmentation,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.safety_net = safety_net.to(device)
        self.device = device
        self.config = config
        
        # Safety parameters
        self.d_safe = config.get('d_safe', 0.05)
        self.d_warning = config.get('d_warning', 0.1)
        self.d_critical = config.get('d_critical', 0.02)
        
        # Action projection parameters
        action_config = config.get('action_projection', {})
        self.scaling_factor = action_config.get('scaling_factor', 0.5)
        self.max_scaling = action_config.get('max_scaling', 0.1)
        self.projection_method = action_config.get('projection_method', 'scale')
        
        # Safety monitoring
        monitoring_config = config.get('monitoring', {})
        self.violation_cooldown = monitoring_config.get('violation_cooldown', 10)
        self.max_violations_per_episode = monitoring_config.get('max_violations_per_episode', 5)
        self.violation_penalty = monitoring_config.get('violation_penalty', -1.0)
        self.safety_reward = monitoring_config.get('safety_reward', 0.1)
        
        # Emergency stop
        emergency_config = config.get('emergency_stop', {})
        self.emergency_enabled = emergency_config.get('enabled', True)
        self.critical_distance_threshold = emergency_config.get('critical_distance_threshold', 0.01)
        self.max_consecutive_violations = emergency_config.get('max_consecutive_violations', 3)
        self.reset_episode_on_stop = emergency_config.get('reset_episode_on_stop', True)
        
        # State tracking
        self.violation_count = 0
        self.consecutive_violations = 0
        self.last_safety_state = True
        self.cooldown_timer = 0
        self.episode_violations = 0
        
        # Create safety shield
        self.shield = SafetyShield(
            safety_net=safety_net,
            d_safe=self.d_safe,
            d_warning=self.d_warning,
            scaling_factor=self.scaling_factor,
            max_scaling=self.max_scaling
        )
    
    def reset(self) -> None:
        """Reset safety shield state."""
        self.violation_count = 0
        self.consecutive_violations = 0
        self.last_safety_state = True
        self.cooldown_timer = 0
        self.episode_violations = 0
        self.shield.reset()
    
    def apply_safety_shield(
        self, 
        image: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply safety shield to action."""
        # Check cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return torch.zeros_like(action), {'safety_level': 'cooldown', 'action_scale': 0.0}
        
        # Apply safety shield
        projected_action, safety_info = self.shield(image, action)
        
        # Update safety state
        safety_level = safety_info['safety_level']
        is_safe = safety_info.get('is_safe', True)
        
        # Track violations
        if safety_level in ['critical', 'warning']:
            self.violation_count += 1
            self.consecutive_violations += 1
            self.episode_violations += 1
            
            # Check for emergency stop
            if self._should_emergency_stop(safety_info):
                projected_action = torch.zeros_like(action)
                safety_info['safety_level'] = 'emergency_stop'
                safety_info['action_scale'] = 0.0
                
                # Set cooldown
                self.cooldown_timer = self.violation_cooldown
                
                logger.warning("Emergency stop triggered due to safety violations")
        else:
            self.consecutive_violations = 0
        
        # Update safety info
        safety_info.update({
            'violation_count': self.violation_count,
            'consecutive_violations': self.consecutive_violations,
            'episode_violations': self.episode_violations,
            'cooldown_timer': self.cooldown_timer
        })
        
        return projected_action, safety_info
    
    def _should_emergency_stop(self, safety_info: Dict[str, Any]) -> bool:
        """Check if emergency stop should be triggered."""
        if not self.emergency_enabled:
            return False
        
        # Check critical distance
        distance = safety_info.get('distance', 1.0)
        if distance < self.critical_distance_threshold:
            return True
        
        # Check consecutive violations
        if self.consecutive_violations >= self.max_consecutive_violations:
            return True
        
        # Check episode violations
        if self.episode_violations >= self.max_violations_per_episode:
            return True
        
        return False
    
    def calculate_safety_reward(
        self, 
        safety_info: Dict[str, Any], 
        base_reward: float
    ) -> float:
        """Calculate safety-adjusted reward."""
        safety_level = safety_info.get('safety_level', 'safe')
        
        # Base reward
        reward = base_reward
        
        # Safety reward for staying safe
        if safety_level == 'safe':
            reward += self.safety_reward
        
        # Violation penalty
        if safety_level in ['warning', 'critical']:
            reward += self.violation_penalty
        
        # Emergency stop penalty
        if safety_level == 'emergency_stop':
            reward += self.violation_penalty * 2
        
        return reward
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get current safety metrics."""
        return {
            'violation_count': self.violation_count,
            'consecutive_violations': self.consecutive_violations,
            'episode_violations': self.episode_violations,
            'cooldown_timer': self.cooldown_timer,
            'last_safety_state': self.last_safety_state
        }


class SafetyAwareReward:
    """
    Safety-aware reward shaping for RL training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Reward weights
        self.alpha = config.get('alpha', 1.0)  # liquid mass reduction
        self.beta = config.get('beta', 0.5)    # contaminant mass reduction
        self.lambda_time = config.get('lambda_time', -0.01)  # time penalty
        self.lambda_action = config.get('lambda_action', -0.001)  # action smoothness
        self.lambda_collision = config.get('lambda_collision', -1.0)  # collision penalty
        self.lambda_safety = config.get('lambda_safety', -2.0)  # safety violation penalty
        
        # Safety-specific weights
        self.safety_reward_weight = config.get('safety_reward_weight', 0.1)
        self.proximity_reward_weight = config.get('proximity_reward_weight', 0.05)
        self.smooth_action_reward_weight = config.get('smooth_action_reward_weight', 0.02)
    
    def calculate_reward(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray, 
        next_obs: Dict[str, np.ndarray], 
        info: Dict[str, Any],
        safety_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate safety-aware reward."""
        # Base reward components
        liquid_reward = self._calculate_liquid_reward(obs, next_obs)
        contaminant_reward = self._calculate_contaminant_reward(obs, next_obs)
        time_penalty = self.lambda_time
        action_penalty = self._calculate_action_penalty(action)
        collision_penalty = self._calculate_collision_penalty(info)
        
        # Safety reward components
        safety_reward = 0.0
        if safety_info is not None:
            safety_reward = self._calculate_safety_reward(safety_info)
        
        # Combine rewards
        total_reward = (
            self.alpha * liquid_reward +
            self.beta * contaminant_reward +
            time_penalty +
            action_penalty +
            collision_penalty +
            safety_reward
        )
        
        return total_reward
    
    def _calculate_liquid_reward(self, obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray]) -> float:
        """Calculate liquid mass reduction reward."""
        current_liquid = obs['aux'][1] if 'aux' in obs else 0.0
        next_liquid = next_obs['aux'][1] if 'aux' in next_obs else 0.0
        
        reduction = max(0, current_liquid - next_liquid)
        return reduction
    
    def _calculate_contaminant_reward(self, obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray]) -> float:
        """Calculate contaminant mass reduction reward."""
        current_contaminant = obs['aux'][2] if 'aux' in obs else 0.0
        next_contaminant = next_obs['aux'][2] if 'aux' in next_obs else 0.0
        
        reduction = max(0, current_contaminant - next_contaminant)
        return reduction
    
    def _calculate_action_penalty(self, action: np.ndarray) -> float:
        """Calculate action smoothness penalty."""
        action_magnitude = np.linalg.norm(action)
        return self.lambda_action * action_magnitude
    
    def _calculate_collision_penalty(self, info: Dict[str, Any]) -> float:
        """Calculate collision penalty."""
        collisions = info.get('collisions', 0)
        return self.lambda_collision * collisions
    
    def _calculate_safety_reward(self, safety_info: Dict[str, Any]) -> float:
        """Calculate safety-specific reward."""
        safety_level = safety_info.get('safety_level', 'safe')
        distance = safety_info.get('distance', 1.0)
        action_scale = safety_info.get('action_scale', 1.0)
        
        reward = 0.0
        
        # Safety level reward
        if safety_level == 'safe':
            reward += self.safety_reward_weight
        elif safety_level == 'warning':
            reward += self.safety_reward_weight * 0.5
        elif safety_level in ['critical', 'emergency_stop']:
            reward += self.lambda_safety
        
        # Proximity reward (closer to target is better, but not too close)
        if 0.1 <= distance <= 0.5:
            reward += self.proximity_reward_weight * (0.5 - distance)
        
        # Smooth action reward
        if action_scale > 0.8:  # High action scale means smooth action
            reward += self.smooth_action_reward_weight
        
        return reward


class SafetyCurriculum:
    """
    Safety curriculum that adjusts safety thresholds based on performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Curriculum parameters
        self.initial_d_safe = config.get('initial_d_safe', 0.1)
        self.final_d_safe = config.get('final_d_safe', 0.05)
        self.initial_d_warning = config.get('initial_d_warning', 0.2)
        self.final_d_warning = config.get('final_d_warning', 0.1)
        
        # Performance tracking
        self.success_rate_window = config.get('success_rate_window', 100)
        self.success_rates = []
        self.current_d_safe = self.initial_d_safe
        self.current_d_warning = self.initial_d_warning
        
        # Update parameters
        self.update_frequency = config.get('update_frequency', 1000)
        self.success_threshold = config.get('success_threshold', 0.8)
        self.difficulty_increase = config.get('difficulty_increase', 0.01)
    
    def update(self, episode_success: bool) -> None:
        """Update curriculum based on episode success."""
        self.success_rates.append(episode_success)
        
        # Keep only recent episodes
        if len(self.success_rates) > self.success_rate_window:
            self.success_rates.pop(0)
        
        # Update safety thresholds
        if len(self.success_rates) >= self.success_rate_window:
            success_rate = np.mean(self.success_rates)
            
            if success_rate >= self.success_threshold:
                # Increase difficulty (tighter safety thresholds)
                self.current_d_safe = max(
                    self.final_d_safe,
                    self.current_d_safe - self.difficulty_increase
                )
                self.current_d_warning = max(
                    self.final_d_warning,
                    self.current_d_warning - self.difficulty_increase
                )
            else:
                # Decrease difficulty (looser safety thresholds)
                self.current_d_safe = min(
                    self.initial_d_safe,
                    self.current_d_safe + self.difficulty_increase
                )
                self.current_d_warning = min(
                    self.initial_d_warning,
                    self.current_d_warning + self.difficulty_increase
                )
    
    def get_safety_thresholds(self) -> Dict[str, float]:
        """Get current safety thresholds."""
        return {
            'd_safe': self.current_d_safe,
            'd_warning': self.current_d_warning
        }
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get curriculum information."""
        return {
            'current_d_safe': self.current_d_safe,
            'current_d_warning': self.current_d_warning,
            'success_rate': np.mean(self.success_rates) if self.success_rates else 0.0,
            'num_episodes': len(self.success_rates)
        }
