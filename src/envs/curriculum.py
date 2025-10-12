"""
Curriculum learning implementation for progressive difficulty increase.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CurriculumLearning:
    """
    Curriculum learning system that progressively increases task difficulty.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Curriculum parameters
        self.min_difficulty = config.get('min_difficulty', 0.3)
        self.max_difficulty = config.get('max_difficulty', 1.0)
        self.success_threshold = config.get('success_threshold', 0.7)
        self.difficulty_increase = config.get('difficulty_increase', 0.1)
        self.evaluation_episodes = config.get('evaluation_episodes', 10)
        self.update_frequency = config.get('update_frequency', 1000)
        
        # State
        self.current_difficulty = self.min_difficulty
        self.episode_count = 0
        self.recent_episodes = deque(maxlen=self.evaluation_episodes)
        self.success_rate = 0.0
        
        # Difficulty components
        self.difficulty_components = {
            'liquid_mass': 0.0,      # Initial liquid mass multiplier
            'contaminant_mass': 0.0, # Initial contaminant mass multiplier
            'noise_level': 0.0,      # Action noise level
            'time_limit': 0.0,       # Episode time limit
            'precision_required': 0.0 # Required precision for success
        }
    
    def update(self, episode_info: Dict[str, Any]) -> None:
        """Update curriculum based on episode results."""
        self.episode_count += 1
        
        # Check if episode was successful
        success = self._evaluate_episode_success(episode_info)
        self.recent_episodes.append(success)
        
        # Update curriculum if enough episodes have been collected
        if len(self.recent_episodes) >= self.evaluation_episodes:
            if self.episode_count % self.update_frequency == 0:
                self._update_difficulty()
    
    def _evaluate_episode_success(self, episode_info: Dict[str, Any]) -> bool:
        """Evaluate if an episode was successful."""
        # Success criteria
        liquid_mass_remaining = episode_info.get('liquid_mass_remaining', 1.0)
        contaminant_mass_remaining = episode_info.get('contaminant_mass_remaining', 1.0)
        safety_violations = episode_info.get('episode_safety_violations', 0)
        collisions = episode_info.get('episode_collisions', 0)
        
        # Success thresholds (become stricter with difficulty)
        liquid_threshold = 0.1 + (1.0 - self.current_difficulty) * 0.05
        contaminant_threshold = 0.05 + (1.0 - self.current_difficulty) * 0.02
        max_violations = int(5 * (1.0 - self.current_difficulty))
        max_collisions = int(3 * (1.0 - self.current_difficulty))
        
        # Check success criteria
        liquid_success = liquid_mass_remaining < liquid_threshold
        contaminant_success = contaminant_mass_remaining < contaminant_threshold
        safety_success = safety_violations <= max_violations
        collision_success = collisions <= max_collisions
        
        return liquid_success and contaminant_success and safety_success and collision_success
    
    def _update_difficulty(self) -> None:
        """Update difficulty based on recent performance."""
        # Calculate success rate
        self.success_rate = np.mean(list(self.recent_episodes))
        
        logger.info(f"Curriculum update - Success rate: {self.success_rate:.3f}, "
                   f"Current difficulty: {self.current_difficulty:.3f}")
        
        # Increase difficulty if success rate is high enough
        if self.success_rate >= self.success_threshold:
            old_difficulty = self.current_difficulty
            self.current_difficulty = min(
                self.max_difficulty,
                self.current_difficulty + self.difficulty_increase
            )
            
            logger.info(f"Difficulty increased: {old_difficulty:.3f} -> {self.current_difficulty:.3f}")
            
            # Clear recent episodes for next evaluation
            self.recent_episodes.clear()
        else:
            logger.info(f"Success rate {self.success_rate:.3f} below threshold {self.success_threshold:.3f}, "
                       f"keeping difficulty at {self.current_difficulty:.3f}")
    
    def get_difficulty_params(self) -> Dict[str, Any]:
        """Get current difficulty parameters for environment."""
        # Calculate difficulty components
        self.difficulty_components['liquid_mass'] = 0.5 + 0.5 * self.current_difficulty
        self.difficulty_components['contaminant_mass'] = 0.3 + 0.7 * self.current_difficulty
        self.difficulty_components['noise_level'] = 0.1 * (1.0 - self.current_difficulty)
        self.difficulty_components['time_limit'] = int(1000 * (2.0 - self.current_difficulty))
        self.difficulty_components['precision_required'] = self.current_difficulty
        
        return self.difficulty_components.copy()
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get current curriculum information."""
        return {
            'current_difficulty': self.current_difficulty,
            'success_rate': self.success_rate,
            'episode_count': self.episode_count,
            'recent_episodes_count': len(self.recent_episodes),
            'difficulty_components': self.get_difficulty_params()
        }
    
    def reset(self) -> None:
        """Reset curriculum to initial state."""
        self.current_difficulty = self.min_difficulty
        self.episode_count = 0
        self.recent_episodes.clear()
        self.success_rate = 0.0
        
        logger.info("Curriculum reset to initial state")


class AdaptiveCurriculum(CurriculumLearning):
    """
    Adaptive curriculum that adjusts based on learning progress.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Adaptive parameters
        self.performance_window = config.get('performance_window', 50)
        self.performance_history = deque(maxlen=self.performance_window)
        self.adaptation_rate = config.get('adaptation_rate', 0.05)
        self.min_adaptation = config.get('min_adaptation', 0.01)
        
        # Performance tracking
        self.performance_trend = 0.0
        self.last_performance = 0.0
    
    def update(self, episode_info: Dict[str, Any]) -> None:
        """Update adaptive curriculum."""
        super().update(episode_info)
        
        # Track performance trend
        current_performance = self._calculate_performance(episode_info)
        self.performance_history.append(current_performance)
        
        if len(self.performance_history) >= 10:
            self._update_performance_trend()
            self._adaptive_difficulty_adjustment()
    
    def _calculate_performance(self, episode_info: Dict[str, Any]) -> float:
        """Calculate episode performance score."""
        liquid_mass = episode_info.get('liquid_mass_remaining', 1.0)
        contaminant_mass = episode_info.get('contaminant_mass_remaining', 1.0)
        safety_violations = episode_info.get('episode_safety_violations', 0)
        collisions = episode_info.get('episode_collisions', 0)
        
        # Performance components
        liquid_score = max(0, 1.0 - liquid_mass)
        contaminant_score = max(0, 1.0 - contaminant_mass)
        safety_score = max(0, 1.0 - safety_violations / 10.0)
        collision_score = max(0, 1.0 - collisions / 5.0)
        
        # Weighted performance
        performance = (
            0.4 * liquid_score +
            0.3 * contaminant_score +
            0.2 * safety_score +
            0.1 * collision_score
        )
        
        return performance
    
    def _update_performance_trend(self) -> None:
        """Update performance trend analysis."""
        if len(self.performance_history) < 10:
            return
        
        # Calculate trend using linear regression
        x = np.arange(len(self.performance_history))
        y = np.array(list(self.performance_history))
        
        # Simple linear trend
        self.performance_trend = np.polyfit(x, y, 1)[0]
        self.last_performance = y[-1]
    
    def _adaptive_difficulty_adjustment(self) -> None:
        """Adjust difficulty based on performance trend."""
        if abs(self.performance_trend) < 0.001:  # No clear trend
            return
        
        # Calculate adaptation amount
        adaptation = self.adaptation_rate * self.performance_trend
        adaptation = np.clip(adaptation, -self.adaptation_rate, self.adaptation_rate)
        
        # Apply adaptation
        old_difficulty = self.current_difficulty
        self.current_difficulty = np.clip(
            self.current_difficulty + adaptation,
            self.min_difficulty,
            self.max_difficulty
        )
        
        # Log significant changes
        if abs(self.current_difficulty - old_difficulty) > self.min_adaptation:
            logger.info(f"Adaptive difficulty adjustment: {old_difficulty:.3f} -> "
                       f"{self.current_difficulty:.3f} (trend: {self.performance_trend:.4f})")
