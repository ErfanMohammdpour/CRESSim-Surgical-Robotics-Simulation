"""
Curriculum learning for surgical robotics tasks.
"""

import numpy as np
from typing import Dict, Any, Tuple
from collections import deque


class CurriculumLearning:
    """Dynamic curriculum learning for surgical tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_level = 0
        self.max_level = config.get('max_level', 5)
        self.success_window = deque(maxlen=config.get('success_window', 100))
        self.success_threshold = config.get('success_threshold', 0.8)
        self.level_increase_threshold = config.get('level_increase_threshold', 0.8)
        self.level_decrease_threshold = config.get('level_decrease_threshold', 0.3)
        
        # Task parameters for each level
        self.task_params = self._initialize_task_params()
    
    def _initialize_task_params(self) -> Dict[int, Dict[str, Any]]:
        """Initialize task parameters for each curriculum level."""
        params = {}
        
        for level in range(self.max_level + 1):
            # Liquid mass decreases with level (easier -> harder)
            liquid_mass = max(0.1, 1.0 - level * 0.15)
            
            # Contaminant density increases with level
            contaminant_density = min(1.0, 0.3 + level * 0.1)
            
            # Tool precision requirements increase
            precision_requirement = min(1.0, 0.5 + level * 0.1)
            
            # Episode length increases
            max_steps = min(1000, 200 + level * 100)
            
            params[level] = {
                'liquid_mass': liquid_mass,
                'contaminant_density': contaminant_density,
                'precision_requirement': precision_requirement,
                'max_steps': max_steps,
                'reward_scale': 1.0 + level * 0.1,  # Higher rewards for harder tasks
                'safety_threshold': max(0.01, 0.05 - level * 0.005)  # Tighter safety
            }
        
        return params
    
    def update(self, episode_reward: float, success: bool, steps: int) -> None:
        """Update curriculum based on episode performance."""
        self.success_window.append(success)
        
        if len(self.success_window) >= self.success_window.maxlen:
            success_rate = np.mean(self.success_window)
            self._adjust_level(success_rate)
    
    def _adjust_level(self, success_rate: float) -> None:
        """Adjust curriculum level based on success rate."""
        if success_rate >= self.level_increase_threshold and self.current_level < self.max_level:
            self.current_level += 1
            print(f"🎓 Curriculum level increased to {self.current_level}")
        elif success_rate < self.level_decrease_threshold and self.current_level > 0:
            self.current_level -= 1
            print(f"📚 Curriculum level decreased to {self.current_level}")
    
    def get_task_params(self) -> Dict[str, Any]:
        """Get current task parameters."""
        return self.task_params[self.current_level].copy()
    
    def get_level(self) -> int:
        """Get current curriculum level."""
        return self.current_level
    
    def get_success_rate(self) -> float:
        """Get current success rate."""
        if len(self.success_window) == 0:
            return 0.0
        return np.mean(self.success_window)


class AdaptiveDifficulty:
    """Adaptive difficulty adjustment based on performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = deque(maxlen=config.get('history_size', 50))
        self.difficulty_factor = 1.0
        self.min_difficulty = config.get('min_difficulty', 0.5)
        self.max_difficulty = config.get('max_difficulty', 2.0)
        self.adjustment_rate = config.get('adjustment_rate', 0.1)
    
    def update(self, performance: float) -> None:
        """Update difficulty based on performance."""
        self.performance_history.append(performance)
        
        if len(self.performance_history) >= 10:
            avg_performance = np.mean(self.performance_history)
            target_performance = self.config.get('target_performance', 0.7)
            
            if avg_performance > target_performance + 0.1:
                # Performance too high, increase difficulty
                self.difficulty_factor = min(
                    self.max_difficulty,
                    self.difficulty_factor + self.adjustment_rate
                )
            elif avg_performance < target_performance - 0.1:
                # Performance too low, decrease difficulty
                self.difficulty_factor = max(
                    self.min_difficulty,
                    self.difficulty_factor - self.adjustment_rate
                )
    
    def get_difficulty_factor(self) -> float:
        """Get current difficulty factor."""
        return self.difficulty_factor


class SafetyCurriculum:
    """Curriculum learning focused on safety constraints."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.safety_violations = deque(maxlen=config.get('violation_window', 100))
        self.current_safety_level = 0
        self.max_safety_level = config.get('max_safety_level', 3)
        
        # Safety thresholds for each level
        self.safety_thresholds = [
            0.05,  # Level 0: Very safe
            0.03,  # Level 1: Safe
            0.02,  # Level 2: Tight
            0.01,  # Level 3: Very tight
        ]
    
    def update(self, violation_count: int, total_actions: int) -> None:
        """Update safety curriculum based on violation rate."""
        violation_rate = violation_count / max(total_actions, 1)
        self.safety_violations.append(violation_rate)
        
        if len(self.safety_violations) >= self.safety_violations.maxlen:
            avg_violation_rate = np.mean(self.safety_violations)
            self._adjust_safety_level(avg_violation_rate)
    
    def _adjust_safety_level(self, violation_rate: float) -> None:
        """Adjust safety level based on violation rate."""
        target_violation_rate = self.config.get('target_violation_rate', 0.05)
        
        if violation_rate < target_violation_rate * 0.5 and self.current_safety_level < self.max_safety_level:
            self.current_safety_level += 1
            print(f"🛡️ Safety level increased to {self.current_safety_level}")
        elif violation_rate > target_violation_rate * 2.0 and self.current_safety_level > 0:
            self.current_safety_level -= 1
            print(f"⚠️ Safety level decreased to {self.current_safety_level}")
    
    def get_safety_threshold(self) -> float:
        """Get current safety threshold."""
        return self.safety_thresholds[self.current_safety_level]
    
    def get_safety_level(self) -> int:
        """Get current safety level."""
        return self.current_safety_level


class MultiObjectiveCurriculum:
    """Curriculum learning for multiple objectives."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.objectives = config.get('objectives', ['efficiency', 'safety', 'precision'])
        self.weights = config.get('weights', [0.4, 0.4, 0.2])
        self.performance_history = {obj: deque(maxlen=100) for obj in self.objectives}
        
        # Objective-specific curricula
        self.efficiency_curriculum = CurriculumLearning(config.get('efficiency', {}))
        self.safety_curriculum = SafetyCurriculum(config.get('safety', {}))
        self.precision_curriculum = AdaptiveDifficulty(config.get('precision', {}))
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update all curricula based on performance metrics."""
        for obj in self.objectives:
            if obj in metrics:
                self.performance_history[obj].append(metrics[obj])
        
        # Update individual curricula
        if 'efficiency' in metrics:
            self.efficiency_curriculum.update(
                metrics['efficiency'], 
                metrics.get('success', False),
                metrics.get('steps', 0)
            )
        
        if 'safety_violation_rate' in metrics:
            self.safety_curriculum.update(
                metrics.get('violation_count', 0),
                metrics.get('total_actions', 1)
            )
        
        if 'precision' in metrics:
            self.precision_curriculum.update(metrics['precision'])
    
    def get_combined_params(self) -> Dict[str, Any]:
        """Get combined parameters from all curricula."""
        params = {}
        
        # Efficiency parameters
        params.update(self.efficiency_curriculum.get_task_params())
        
        # Safety parameters
        params['safety_threshold'] = self.safety_curriculum.get_safety_threshold()
        
        # Precision parameters
        params['precision_factor'] = self.precision_curriculum.get_difficulty_factor()
        
        return params
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all objectives."""
        summary = {}
        
        for obj in self.objectives:
            if len(self.performance_history[obj]) > 0:
                summary[f'{obj}_mean'] = np.mean(self.performance_history[obj])
                summary[f'{obj}_std'] = np.std(self.performance_history[obj])
                summary[f'{obj}_trend'] = self._calculate_trend(self.performance_history[obj])
        
        return summary
    
    def _calculate_trend(self, history: deque) -> str:
        """Calculate trend direction."""
        if len(history) < 10:
            return "insufficient_data"
        
        recent = list(history)[-10:]
        older = list(history)[-20:-10] if len(history) >= 20 else list(history)[:-10]
        
        if len(older) == 0:
            return "insufficient_data"
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        if recent_mean > older_mean * 1.05:
            return "improving"
        elif recent_mean < older_mean * 0.95:
            return "declining"
        else:
            return "stable"
