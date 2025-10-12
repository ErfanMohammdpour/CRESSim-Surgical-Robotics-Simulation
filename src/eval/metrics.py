"""
Evaluation metrics for RL performance assessment.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate various metrics for RL evaluation.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_metrics = []
    
    def add_episode(self, episode_data: Dict[str, Any]) -> None:
        """Add episode data for metric calculation."""
        self.episode_metrics.append(episode_data)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all metrics from episode data."""
        if not self.episode_metrics:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics())
        
        # Performance metrics
        metrics.update(self._calculate_performance_metrics())
        
        # Safety metrics
        metrics.update(self._calculate_safety_metrics())
        
        # Efficiency metrics
        metrics.update(self._calculate_efficiency_metrics())
        
        # Quality metrics
        metrics.update(self._calculate_quality_metrics())
        
        return metrics
    
    def _calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic episode metrics."""
        episode_rewards = [ep['total_reward'] for ep in self.episode_metrics]
        episode_lengths = [ep['episode_length'] for ep in self.episode_metrics]
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths)),
            'min_length': int(np.min(episode_lengths)),
            'max_length': int(np.max(episode_lengths)),
            'num_episodes': len(self.episode_metrics)
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance-related metrics."""
        success_rates = []
        liquid_reductions = []
        contaminant_reductions = []
        
        for ep in self.episode_metrics:
            # Success rate
            liquid_mass = ep.get('final_liquid_mass', 1.0)
            contaminant_mass = ep.get('final_contaminant_mass', 1.0)
            success = (liquid_mass < 0.1) and (contaminant_mass < 0.05)
            success_rates.append(success)
            
            # Mass reductions
            initial_liquid = ep.get('initial_liquid_mass', 1.0)
            initial_contaminant = ep.get('initial_contaminant_mass', 0.5)
            
            liquid_reduction = max(0, initial_liquid - liquid_mass) / max(initial_liquid, 1e-6)
            contaminant_reduction = max(0, initial_contaminant - contaminant_mass) / max(initial_contaminant, 1e-6)
            
            liquid_reductions.append(liquid_reduction)
            contaminant_reductions.append(contaminant_reduction)
        
        return {
            'success_rate': float(np.mean(success_rates)),
            'mean_liquid_reduction': float(np.mean(liquid_reductions)),
            'std_liquid_reduction': float(np.std(liquid_reductions)),
            'mean_contaminant_reduction': float(np.mean(contaminant_reductions)),
            'std_contaminant_reduction': float(np.std(contaminant_reductions))
        }
    
    def _calculate_safety_metrics(self) -> Dict[str, float]:
        """Calculate safety-related metrics."""
        total_collisions = 0
        total_safety_violations = 0
        episodes_with_violations = 0
        violation_durations = []
        
        for ep in self.episode_metrics:
            collisions = ep.get('total_collisions', 0)
            safety_violations = ep.get('total_safety_violations', 0)
            violation_duration = ep.get('violation_duration', 0)
            
            total_collisions += collisions
            total_safety_violations += safety_violations
            
            if safety_violations > 0:
                episodes_with_violations += 1
                violation_durations.append(violation_duration)
        
        num_episodes = len(self.episode_metrics)
        
        return {
            'mean_collisions_per_episode': float(total_collisions / num_episodes),
            'mean_safety_violations_per_episode': float(total_safety_violations / num_episodes),
            'episodes_with_violations_rate': float(episodes_with_violations / num_episodes),
            'mean_violation_duration': float(np.mean(violation_durations)) if violation_durations else 0.0,
            'std_violation_duration': float(np.std(violation_durations)) if violation_durations else 0.0
        }
    
    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency-related metrics."""
        path_lengths = []
        action_smoothness = []
        energy_consumption = []
        
        for ep in self.episode_metrics:
            # Path length (sum of action magnitudes)
            actions = ep.get('actions', [])
            if actions:
                path_length = np.sum([np.linalg.norm(action) for action in actions])
                path_lengths.append(path_length)
                
                # Action smoothness (inverse of action changes)
                if len(actions) > 1:
                    action_diffs = np.diff(actions, axis=0)
                    smoothness = 1.0 / (1.0 + np.mean([np.linalg.norm(diff) for diff in action_diffs]))
                    action_smoothness.append(smoothness)
            
            # Energy consumption (sum of squared actions)
            if actions:
                energy = np.sum([np.sum(action**2) for action in actions])
                energy_consumption.append(energy)
        
        return {
            'mean_path_length': float(np.mean(path_lengths)) if path_lengths else 0.0,
            'std_path_length': float(np.std(path_lengths)) if path_lengths else 0.0,
            'mean_action_smoothness': float(np.mean(action_smoothness)) if action_smoothness else 0.0,
            'std_action_smoothness': float(np.std(action_smoothness)) if action_smoothness else 0.0,
            'mean_energy_consumption': float(np.mean(energy_consumption)) if energy_consumption else 0.0,
            'std_energy_consumption': float(np.std(energy_consumption)) if energy_consumption else 0.0
        }
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate quality-related metrics."""
        # Task completion quality
        completion_qualities = []
        precision_scores = []
        
        for ep in self.episode_metrics:
            # Completion quality (based on remaining mass)
            liquid_mass = ep.get('final_liquid_mass', 1.0)
            contaminant_mass = ep.get('final_contaminant_mass', 0.5)
            
            # Quality score (higher is better)
            quality = 1.0 - (liquid_mass + contaminant_mass) / 1.5
            completion_qualities.append(max(0, quality))
            
            # Precision score (based on action consistency)
            actions = ep.get('actions', [])
            if len(actions) > 1:
                action_consistency = 1.0 - np.std([np.linalg.norm(action) for action in actions])
                precision_scores.append(max(0, action_consistency))
        
        return {
            'mean_completion_quality': float(np.mean(completion_qualities)),
            'std_completion_quality': float(np.std(completion_qualities)),
            'mean_precision_score': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'std_precision_score': float(np.std(precision_scores)) if precision_scores else 0.0
        }
    
    def get_episode_breakdown(self) -> List[Dict[str, Any]]:
        """Get detailed breakdown for each episode."""
        breakdown = []
        
        for i, ep in enumerate(self.episode_metrics):
            episode_info = {
                'episode_id': i,
                'total_reward': ep.get('total_reward', 0.0),
                'episode_length': ep.get('episode_length', 0),
                'success': self._is_episode_successful(ep),
                'liquid_reduction': self._calculate_liquid_reduction(ep),
                'contaminant_reduction': self._calculate_contaminant_reduction(ep),
                'collisions': ep.get('total_collisions', 0),
                'safety_violations': ep.get('total_safety_violations', 0),
                'path_length': self._calculate_path_length(ep),
                'action_smoothness': self._calculate_action_smoothness(ep)
            }
            breakdown.append(episode_info)
        
        return breakdown
    
    def _is_episode_successful(self, ep: Dict[str, Any]) -> bool:
        """Check if episode was successful."""
        liquid_mass = ep.get('final_liquid_mass', 1.0)
        contaminant_mass = ep.get('final_contaminant_mass', 0.5)
        return (liquid_mass < 0.1) and (contaminant_mass < 0.05)
    
    def _calculate_liquid_reduction(self, ep: Dict[str, Any]) -> float:
        """Calculate liquid mass reduction for episode."""
        initial = ep.get('initial_liquid_mass', 1.0)
        final = ep.get('final_liquid_mass', 1.0)
        return max(0, initial - final) / max(initial, 1e-6)
    
    def _calculate_contaminant_reduction(self, ep: Dict[str, Any]) -> float:
        """Calculate contaminant mass reduction for episode."""
        initial = ep.get('initial_contaminant_mass', 0.5)
        final = ep.get('final_contaminant_mass', 0.5)
        return max(0, initial - final) / max(initial, 1e-6)
    
    def _calculate_path_length(self, ep: Dict[str, Any]) -> float:
        """Calculate path length for episode."""
        actions = ep.get('actions', [])
        if not actions:
            return 0.0
        return float(np.sum([np.linalg.norm(action) for action in actions]))
    
    def _calculate_action_smoothness(self, ep: Dict[str, Any]) -> float:
        """Calculate action smoothness for episode."""
        actions = ep.get('actions', [])
        if len(actions) < 2:
            return 1.0
        
        action_diffs = np.diff(actions, axis=0)
        smoothness = 1.0 / (1.0 + np.mean([np.linalg.norm(diff) for diff in action_diffs]))
        return float(smoothness)
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        metrics = self.calculate_metrics()
        episode_breakdown = self.get_episode_breakdown()
        
        data = {
            'metrics': metrics,
            'episode_breakdown': episode_breakdown,
            'summary': {
                'total_episodes': len(self.episode_metrics),
                'success_rate': metrics.get('success_rate', 0.0),
                'mean_reward': metrics.get('mean_reward', 0.0),
                'mean_liquid_reduction': metrics.get('mean_liquid_reduction', 0.0),
                'mean_contaminant_reduction': metrics.get('mean_contaminant_reduction', 0.0)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


class ComparativeMetrics:
    """
    Calculate comparative metrics between different models/approaches.
    """
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, name: str, metrics: Dict[str, float]) -> None:
        """Add result for comparison."""
        self.results[name] = metrics
    
    def calculate_comparison(self) -> Dict[str, Any]:
        """Calculate comparative analysis."""
        if len(self.results) < 2:
            return {}
        
        comparison = {
            'methods': list(self.results.keys()),
            'metrics': {},
            'rankings': {},
            'statistical_tests': {}
        }
        
        # Extract all metric names
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        
        # Calculate comparison for each metric
        for metric in all_metrics:
            values = {}
            for name, metrics in self.results.items():
                if metric in metrics:
                    values[name] = metrics[metric]
            
            if values:
                comparison['metrics'][metric] = values
                
                # Calculate ranking
                sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
                comparison['rankings'][metric] = [name for name, _ in sorted_values]
        
        return comparison
    
    def save_comparison(self, filepath: str) -> None:
        """Save comparison results to JSON file."""
        comparison = self.calculate_comparison()
        
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison results saved to {filepath}")


def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate confidence interval for data."""
    if not data:
        return 0.0, 0.0, 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    t_value = 1.96  # Approximate for 95% confidence
    margin_error = t_value * (std / np.sqrt(n))
    
    lower = mean - margin_error
    upper = mean + margin_error
    
    return mean, lower, upper


def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size between two groups."""
    if not group1 or not group2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1), np.std(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                         (len(group1) + len(group2) - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std
