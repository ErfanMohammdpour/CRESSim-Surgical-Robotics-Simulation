"""
Demo filtering and weighting for improved imitation learning.
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class DemoFilter:
    """
    Filter demonstrations based on quality metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get('method', 'gmm')
        self.gmm_components = config.get('gmm_components', 3)
        self.percentile_threshold = config.get('percentile_threshold', 0.7)
        self.min_episode_length = config.get('min_episode_length', 100)
        self.max_episode_length = config.get('max_episode_length', 1000)
        
        # Quality metrics weights
        self.quality_weights = config.get('quality_metrics', {})
        self.liquid_weight = self.quality_weights.get('liquid_reduction_weight', 0.4)
        self.contaminant_weight = self.quality_weights.get('contaminant_reduction_weight', 0.3)
        self.smoothness_weight = self.quality_weights.get('smoothness_weight', 0.2)
        self.safety_weight = self.quality_weights.get('safety_weight', 0.1)
    
    def filter_demos(self, demos: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Filter demonstrations and return weights."""
        logger.info(f"Filtering {len(demos)} demonstrations...")
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(demos)
        
        # Filter by episode length
        valid_indices = self._filter_by_length(demos)
        valid_scores = quality_scores[valid_indices]
        valid_demos = [demos[i] for i in valid_indices]
        
        logger.info(f"After length filtering: {len(valid_demos)} demos")
        
        # Apply filtering method
        if self.method == 'gmm':
            weights = self._gmm_filtering(valid_scores)
        elif self.method == 'percentile':
            weights = self._percentile_filtering(valid_scores)
        else:
            raise ValueError(f"Unknown filtering method: {self.method}")
        
        # Filter out low-weight demos
        keep_indices = weights > 0.1  # Keep demos with weight > 0.1
        filtered_demos = [valid_demos[i] for i in range(len(valid_demos)) if keep_indices[i]]
        filtered_weights = weights[keep_indices]
        
        logger.info(f"After quality filtering: {len(filtered_demos)} demos")
        
        return filtered_demos, filtered_weights
    
    def _calculate_quality_scores(self, demos: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate quality scores for demonstrations."""
        scores = []
        
        for demo in demos:
            # Liquid reduction score
            liquid_scores = self._calculate_liquid_reduction_score(demo)
            
            # Contaminant reduction score
            contaminant_scores = self._calculate_contaminant_reduction_score(demo)
            
            # Smoothness score
            smoothness_scores = self._calculate_smoothness_score(demo)
            
            # Safety score
            safety_scores = self._calculate_safety_score(demo)
            
            # Weighted combination
            total_score = (
                self.liquid_weight * liquid_scores +
                self.contaminant_weight * contaminant_scores +
                self.smoothness_weight * smoothness_scores +
                self.safety_weight * safety_scores
            )
            
            scores.append(total_score)
        
        return np.array(scores)
    
    def _calculate_liquid_reduction_score(self, demo: Dict[str, Any]) -> float:
        """Calculate liquid reduction score."""
        infos = demo['infos']
        if not infos:
            return 0.0
        
        # Get initial and final liquid mass
        initial_liquid = infos[0].get('liquid_mass_remaining', 1.0)
        final_liquid = infos[-1].get('liquid_mass_remaining', 1.0)
        
        # Calculate reduction
        reduction = max(0, initial_liquid - final_liquid)
        reduction_rate = reduction / max(initial_liquid, 1e-6)
        
        return min(reduction_rate, 1.0)
    
    def _calculate_contaminant_reduction_score(self, demo: Dict[str, Any]) -> float:
        """Calculate contaminant reduction score."""
        infos = demo['infos']
        if not infos:
            return 0.0
        
        # Get initial and final contaminant mass
        initial_contaminant = infos[0].get('contaminant_mass_remaining', 1.0)
        final_contaminant = infos[-1].get('contaminant_mass_remaining', 1.0)
        
        # Calculate reduction
        reduction = max(0, initial_contaminant - final_contaminant)
        reduction_rate = reduction / max(initial_contaminant, 1e-6)
        
        return min(reduction_rate, 1.0)
    
    def _calculate_smoothness_score(self, demo: Dict[str, Any]) -> float:
        """Calculate action smoothness score."""
        actions = demo['actions']
        if len(actions) < 2:
            return 0.0
        
        # Calculate action differences
        action_diffs = np.diff(actions, axis=0)
        action_magnitudes = np.linalg.norm(action_diffs, axis=1)
        
        # Smoothness is inverse of action magnitude
        smoothness = 1.0 / (1.0 + np.mean(action_magnitudes))
        
        return smoothness
    
    def _calculate_safety_score(self, demo: Dict[str, Any]) -> float:
        """Calculate safety score."""
        infos = demo['infos']
        if not infos:
            return 1.0
        
        # Count safety violations
        collisions = sum(info.get('collisions', 0) for info in infos)
        safety_violations = sum(info.get('safety_violation', 0) for info in infos)
        
        # Safety score decreases with violations
        safety_score = 1.0 / (1.0 + collisions + safety_violations)
        
        return safety_score
    
    def _filter_by_length(self, demos: List[Dict[str, Any]]) -> np.ndarray:
        """Filter demos by episode length."""
        lengths = [demo['episode_length'] for demo in demos]
        valid_indices = np.array([
            self.min_episode_length <= length <= self.max_episode_length
            for length in lengths
        ])
        
        return valid_indices
    
    def _gmm_filtering(self, scores: np.ndarray) -> np.ndarray:
        """Filter using Gaussian Mixture Model."""
        if len(scores) < self.gmm_components:
            # Not enough data for GMM
            return np.ones(len(scores))
        
        # Fit GMM
        gmm = GaussianMixture(n_components=self.gmm_components, random_state=42)
        gmm.fit(scores.reshape(-1, 1))
        
        # Get component assignments
        assignments = gmm.predict(scores.reshape(-1, 1))
        
        # Calculate weights based on component probabilities
        weights = np.zeros(len(scores))
        for i in range(self.gmm_components):
            component_mask = assignments == i
            component_scores = scores[component_mask]
            
            if len(component_scores) > 0:
                # Weight by component probability and score quality
                component_weights = gmm.predict_proba(scores.reshape(-1, 1))[:, i]
                component_weights[component_mask] *= (component_scores / np.max(component_scores))
                weights += component_weights
        
        # Normalize weights
        weights = weights / np.max(weights)
        
        return weights
    
    def _percentile_filtering(self, scores: np.ndarray) -> np.ndarray:
        """Filter using percentile threshold."""
        threshold = np.percentile(scores, (1 - self.percentile_threshold) * 100)
        
        # Binary weights based on threshold
        weights = (scores >= threshold).astype(float)
        
        # Scale weights by score quality
        weights = weights * (scores / np.max(scores))
        
        return weights


class DemoWeighting:
    """
    Weight demonstrations for weighted behavior cloning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weighting_method = config.get('weighting_method', 'quality_based')
        self.temperature = config.get('temperature', 1.0)
        self.min_weight = config.get('min_weight', 0.01)
        self.max_weight = config.get('max_weight', 10.0)
    
    def calculate_weights(
        self, 
        demos: List[Dict[str, Any]], 
        quality_scores: np.ndarray
    ) -> np.ndarray:
        """Calculate weights for demonstrations."""
        if self.weighting_method == 'quality_based':
            weights = self._quality_based_weights(quality_scores)
        elif self.weighting_method == 'uniform':
            weights = np.ones(len(demos))
        elif self.weighting_method == 'length_based':
            weights = self._length_based_weights(demos)
        elif self.weighting_method == 'reward_based':
            weights = self._reward_based_weights(demos)
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting_method}")
        
        # Apply temperature scaling
        weights = np.power(weights, 1.0 / self.temperature)
        
        # Clip weights
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        
        return weights
    
    def _quality_based_weights(self, quality_scores: np.ndarray) -> np.ndarray:
        """Calculate weights based on quality scores."""
        # Softmax weighting
        exp_scores = np.exp(quality_scores / self.temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        return weights
    
    def _length_based_weights(self, demos: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate weights based on episode length."""
        lengths = np.array([demo['episode_length'] for demo in demos])
        
        # Longer episodes get higher weights
        weights = lengths / np.mean(lengths)
        
        return weights
    
    def _reward_based_weights(self, demos: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate weights based on episode rewards."""
        rewards = np.array([np.sum(demo['rewards']) for demo in demos])
        
        # Shift rewards to be positive
        rewards = rewards - np.min(rewards) + 1e-6
        
        # Weight by reward magnitude
        weights = rewards / np.mean(rewards)
        
        return weights


class DemoProcessor:
    """
    Complete demo processing pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.filter = DemoFilter(config.get('filtering', {}))
        self.weighting = DemoWeighting(config.get('weighting', {}))
    
    def process_demos(
        self, 
        demos: List[Dict[str, Any]], 
        output_dir: str
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Process demonstrations: filter and weight."""
        logger.info(f"Processing {len(demos)} demonstrations...")
        
        # Filter demonstrations
        filtered_demos, quality_scores = self.filter.filter_demos(demos)
        
        # Calculate weights
        weights = self.weighting.calculate_weights(filtered_demos, quality_scores)
        
        # Save processed data
        self._save_processed_demos(filtered_demos, weights, quality_scores, output_dir)
        
        # Generate statistics
        self._generate_statistics(filtered_demos, weights, quality_scores, output_dir)
        
        logger.info(f"Processed {len(filtered_demos)} demonstrations")
        logger.info(f"Weight statistics: mean={np.mean(weights):.3f}, "
                   f"std={np.std(weights):.3f}, min={np.min(weights):.3f}, max={np.max(weights):.3f}")
        
        return filtered_demos, weights
    
    def _save_processed_demos(
        self, 
        demos: List[Dict[str, Any]], 
        weights: np.ndarray, 
        quality_scores: np.ndarray, 
        output_dir: str
    ) -> None:
        """Save processed demonstrations."""
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        # Save weights
        weights_file = output_dir / "weights.npz"
        np.savez_compressed(weights_file, weights=weights, quality_scores=quality_scores)
        
        # Save filtered demos
        demos_file = output_dir / "filtered_demos.npz"
        np.savez_compressed(demos_file, demos=demos)
        
        logger.info(f"Saved processed demos to {output_dir}")
    
    def _generate_statistics(
        self, 
        demos: List[Dict[str, Any]], 
        weights: np.ndarray, 
        quality_scores: np.ndarray, 
        output_dir: str
    ) -> None:
        """Generate processing statistics."""
        stats = {
            'num_original_demos': len(demos),
            'num_filtered_demos': len(demos),
            'weight_statistics': {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'median': float(np.median(weights))
            },
            'quality_statistics': {
                'mean': float(np.mean(quality_scores)),
                'std': float(np.std(quality_scores)),
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores)),
                'median': float(np.median(quality_scores))
            },
            'episode_lengths': {
                'mean': float(np.mean([d['episode_length'] for d in demos])),
                'std': float(np.std([d['episode_length'] for d in demos])),
                'min': int(np.min([d['episode_length'] for d in demos])),
                'max': int(np.max([d['episode_length'] for d in demos]))
            }
        }
        
        # Save statistics
        stats_file = Path(output_dir) / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing statistics saved to {stats_file}")


def process_demos(
    demos: List[Dict[str, Any]], 
    config: Dict[str, Any], 
    output_dir: str
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Process demonstrations using the complete pipeline."""
    processor = DemoProcessor(config)
    return processor.process_demos(demos, output_dir)
