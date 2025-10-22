"""
Demo filtering and weighting utilities.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DemoFilter:
    """Filter demonstrations based on quality criteria."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_filtering = config.get('quality_filtering', {})
        self.enabled = self.quality_filtering.get('enabled', True)
        
        # Quality thresholds
        self.min_liquid_reduction = self.quality_filtering.get('min_liquid_reduction', 0.3)
        self.min_contaminant_reduction = self.quality_filtering.get('min_contaminant_reduction', 0.2)
        self.max_collision_rate = self.quality_filtering.get('max_collision_rate', 0.1)
        self.max_safety_violations = self.quality_filtering.get('max_safety_violations', 0.05)
        
        logger.info(f"DemoFilter initialized: enabled={self.enabled}")
    
    def filter_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter episodes based on quality criteria."""
        if not self.enabled:
            return episodes
        
        filtered_episodes = []
        
        for episode in episodes:
            if self._is_episode_good(episode):
                filtered_episodes.append(episode)
        
        logger.info(f"Filtered {len(episodes)} -> {len(filtered_episodes)} episodes")
        return filtered_episodes
    
    def _is_episode_good(self, episode: Dict[str, Any]) -> bool:
        """Check if episode meets quality criteria."""
        # Get final episode info
        final_info = episode['infos'][-1]
        
        # Check liquid reduction
        liquid_reduction = final_info.get('liquid_reduction', 0)
        if liquid_reduction < self.min_liquid_reduction:
            return False
        
        # Check contaminant reduction
        contaminant_reduction = final_info.get('contaminant_reduction', 0)
        if contaminant_reduction < self.min_contaminant_reduction:
            return False
        
        # Check collision rate
        collision_count = final_info.get('collision_count', 0)
        collision_rate = collision_count / episode['length']
        if collision_rate > self.max_collision_rate:
            return False
        
        # Check safety violations
        safety_violations = sum(1 for info in episode['infos'] if info.get('safety_violations', 0) > 0)
        safety_violation_rate = safety_violations / episode['length']
        if safety_violation_rate > self.max_safety_violations:
            return False
        
        return True


class DemoWeighting:
    """Calculate weights for demonstration data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weighting = config.get('weighting', {})
        self.enabled = self.weighting.get('enabled', True)
        
        # Weighting parameters
        self.gmm_components = self.weighting.get('gmm_components', 2)
        self.features = self.weighting.get('features', [
            'liquid_reduction',
            'contaminant_reduction',
            'smoothness',
            'safety_compliance'
        ])
        self.feature_weights = self.weighting.get('weights', {
            'liquid_reduction': 0.4,
            'contaminant_reduction': 0.3,
            'smoothness': 0.2,
            'safety_compliance': 0.1
        })
        
        logger.info(f"DemoWeighting initialized: enabled={self.enabled}")
    
    def calculate_weights(self, episodes: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate weights for episodes."""
        if not self.enabled:
            return np.ones(len(episodes))
        
        # Extract features
        features = self._extract_features(episodes)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Apply feature weights
        weighted_features = features_normalized * np.array([
            self.feature_weights.get(feat, 1.0) for feat in self.features
        ])
        
        # Fit GMM
        gmm = GaussianMixture(n_components=self.gmm_components, random_state=42)
        gmm.fit(weighted_features)
        
        # Calculate weights (probability of "good" component)
        weights = gmm.predict_proba(weighted_features)[:, 1]
        
        logger.info(f"Calculated weights: mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")
        return weights
    
    def _extract_features(self, episodes: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from episodes."""
        features = []
        
        for episode in episodes:
            episode_features = []
            
            # Get final episode info
            final_info = episode['infos'][-1]
            
            # Liquid reduction
            if 'liquid_reduction' in self.features:
                liquid_reduction = final_info.get('liquid_reduction', 0)
                episode_features.append(liquid_reduction)
            
            # Contaminant reduction
            if 'contaminant_reduction' in self.features:
                contaminant_reduction = final_info.get('contaminant_reduction', 0)
                episode_features.append(contaminant_reduction)
            
            # Smoothness (action variance)
            if 'smoothness' in self.features:
                actions = episode['actions']
                smoothness = 1.0 / (1.0 + np.var(actions, axis=0).sum())
                episode_features.append(smoothness)
            
            # Safety compliance
            if 'safety_compliance' in self.features:
                safety_compliance = np.mean(episode['safety_flags'])
                episode_features.append(safety_compliance)
            
            features.append(episode_features)
        
        return np.array(features)


def filter_weight_demos(
    episodes: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Filter and weight demonstration episodes."""
    # Filter episodes
    filter_obj = DemoFilter(config)
    filtered_episodes = filter_obj.filter_episodes(episodes)
    
    # Calculate weights
    weighting_obj = DemoWeighting(config)
    weights = weighting_obj.calculate_weights(filtered_episodes)
    
    return filtered_episodes, weights
