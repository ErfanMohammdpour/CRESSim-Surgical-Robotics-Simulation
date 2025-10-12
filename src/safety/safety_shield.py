"""
Visual safety shield for action projection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SafetyShield:
    """Visual safety shield for action projection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.d_safe = config.get("d_safe", 0.05)  # Safe distance in meters
        self.segmentation_threshold = config.get("segmentation_threshold", 0.5)
        self.projection_scaling_factor = config.get("projection_scaling_factor", 0.8)
        self.cooldown_steps = config.get("cooldown_steps", 10)
        
        # Safety state
        self.violation_count = 0
        self.cooldown_counter = 0
        
        logger.info(f"SafetyShield initialized with d_safe={self.d_safe}")
    
    def apply_shield(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply safety shield to action."""
        # Check if in cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return action
        
        # Estimate proximity from observation
        proximity = self._estimate_proximity(observation)
        
        # Check safety constraint
        if proximity < self.d_safe:
            # Safety violation detected
            self.violation_count += 1
            self.cooldown_counter = self.cooldown_steps
            
            # Project action to safe region
            safe_action = self._project_action(action, proximity)
            
            logger.warning(f"Safety violation detected! Proximity: {proximity:.3f}, Action projected")
            return safe_action
        
        return action
    
    def _estimate_proximity(self, observation: np.ndarray) -> float:
        """Estimate proximity to obstacles from observation."""
        # Simple heuristic: analyze image edges and color changes
        if len(observation.shape) == 4:  # Batch dimension
            obs = observation[0]  # Take first observation
        else:
            obs = observation
        
        # Convert to grayscale if needed
        if obs.shape[-1] == 3:  # RGB
            gray = np.mean(obs, axis=-1)
        else:
            gray = obs
        
        # Calculate edge density (simple edge detection)
        edges = self._detect_edges(gray)
        edge_density = np.mean(edges)
        
        # Convert edge density to proximity estimate
        # Higher edge density = closer to obstacles
        proximity = min(edge_density * 0.1, 0.2)  # Cap at 0.2 meters
        
        return proximity
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Simple edge detection using Sobel operator."""
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply convolution
        edges_x = self._convolve2d(image, sobel_x)
        edges_y = self._convolve2d(image, sobel_y)
        
        # Calculate magnitude
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        return edges
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution implementation."""
        h, w = image.shape
        kh, kw = kernel.shape
        
        # Pad image
        pad_h = kh // 2
        pad_w = kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        # Apply convolution
        result = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return result
    
    def _project_action(self, action: np.ndarray, proximity: float) -> np.ndarray:
        """Project action to safe region."""
        # Scale down action based on proximity
        scale_factor = self.projection_scaling_factor * (proximity / self.d_safe)
        scale_factor = max(scale_factor, 0.1)  # Minimum scale
        
        # Apply scaling to position components (first 3 elements)
        safe_action = action.copy()
        safe_action[:3] *= scale_factor
        
        # Keep rotation and suction unchanged
        # safe_action[3:] remains the same
        
        return safe_action
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        return {
            "violation_count": self.violation_count,
            "cooldown_counter": self.cooldown_counter,
            "d_safe": self.d_safe,
            "projection_scaling_factor": self.projection_scaling_factor
        }
    
    def reset_stats(self):
        """Reset safety statistics."""
        self.violation_count = 0
        self.cooldown_counter = 0


class SegmentationHead(nn.Module):
    """Segmentation head for safety shield."""
    
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation."""
        return self.segmentation_head(x)


class ProximityEstimator(nn.Module):
    """Proximity estimation head for safety shield."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.proximity_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for proximity estimation."""
        return self.proximity_head(x)


if __name__ == "__main__":
    # Test safety shield
    config = {
        "d_safe": 0.05,
        "segmentation_threshold": 0.5,
        "projection_scaling_factor": 0.8,
        "cooldown_steps": 10
    }
    
    shield = SafetyShield(config)
    
    # Test with dummy observation and action
    obs = np.random.rand(128, 128, 3)
    action = np.array([0.1, 0.2, 0.05, 0.1, 1.0])
    
    safe_action = shield.apply_shield(obs, action)
    print(f"Original action: {action}")
    print(f"Safe action: {safe_action}")
    
    stats = shield.get_safety_stats()
    print(f"Safety stats: {stats}")

