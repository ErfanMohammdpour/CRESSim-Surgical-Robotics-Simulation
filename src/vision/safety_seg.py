"""
Safety segmentation network for visual safety shield.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UNetTiny(nn.Module):
    """Tiny U-Net for safety segmentation."""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (64, 64),
        num_classes: int = 3,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = self._conv_block(3, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)
        
        # Output layers
        self.segmentation_head = nn.Conv2d(base_channels, num_classes, 1)
        self.distance_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"UNetTiny created: {input_size} -> {num_classes} classes")
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))
        
        # Decoder
        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[2:]), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:]), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:]), e1], dim=1))
        
        # Outputs
        segmentation = self.segmentation_head(d1)
        distance = self.distance_head(d1)
        
        return segmentation, distance


class SafetySegmentation(nn.Module):
    """Safety segmentation network with distance estimation."""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (64, 64),
        num_classes: int = 3,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Create U-Net backbone
        self.unet = UNetTiny(input_size, num_classes, base_channels)
        
        logger.info(f"SafetySegmentation created: {input_size} -> {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.unet(x)
    
    def predict_safety(self, x: torch.Tensor) -> Dict[str, Any]:
        """Predict safety information."""
        with torch.no_grad():
            segmentation, distance = self.forward(x)
            
            # Get dangerous areas (class 2)
            dangerous_mask = (segmentation.argmax(dim=1) == 2).float()
            dangerous_ratio = dangerous_mask.mean(dim=(1, 2))
            
            # Determine safety
            is_safe = (distance.squeeze() > 0.1) & (dangerous_ratio < 0.1)
            
            return {
                'segmentation': segmentation,
                'distance': distance,
                'dangerous_mask': dangerous_mask,
                'dangerous_ratio': dangerous_ratio,
                'is_safe': is_safe
            }


class SafetyShield(nn.Module):
    """Safety shield for action projection."""
    
    def __init__(
        self,
        safety_net: SafetySegmentation,
        d_safe: float = 0.05,
        d_warning: float = 0.1,
        scaling_factor: float = 0.5,
        max_scaling: float = 0.1
    ):
        super().__init__()
        
        self.safety_net = safety_net
        self.d_safe = d_safe
        self.d_warning = d_warning
        self.scaling_factor = scaling_factor
        self.max_scaling = max_scaling
        
        self.violation_count = 0
        self.consecutive_violations = 0
        
        logger.info(f"SafetyShield created: d_safe={d_safe}, d_warning={d_warning}")
    
    def forward(
        self, 
        image: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Project action based on safety assessment."""
        # Get safety information
        safety_info = self.safety_net.predict_safety(image)
        
        distance = safety_info['distance'].item()
        dangerous_ratio = safety_info['dangerous_ratio'].item()
        is_safe = safety_info['is_safe'].item()
        
        # Determine safety level
        if distance < self.d_safe or dangerous_ratio > 0.2:
            safety_level = 'critical'
        elif distance < self.d_warning or dangerous_ratio > 0.1:
            safety_level = 'warning'
        else:
            safety_level = 'safe'
        
        # Project action based on safety level
        if safety_level == 'critical':
            projected_action = torch.zeros_like(action)
            self.violation_count += 1
            self.consecutive_violations += 1
        elif safety_level == 'warning':
            scale = max(self.max_scaling, distance / self.d_warning)
            projected_action = action * scale
            self.violation_count += 1
            self.consecutive_violations += 1
        else:
            projected_action = action
            self.consecutive_violations = 0
        
        # Create safety info
        safety_info.update({
            'safety_level': safety_level,
            'violation_count': self.violation_count,
            'consecutive_violations': self.consecutive_violations,
            'action_scale': projected_action.norm().item() / action.norm().item() if action.norm() > 0 else 1.0
        })
        
        return projected_action, safety_info
    
    def reset_violations(self):
        """Reset violation counters."""
        self.violation_count = 0
        self.consecutive_violations = 0


def create_safety_network(
    input_size: Tuple[int, int] = (64, 64),
    num_classes: int = 3,
    base_channels: int = 32
) -> SafetySegmentation:
    """Create safety segmentation network."""
    return SafetySegmentation(input_size, num_classes, base_channels)