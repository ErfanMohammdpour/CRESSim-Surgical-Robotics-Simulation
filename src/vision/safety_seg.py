"""
Safety segmentation network for visual safety shield.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np


class TinyUNet(nn.Module):
    """
    Tiny U-Net for safety segmentation.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        input_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.input_size = input_size
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)
        
        # Final classification
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder with skip connections
        dec3 = self.dec3(torch.cat([self.upsample(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))
        
        # Final classification
        output = self.final_conv(dec1)
        
        return output


class SafetySegmentation(nn.Module):
    """
    Safety segmentation network with distance estimation.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (64, 64),
        num_classes: int = 3,
        base_channels: int = 32,
        confidence_threshold: float = 0.8
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # Segmentation network
        self.segmentation_net = TinyUNet(
            input_channels=3,
            num_classes=num_classes,
            base_channels=base_channels,
            input_size=input_size
        )
        
        # Distance estimation head
        self.distance_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Get segmentation features
        seg_features = self.segmentation_net.enc1(x)
        seg_features = self.segmentation_net.enc2(self.segmentation_net.pool(seg_features))
        seg_features = self.segmentation_net.enc3(self.segmentation_net.pool(seg_features))
        
        # Segmentation output
        segmentation = self.segmentation_net(x)
        
        # Distance estimation
        distance = self.distance_head(seg_features)
        
        return segmentation, distance
    
    def predict_safety(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict safety information."""
        with torch.no_grad():
            segmentation, distance = self.forward(x)
            
            # Get dangerous areas (class 2)
            dangerous_mask = (segmentation.argmax(dim=1) == 2).float()
            
            # Calculate safety metrics
            dangerous_pixels = dangerous_mask.sum(dim=(1, 2))
            total_pixels = dangerous_mask.numel() // dangerous_mask.size(0)
            dangerous_ratio = dangerous_pixels / total_pixels
            
            # Safety status
            is_safe = (dangerous_ratio < 0.1) & (distance.squeeze() > 0.3)
            
            return {
                'segmentation': segmentation,
                'distance': distance,
                'dangerous_mask': dangerous_mask,
                'dangerous_ratio': dangerous_ratio,
                'is_safe': is_safe
            }


class SafetyShield(nn.Module):
    """
    Visual safety shield that projects actions based on safety predictions.
    """
    
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
        
        # Safety state
        self.violation_count = 0
        self.last_safety_state = True
    
    def forward(
        self, 
        image: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply safety shield to action."""
        # Get safety predictions
        safety_info = self.safety_net.predict_safety(image)
        
        # Extract safety metrics
        distance = safety_info['distance'].squeeze()
        dangerous_ratio = safety_info['dangerous_ratio']
        is_safe = safety_info['is_safe']
        
        # Determine safety level
        if distance < self.d_safe or dangerous_ratio > 0.2:
            safety_level = "critical"
        elif distance < self.d_warning or dangerous_ratio > 0.1:
            safety_level = "warning"
        else:
            safety_level = "safe"
        
        # Project action based on safety level
        if safety_level == "critical":
            # Emergency stop - zero action
            projected_action = torch.zeros_like(action)
            self.violation_count += 1
        elif safety_level == "warning":
            # Scale down action
            scale = max(self.max_scaling, self.scaling_factor)
            projected_action = action * scale
            self.violation_count += 1
        else:
            # Safe - no modification
            projected_action = action
            self.violation_count = 0
        
        # Update safety state
        self.last_safety_state = is_safe.item() if is_safe.numel() == 1 else is_safe[0].item()
        
        # Create safety info
        safety_info.update({
            'safety_level': safety_level,
            'projected_action': projected_action,
            'violation_count': self.violation_count,
            'action_scale': scale if safety_level == "warning" else 1.0
        })
        
        return projected_action, safety_info
    
    def reset(self):
        """Reset safety shield state."""
        self.violation_count = 0
        self.last_safety_state = True


def create_safety_network(config: Dict[str, Any]) -> SafetySegmentation:
    """Create safety segmentation network from config."""
    seg_config = config.get('segmentation', {})
    
    return SafetySegmentation(
        input_size=tuple(seg_config.get('input_size', [64, 64])),
        num_classes=seg_config.get('num_classes', 3),
        base_channels=32,
        confidence_threshold=seg_config.get('confidence_threshold', 0.8)
    )


def create_safety_shield(
    safety_net: SafetySegmentation,
    config: Dict[str, Any]
) -> SafetyShield:
    """Create safety shield from config."""
    action_config = config.get('action_projection', {})
    
    return SafetyShield(
        safety_net=safety_net,
        d_safe=config.get('d_safe', 0.05),
        d_warning=config.get('d_warning', 0.1),
        scaling_factor=action_config.get('scaling_factor', 0.5),
        max_scaling=action_config.get('max_scaling', 0.1)
    )


class SafetyLoss(nn.Module):
    """
    Loss function for safety segmentation training.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Cross entropy loss
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self, 
        segmentation: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate safety segmentation loss."""
        # Standard cross entropy
        ce_loss = self.ce_loss(segmentation, target)
        
        # Focal loss for hard examples
        probs = F.softmax(segmentation, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
        focal_loss = (focal_weight * F.cross_entropy(segmentation, target, reduction='none')).mean()
        
        return ce_loss + focal_loss
