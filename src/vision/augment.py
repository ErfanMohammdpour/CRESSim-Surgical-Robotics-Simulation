"""
Data augmentation utilities for vision training.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_augmentation_pipeline(
    image_size: Tuple[int, int] = (64, 64),
    config: Dict[str, Any] = None
) -> A.Compose:
    """Create augmentation pipeline for training."""
    if config is None:
        config = {
            'brightness_limit': 0.3,
            'contrast_limit': 0.2,
            'blur_limit': 1.5,
            'cutout_size': (8, 24),
            'cutout_prob': 0.5
        }
    
    transforms = [
        # Resize
        A.Resize(image_size[0], image_size[1]),
        
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=config.get('brightness_limit', 0.3),
            contrast_limit=config.get('contrast_limit', 0.2),
            p=0.8
        ),
        
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        
        # Blur augmentation
        A.OneOf([
            A.GaussianBlur(blur_limit=config.get('blur_limit', 1.5), p=0.5),
            A.MotionBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        
        # Cutout augmentation
        A.CoarseDropout(
            max_holes=1,
            max_height=config.get('cutout_size', (8, 24))[1],
            max_width=config.get('cutout_size', (8, 24))[1],
            min_holes=1,
            min_height=config.get('cutout_size', (8, 24))[0],
            min_width=config.get('cutout_size', (8, 24))[0],
            fill_value=0,
            p=config.get('cutout_prob', 0.5)
        ),
        
        # Convert to tensor
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def create_validation_pipeline(image_size: Tuple[int, int] = (64, 64)) -> A.Compose:
    """Create validation pipeline (no augmentation)."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        ToTensorV2()
    ])


def create_test_pipeline(image_size: Tuple[int, int] = (64, 64)) -> A.Compose:
    """Create test pipeline (minimal augmentation)."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])