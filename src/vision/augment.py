"""
Data augmentation utilities for vision training.
"""

import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import cv2


class ImageAugmentation:
    """
    Image augmentation pipeline using Albumentations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.augment_prob = config.get('augment_prob', 0.5)
        
        # Training augmentations
        self.train_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=config.get('brightness_limit', 0.2),
                contrast_limit=config.get('contrast_limit', 0.2),
                p=config.get('brightness_prob', 0.5)
            ),
            A.HueSaturationValue(
                hue_shift_limit=config.get('hue_shift_limit', 20),
                sat_shift_limit=config.get('sat_shift_limit', 30),
                val_shift_limit=config.get('val_shift_limit', 20),
                p=config.get('hsv_prob', 0.5)
            ),
            A.RandomGamma(
                gamma_limit=config.get('gamma_limit', (80, 120)),
                p=config.get('gamma_prob', 0.3)
            ),
            A.GaussNoise(
                var_limit=config.get('noise_var_limit', (10.0, 50.0)),
                p=config.get('noise_prob', 0.3)
            ),
            A.Blur(
                blur_limit=config.get('blur_limit', 3),
                p=config.get('blur_prob', 0.3)
            ),
            A.MotionBlur(
                blur_limit=config.get('motion_blur_limit', 3),
                p=config.get('motion_blur_prob', 0.2)
            ),
            A.RandomRotate90(p=config.get('rotate90_prob', 0.1)),
            A.HorizontalFlip(p=config.get('hflip_prob', 0.1)),
            A.VerticalFlip(p=config.get('vflip_prob', 0.05)),
            A.ShiftScaleRotate(
                shift_limit=config.get('shift_limit', 0.1),
                scale_limit=config.get('scale_limit', 0.1),
                rotate_limit=config.get('rotate_limit', 15),
                p=config.get('ssr_prob', 0.3)
            ),
            A.CoarseDropout(
                max_holes=config.get('dropout_holes', 8),
                max_height=config.get('dropout_height', 8),
                max_width=config.get('dropout_width', 8),
                p=config.get('dropout_prob', 0.1)
            ),
        ])
        
        # Validation augmentations (minimal)
        self.val_transform = A.Compose([
            A.Normalize(
                mean=config.get('normalize_mean', [0.485, 0.456, 0.406]),
                std=config.get('normalize_std', [0.229, 0.224, 0.225])
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray, training: bool = True) -> torch.Tensor:
        """Apply augmentation to image."""
        if training and np.random.random() < self.augment_prob:
            augmented = self.train_transform(image=image)['image']
        else:
            augmented = self.val_transform(image=image)['image']
        
        return augmented


class SegmentationAugmentation:
    """
    Augmentation pipeline for segmentation tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.augment_prob = config.get('augment_prob', 0.5)
        
        # Training augmentations with mask support
        self.train_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.3
            ),
            A.Blur(
                blur_limit=3,
                p=0.3
            ),
            A.RandomRotate90(p=0.1),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.05),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.3
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.2
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.2
            ),
        ])
        
        # Validation augmentations
        self.val_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to image and mask."""
        if training and np.random.random() < self.augment_prob:
            augmented = self.train_transform(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = augmented['mask']
        else:
            augmented = self.val_transform(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = torch.from_numpy(mask).long()
        
        return image_aug, mask_aug


class DomainAdaptationAugmentation:
    """
    Augmentation for domain adaptation between synthetic and real data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Style transfer augmentations
        self.style_transform = A.Compose([
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.8
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),
                p=0.5
            ),
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.GaussNoise(
                var_limit=(5.0, 25.0),
                p=0.3
            ),
            A.Blur(
                blur_limit=2,
                p=0.2
            ),
        ])
        
        # Geometric augmentations
        self.geometric_transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.8
            ),
            A.Perspective(
                scale=(0.05, 0.1),
                p=0.3
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.2
            ),
        ])
    
    def __call__(self, image: np.ndarray, domain: str = "synthetic") -> np.ndarray:
        """Apply domain adaptation augmentation."""
        # Apply style transfer for domain adaptation
        if domain == "synthetic":
            # Make synthetic look more real
            image = self.style_transform(image=image)['image']
        else:
            # Make real look more synthetic
            image = self.geometric_transform(image=image)['image']
        
        return image


class MixUpAugmentation:
    """
    MixUp augmentation for improved generalization.
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(
        self, 
        image1: torch.Tensor, 
        image2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # Mix labels
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label, lam


class CutMixAugmentation:
    """
    CutMix augmentation for improved generalization.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(
        self, 
        image1: torch.Tensor, 
        image2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get image dimensions
        H, W = image1.shape[-2:]
        
        # Calculate cut region
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Calculate bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_image = image1.clone()
        mixed_image[..., bby1:bby2, bbx1:bbx2] = image2[..., bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Mix labels
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label, lam


def create_augmentation_pipeline(
    task_type: str,
    config: Dict[str, Any]
) -> Any:
    """Create augmentation pipeline based on task type."""
    if task_type == "classification":
        return ImageAugmentation(config)
    elif task_type == "segmentation":
        return SegmentationAugmentation(config)
    elif task_type == "domain_adaptation":
        return DomainAdaptationAugmentation(config)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def apply_test_time_augmentation(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_augmentations: int = 8
) -> torch.Tensor:
    """Apply test time augmentation for better predictions."""
    model.eval()
    
    # Original prediction
    with torch.no_grad():
        pred = model(image)
    
    # Augmented predictions
    augmented_preds = []
    for _ in range(num_augmentations - 1):
        # Apply random augmentation
        aug_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.25),
        ])
        
        # Convert to numpy for augmentation
        if image.dim() == 4:  # Batch
            image_np = image[0].permute(1, 2, 0).cpu().numpy()
        else:  # Single image
            image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Apply augmentation
        augmented = aug_transform(image=image_np)['image']
        
        # Convert back to tensor
        augmented_tensor = torch.from_numpy(augmented).permute(2, 0, 1).unsqueeze(0)
        if image.is_cuda:
            augmented_tensor = augmented_tensor.cuda()
        
        # Get prediction
        with torch.no_grad():
            aug_pred = model(augmented_tensor)
            augmented_preds.append(aug_pred)
    
    # Average predictions
    all_preds = [pred] + augmented_preds
    final_pred = torch.stack(all_preds).mean(dim=0)
    
    return final_pred
