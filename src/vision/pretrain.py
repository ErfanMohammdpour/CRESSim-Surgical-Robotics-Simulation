"""
Vision pretraining utilities with GPU support and AMP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .safety_seg import SafetySegmentation
from .augment import create_augmentation_pipeline
from ..utils.device import (
    get_device_from_config, log_device_info, optimize_for_gpu, 
    create_optimizer, create_grad_scaler, get_dataloader_kwargs,
    handle_oom_error, warmup_model, log_memory_usage, amp_enabled
)

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """Dataset for segmentation pretraining."""
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        image_size: Tuple[int, int] = (64, 64),
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # Load data
        self.samples = self._load_samples()
        
        # Create augmentation pipeline
        if augment:
            self.transform = create_augmentation_pipeline(image_size)
        else:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                ToTensorV2()
            ])
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self):
        """Load image and mask pairs."""
        samples = []
        
        # Look for images and masks
        image_dir = self.data_dir / "images"
        mask_dir = self.data_dir / "masks"
        
        if not image_dir.exists() or not mask_dir.exists():
            logger.warning(f"Data directories not found: {image_dir}, {mask_dir}")
            return samples
        
        # Get all image files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        for img_path in image_files:
            # Find corresponding mask
            mask_path = mask_dir / img_path.name
            
            if mask_path.exists():
                samples.append((img_path, mask_path))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        
        image = transformed["image"]
        mask = transformed["mask"]
        
        # Convert mask to class indices
        mask = (mask * 255).long()
        
        return image, mask


class SegmentationTrainer:
    """Segmentation trainer with GPU support and AMP."""
    
    def __init__(self, config: Dict[str, Any], data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Setup device and logging
        self.device = get_device_from_config(config)
        log_device_info(self.device, config)
        
        # Setup logging
        self.log_dir = Path(output_dir) / "logs" / "vision"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        self.train_dataset = SegmentationDataset(
            data_dir=Path(data_dir),
            split="train",
            image_size=tuple(config.get("image_size", [64, 64])),
            augment=True
        )
        
        self.val_dataset = SegmentationDataset(
            data_dir=Path(data_dir),
            split="val",
            image_size=tuple(config.get("image_size", [64, 64])),
            augment=False
        )
        
        # Create data loaders with GPU optimizations
        dataloader_kwargs = get_dataloader_kwargs(config)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.get("batch_size", 16),
            shuffle=True,
            **dataloader_kwargs
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.get("batch_size", 16),
            shuffle=False,
            **dataloader_kwargs
        )
        
        # Create model
        self.model = self._create_model()
        
        # Setup optimizer and scaler
        self.optimizer = create_optimizer(self.model, config)
        self.scaler = create_grad_scaler(config)
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.epoch = 0
        self.best_iou = 0.0
        
        logger.info(f"SegmentationTrainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"AMP enabled: {amp_enabled(config)}")
    
    def _create_model(self):
        """Create segmentation model."""
        model = SafetySegmentation(
            input_size=self.config.get("image_size", [64, 64]),
            num_classes=self.config.get("num_classes", 3)
        )
        
        # Optimize for GPU
        model = optimize_for_gpu(model, self.config)
        
        # Warmup model
        warmup_model(model, self.config, (1, 3, 64, 64))
        
        return model
    
    def train_epoch(self) -> float:
        """Train for one epoch with AMP support."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                # Move to device
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with AMP
                if amp_enabled(self.config):
                    with torch.autocast("cuda"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                    
                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
                
                # Log memory usage periodically
                if batch_idx % 50 == 0:
                    log_memory_usage(self.device, f"Batch {batch_idx}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM detected, handling...")
                    self.config = handle_oom_error(e, self.config)
                    # Recreate dataloader with smaller batch size
                    dataloader_kwargs = get_dataloader_kwargs(self.config)
                    self.train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=self.config.get("batch_size", 8),
                        shuffle=True,
                        **dataloader_kwargs
                    )
                    continue
                else:
                    raise e
        
        return total_loss / num_batches
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Metrics
        total_iou = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                # Move to device
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Forward pass with AMP
                if amp_enabled(self.config):
                    with torch.autocast("cuda"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate metrics
                pred_masks = torch.argmax(outputs, dim=1)
                
                # IoU calculation
                iou = self._calculate_iou(pred_masks, masks)
                total_iou += iou
                
                # Dice coefficient
                dice = self._calculate_dice(pred_masks, masks)
                total_dice += dice
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice
        }
    
    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Intersection over Union."""
        # Flatten tensors
        pred = pred.flatten()
        target = target.flatten()
        
        # Calculate intersection and union
        intersection = (pred == target).float().sum()
        union = len(pred)
        
        return (intersection / union).item()
    
    def _calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice coefficient."""
        # Flatten tensors
        pred = pred.flatten()
        target = target.flatten()
        
        # Calculate intersection
        intersection = (pred == target).float().sum()
        
        # Calculate Dice
        dice = (2 * intersection) / (len(pred) + len(target))
        
        return dice.item()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = Path(self.output_dir) / f"safety_checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.output_dir) / "safety_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with IoU {metrics['iou']:.4f}")
    
    def train(self, epochs: int = 20):
        """Train the segmentation model."""
        logger.info(f"Starting segmentation training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Check if best
            is_best = val_metrics['iou'] > self.best_iou
            if is_best:
                self.best_iou = val_metrics['iou']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log progress
            logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"IoU: {val_metrics['iou']:.4f}, "
                f"Dice: {val_metrics['dice']:.4f}, "
                f"Best IoU: {self.best_iou:.4f}"
            )
            
            # Log memory usage
            log_memory_usage(self.device, f"Epoch {epoch} End")
        
        logger.info(f"Training completed. Best IoU: {self.best_iou:.4f}")


def pretrain_safety_segmentation(config: Dict[str, Any], data_dir: str, output_dir: str) -> str:
    """Pretrain safety segmentation model."""
    trainer = SegmentationTrainer(config, data_dir, output_dir)
    epochs = config.get("epochs", 20)
    trainer.train(epochs)
    
    # Return path to best checkpoint
    return str(Path(output_dir) / "safety_best.pth")