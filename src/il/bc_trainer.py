"""
Behavior Cloning trainer for imitation learning with GPU support and AMP.
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
from datetime import datetime

from ..vision.encoders import CNNEncoder
from ..utils.device import (
    get_device_from_config, log_device_info, optimize_for_gpu, 
    create_optimizer, create_grad_scaler, get_dataloader_kwargs,
    handle_oom_error, warmup_model, log_memory_usage, amp_enabled
)

logger = logging.getLogger(__name__)


class DemoDataset(Dataset):
    """Dataset for loading demonstration data."""
    
    def __init__(self, demos_path: str, image_size: Tuple[int, int] = (128, 128)):
        self.demos_path = demos_path
        self.image_size = image_size
        self.observations = []
        self.actions = []
        
        self._load_data()
    
    def _load_data(self):
        """Load demonstration data from HDF5 file."""
        if not os.path.exists(self.demos_path):
            raise FileNotFoundError(f"Demo file not found: {self.demos_path}")
        
        with h5py.File(self.demos_path, 'r') as f:
            # Load observations
            obs_data = f['observations'][:]
            self.observations = obs_data.astype(np.float32)
            
            # Load actions
            action_data = f['actions'][:]
            self.actions = action_data.astype(np.float32)
        
        logger.info(f"Loaded {len(self.observations)} demonstration samples")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        return obs, action


class PolicyNetwork(nn.Module):
    """Policy network for behavior cloning."""
    
    def __init__(self, encoder: nn.Module, hidden_dim: int = 512, action_dim: int = 5):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, obs):
        features = self.encoder(obs)
        action = self.policy_head(features)
        return action


class BCTrainer:
    """Behavior Cloning trainer with GPU support and AMP."""
    
    def __init__(self, config: Dict[str, Any], demos_dir: str, output_dir: str):
        self.config = config
        self.demos_dir = demos_dir
        self.output_dir = output_dir
        
        # Setup device and logging
        self.device = get_device_from_config(config)
        log_device_info(self.device, config)
        
        # Setup logging
        self.log_dir = Path(output_dir) / "logs" / "il"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.dataset = DemoDataset(
            demos_path=Path(demos_dir) / "demos.h5",
            image_size=tuple(config.get("image_size", [128, 128]))
        )
        
        # Create data loader with GPU optimizations
        batch_size = config.get("batch_size", 64)
        dataloader_kwargs = get_dataloader_kwargs(config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            **dataloader_kwargs
        )
        
        # Create model
        self._create_model()
        
        # Setup optimizer and scaler
        self.optimizer = create_optimizer(self.model, config)
        self.scaler = create_grad_scaler(config)
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"BCTrainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"AMP enabled: {amp_enabled(config)}")
    
    def _create_model(self):
        """Create the policy model."""
        # Create encoder
        encoder_config = self.config.get("cnn_encoder", {})
        self.encoder = CNNEncoder(
            input_channels=3,
            output_dim=self.config.get("hidden_dim", 512),
            channels=encoder_config.get("channels", [32, 64, 128, 256]),
            kernel_sizes=encoder_config.get("kernel_sizes", [3, 3, 3, 3]),
            strides=encoder_config.get("strides", [2, 2, 2, 2]),
            padding=encoder_config.get("padding", [1, 1, 1, 1])
        )
        
        # Create policy network
        self.model = PolicyNetwork(
            encoder=self.encoder,
            hidden_dim=self.config.get("hidden_dim", 512),
            action_dim=self.config.get("action_dim", 5)
        )
        
        # Optimize for GPU
        self.model = optimize_for_gpu(self.model, self.config)
        
        # Warmup model
        warmup_model(self.model, self.config, (1, 3, 128, 128))
    
    def train_epoch(self) -> float:
        """Train for one epoch with AMP support."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (obs, actions) in enumerate(pbar):
            try:
                # Move to device
                obs = obs.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with AMP
                if amp_enabled(self.config):
                    with torch.autocast("cuda"):
                        pred_actions = self.model(obs)
                        loss = self.criterion(pred_actions, actions)
                    
                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred_actions = self.model(obs)
                    loss = self.criterion(pred_actions, actions)
                    
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
                    self.dataloader = DataLoader(
                        self.dataset,
                        batch_size=self.config.get("batch_size", 32),
                        shuffle=True,
                        **dataloader_kwargs
                    )
                    continue
                else:
                    raise e
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for obs, actions in self.dataloader:
                # Move to device
                obs = obs.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                
                # Forward pass with AMP
                if amp_enabled(self.config):
                    with torch.autocast("cuda"):
                        pred_actions = self.model(obs)
                        loss = self.criterion(pred_actions, actions)
                else:
                    pred_actions = self.model(obs)
                    loss = self.criterion(pred_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = Path(self.output_dir) / f"il_checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.output_dir) / "il_checkpoint.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with loss {loss:.4f}")
    
    def train(self, epochs: int = 50):
        """Train the model."""
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Check if best
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Log progress
            logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Best: {self.best_loss:.4f}"
            )
            
            # Log memory usage
            log_memory_usage(self.device, f"Epoch {epoch} End")
        
        logger.info(f"Training completed. Best validation loss: {self.best_loss:.4f}")


def train_il_model(config: Dict[str, Any], demos_dir: str, output_dir: str) -> str:
    """Train imitation learning model."""
    trainer = BCTrainer(config, demos_dir, output_dir)
    epochs = config.get("epochs", 50)
    trainer.train(epochs)
    
    # Return path to best checkpoint
    return str(Path(output_dir) / "il_checkpoint.pth")