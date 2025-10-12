"""
Behavior Cloning trainer for imitation learning.
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from tqdm import tqdm
import json
from datetime import datetime

from ..utils.gpu import get_device, optimize_for_gpu
from ..vision.encoders import CNNEncoder
from ..utils.log import setup_logging

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
        logger.info(f"Loading demos from {self.demos_path}")
        
        with h5py.File(self.demos_path, 'r') as f:
            episode_keys = [key for key in f.keys() if key.startswith('episode_')]
            
            for episode_key in episode_keys:
                episode = f[episode_key]
                obs = episode['observations'][:]
                actions = episode['actions'][:]
                
                # Resize images if needed
                if obs.shape[1:3] != self.image_size:
                    obs_resized = []
                    for img in obs:
                        img_resized = torch.nn.functional.interpolate(
                            torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float(),
                            size=self.image_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).permute(1, 2, 0).numpy()
                        obs_resized.append(img_resized)
                    obs = np.array(obs_resized)
                
                self.observations.extend(obs)
                self.actions.extend(actions)
        
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        
        logger.info(f"Loaded {len(self.observations)} samples")
        logger.info(f"Observation shape: {self.observations.shape}")
        logger.info(f"Action shape: {self.actions.shape}")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = torch.tensor(self.observations[idx], dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
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
    """Behavior Cloning trainer."""
    
    def __init__(self, config: Dict[str, Any], demos_dir: str, output_dir: str):
        self.config = config
        self.demos_dir = demos_dir
        self.output_dir = output_dir
        
        # Setup device
        self.device = get_device()
        optimize_for_gpu()
        
        # Setup logging
        self.log_dir = Path(output_dir) / "logs" / "il"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.log_dir)
        
        # Load data
        self.dataset = DemoDataset(
            demos_path=Path(demos_dir) / "demos.h5",
            image_size=tuple(config.get("image_size", [128, 128]))
        )
        
        # Create data loader
        batch_size = config.get("batch_size", 64)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Create model
        self._create_model()
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5)
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"BCTrainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
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
            action_dim=5  # [dx, dy, dz, dyaw, suction_toggle]
        )
        
        self.model = self.model.to(self.device)
    
    def train(self, dagger: bool = False):
        """Train the behavior cloning model."""
        logger.info("Starting behavior cloning training...")
        
        num_epochs = self.config.get("num_epochs", 100)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self._train_epoch()
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {train_loss:.6f}")
            
            # Save checkpoint
            if epoch % 50 == 0 or train_loss < self.best_loss:
                self._save_checkpoint(epoch, train_loss)
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
        
        # Save final model
        self._save_checkpoint(epoch, train_loss, is_final=True)
        logger.info("Training completed!")
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (obs, actions) in enumerate(tqdm(self.dataloader, desc=f"Epoch {self.epoch}")):
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_actions = self.model(obs)
            loss = self.criterion(pred_actions, actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(self, epoch: int, loss: float, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        if is_final:
            checkpoint_path = Path(self.output_dir) / "best_il_model.pth"
        else:
            checkpoint_path = Path(self.output_dir) / f"il_checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained model."""
        self.model.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for obs, actions in self.dataloader:
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                
                pred_actions = self.model(obs)
                loss = self.criterion(pred_actions, actions)
                
                total_loss += loss.item() * obs.size(0)
                num_samples += obs.size(0)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        return {
            'mse_loss': avg_loss,
            'rmse_loss': np.sqrt(avg_loss),
            'num_samples': num_samples
        }


if __name__ == "__main__":
    # Test BC trainer
    config = {
        "image_size": [128, 128],
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "hidden_dim": 256,
        "cnn_encoder": {
            "channels": [32, 64, 128, 256],
            "kernel_sizes": [3, 3, 3, 3],
            "strides": [2, 2, 2, 2],
            "padding": [1, 1, 1, 1]
        }
    }
    
    trainer = BCTrainer(
        config=config,
        demos_dir="data/demos",
        output_dir="data/checkpoints"
    )
    
    trainer.train()
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")