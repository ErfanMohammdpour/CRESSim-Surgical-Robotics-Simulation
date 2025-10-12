"""
Behavior Cloning trainer for imitation learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import json

from ..vision.encoders import create_encoder
from ..utils.log import Logger
from ..utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


class DemoDataset(Dataset):
    """
    Dataset for loading demonstrations.
    """
    
    def __init__(
        self, 
        demos: List[Dict[str, Any]], 
        weights: np.ndarray,
        augment: bool = True
    ):
        self.demos = demos
        self.weights = weights
        self.augment = augment
        
        # Flatten all transitions
        self.transitions = []
        self.transition_weights = []
        
        for demo, weight in zip(demos, weights):
            episode_length = demo['episode_length']
            for i in range(episode_length):
                self.transitions.append({
                    'observation': demo['observations'][i],
                    'action': demo['actions'][i],
                    'reward': demo['rewards'][i],
                    'info': demo['infos'][i]
                })
                self.transition_weights.append(weight)
        
        self.transition_weights = np.array(self.transition_weights)
        logger.info(f"Created dataset with {len(self.transitions)} transitions")
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        transition = self.transitions[idx]
        weight = self.transition_weights[idx]
        
        # Extract observation components
        obs = transition['observation']
        image = torch.from_numpy(obs['image']).float() / 255.0
        aux = torch.from_numpy(obs['aux']).float()
        
        # Extract action
        action = torch.from_numpy(transition['action']).float()
        
        return {
            'image': image,
            'aux': aux,
            'action': action,
            'weight': weight
        }


class BCTrainer:
    """
    Behavior Cloning trainer.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        demos_dir: str, 
        output_dir: str
    ):
        self.config = config
        self.demos_dir = Path(demos_dir)
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.batch_size = config.get('batch_size', 64)
        self.num_epochs = config.get('num_epochs', 100)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.grad_clip_norm = config.get('grad_clip_norm', 1.0)
        
        # Model parameters
        self.model_config = config.get('model', {})
        self.hidden_dim = self.model_config.get('hidden_dim', 256)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        
        # Logger
        self.logger = Logger(str(self.output_dir), "bc_training")
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def _create_model(self) -> nn.Module:
        """Create the behavior cloning model."""
        # Create encoder
        encoder = create_encoder(self.model_config)
        
        # Add action head
        action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 5)  # 5D action space
        )
        
        # Combine encoder and action head
        model = nn.ModuleDict({
            'encoder': encoder,
            'action_head': action_head
        })
        
        return model.to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _create_criterion(self) -> nn.Module:
        """Create loss criterion."""
        return nn.MSELoss(reduction='none')
    
    def train(self, dagger: bool = False) -> None:
        """Train the behavior cloning model."""
        logger.info("Starting behavior cloning training...")
        
        # Load demonstrations
        demos, weights = self._load_demos()
        
        # Create dataset and dataloader
        dataset = DemoDataset(demos, weights, augment=True)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch(dataloader)
            
            # Logging
            self.logger.log_scalar('train/loss', epoch_loss, epoch)
            self.training_history.append(epoch_loss)
            
            # Validation
            if epoch % 10 == 0:
                val_loss = self._validate(dataloader)
                self.logger.log_scalar('val/loss', val_loss, epoch)
                
                # Save checkpoint
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    self._save_checkpoint(epoch, val_loss, is_best=False)
            
            # DAgger-lite (if enabled)
            if dagger and epoch % 20 == 0 and epoch > 0:
                self._dagger_update(demos, weights)
            
            logger.info(f"Epoch {epoch}/{self.num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Save final model
        self._save_checkpoint(self.num_epochs - 1, self.best_loss, is_best=True)
        
        # Generate training summary
        self._generate_summary()
        
        self.logger.close()
        logger.info("Behavior cloning training completed!")
    
    def _load_demos(self) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Load demonstrations and weights."""
        # Load weights
        weights_file = self.demos_dir / "weights.npz"
        if weights_file.exists():
            weights_data = np.load(weights_file)
            weights = weights_data['weights']
        else:
            logger.warning("Weights file not found, using uniform weights")
            weights = np.ones(1000)  # Default weight
        
        # Load demos
        demos_file = self.demos_dir / "filtered_demos.npz"
        if demos_file.exists():
            demos_data = np.load(demos_file, allow_pickle=True)
            demos = demos_data['demos'].tolist()
        else:
            # Load from individual demo files
            demo_files = list(self.demos_dir.glob("demos_*.npz"))
            if not demo_files:
                raise FileNotFoundError("No demo files found")
            
            demos = []
            for demo_file in demo_files:
                demo_data = np.load(demo_file, allow_pickle=True)
                demos.extend(demo_data['demos'].tolist())
        
        logger.info(f"Loaded {len(demos)} demonstrations")
        return demos, weights
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.epoch}"):
            # Move to device
            image = batch['image'].to(self.device)
            aux = batch['aux'].to(self.device)
            action = batch['action'].to(self.device)
            weight = batch['weight'].to(self.device)
            
            # Forward pass
            features = self.model['encoder'](image, aux)
            pred_action = self.model['action_head'](features)
            
            # Calculate loss
            loss = self.criterion(pred_action, action)
            weighted_loss = (loss * weight.unsqueeze(1)).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            
            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()
            
            total_loss += weighted_loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                image = batch['image'].to(self.device)
                aux = batch['aux'].to(self.device)
                action = batch['action'].to(self.device)
                weight = batch['weight'].to(self.device)
                
                # Forward pass
                features = self.model['encoder'](image, aux)
                pred_action = self.model['action_head'](features)
                
                # Calculate loss
                loss = self.criterion(pred_action, action)
                weighted_loss = (loss * weight.unsqueeze(1)).mean()
                
                total_loss += weighted_loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _dagger_update(self, demos: List[Dict[str, Any]], weights: np.ndarray) -> None:
        """Perform DAgger-lite update."""
        logger.info("Performing DAgger-lite update...")
        
        # Generate on-policy rollouts
        new_demos = self._generate_on_policy_demos()
        
        # Add to existing demos
        demos.extend(new_demos)
        
        # Recalculate weights
        from .filter_weight import DemoProcessor
        processor = DemoProcessor(self.config.get('filtering', {}))
        _, new_weights = processor.process_demos(demos, str(self.demos_dir))
        
        logger.info(f"Added {len(new_demos)} on-policy demos")
    
    def _generate_on_policy_demos(self) -> List[Dict[str, Any]]:
        """Generate on-policy demonstrations."""
        # This would integrate with the environment
        # For now, return empty list
        return []
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_file = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_file)
        
        # Save best checkpoint
        if is_best:
            best_file = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_file)
            logger.info(f"Saved best model at epoch {epoch} with loss {loss:.4f}")
    
    def _generate_summary(self) -> None:
        """Generate training summary."""
        summary = {
            'final_loss': self.training_history[-1] if self.training_history else 0.0,
            'best_loss': self.best_loss,
            'num_epochs': len(self.training_history),
            'training_history': self.training_history,
            'config': self.config
        }
        
        summary_file = self.output_dir / "training_summary.json"
        save_json(summary, str(summary_file))
        
        logger.info(f"Training summary saved to {summary_file}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch} with loss {self.best_loss:.4f}")
    
    def predict(self, image: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Predict action for given observation."""
        self.model.eval()
        
        with torch.no_grad():
            features = self.model['encoder'](image, aux)
            action = self.model['action_head'](features)
        
        return action
