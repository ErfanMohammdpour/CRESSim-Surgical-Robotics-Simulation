"""
Training module for surgical robotics pipeline
Handles IL and RL model training
"""

import torch
import torch.nn as nn
import torch.utils.data
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import json
import numpy as np
from PIL import Image

# Import project modules
from src.vision.encoders import CNNEncoder
from src.il.bc_trainer import PolicyNetwork
from src.envs.mock_env import MockSuctionEnv

logger = logging.getLogger(__name__)

class SurgicalDataset(torch.utils.data.Dataset):
    """Surgical dataset for training"""
    
    def __init__(self, data_dir: Path, image_size: tuple = (128, 128)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.samples = []
        
        # Load metadata
        metadata_file = data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                episodes = metadata.get('episodes', [])
            
            # Flatten episodes into samples
            for episode in episodes:
                images = episode.get('images', [])
                actions = episode.get('actions', [])
                
                for img_name, action in zip(images, actions):
                    img_path = data_dir / 'images' / img_name
                    if img_path.exists():
                        self.samples.append({
                            'image_path': img_path,
                            'action': np.array(action, dtype=np.float32)
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image, dtype=np.uint8)
        
        # Convert to tensor (HWC -> CHW)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        action_tensor = torch.tensor(sample['action'], dtype=torch.float32)
        
        return image_tensor, action_tensor

class ModelTrainer:
    """Handles training of IL and RL models"""
    
    def __init__(self, checkpoints_dir: str = "data/checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelTrainer initialized")
    
    def train_il_model(self, train_dir: Path, epochs: int = 50) -> Path:
        """Train Imitation Learning model"""
        logger.info("📚 Training Imitation Learning model...")
        
        # Create dataset
        dataset = SurgicalDataset(train_dir, image_size=(128, 128))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Create model
        encoder = CNNEncoder(
            input_channels=3,
            output_dim=512,
            channels=[32, 64, 128, 256],
            kernel_sizes=[3, 3, 3, 3],
            strides=[2, 2, 2, 2],
            padding=[1, 1, 1, 1]
        )
        
        model = PolicyNetwork(encoder=encoder, hidden_dim=512, action_dim=5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (images, actions) in enumerate(dataloader):
                images = images.to(device)
                actions = actions.to(device)
                
                optimizer.zero_grad()
                pred_actions = model(images)
                loss = criterion(pred_actions, actions)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Save model
        model_path = self.checkpoints_dir / "il_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epochs,
            'loss': total_loss/len(dataloader)
        }, model_path)
        
        logger.info(f"✅ IL model saved: {model_path}")
        return model_path
    
    def train_rl_model(self, timesteps: int = 50000) -> Path:
        """Train Reinforcement Learning model"""
        logger.info("🤖 Training Reinforcement Learning model...")
        
        # Create environment
        env = MockSuctionEnv(image_size=(128, 128), max_steps=1000)
        
        # Create PPO model
        from stable_baselines3 import PPO
        
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=8,
            gamma=0.995,
            verbose=1,
            tensorboard_log=str(self.checkpoints_dir / "tensorboard")
        )
        
        # Train model
        model.learn(total_timesteps=timesteps)
        
        # Save model
        model_path = self.checkpoints_dir / "rl_model"
        model.save(str(model_path))
        
        logger.info(f"✅ RL model saved: {model_path}")
        return model_path
    
    def get_checkpoints_dir(self) -> Path:
        """Get checkpoints directory path"""
        return self.checkpoints_dir
