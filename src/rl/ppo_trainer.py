"""
PPO trainer for reinforcement learning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ..utils.gpu import get_device, optimize_for_gpu
from ..utils.log import setup_logging, log_metrics
from ..envs.mock_env import MockSuctionEnv
from ..vision.encoders import CNNEncoder
from ..safety.safety_shield import SafetyShield

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO trainer with safety shield."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        checkpoint_path: Optional[str] = None,
        output_dir: str = "data/checkpoints",
        mock: bool = False
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.mock = mock
        
        # Setup device
        self.device = get_device()
        optimize_for_gpu()
        
        # Setup logging
        self.log_dir = Path(output_dir) / "logs" / "rl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.log_dir)
        
        # Create environment
        self.env = self._create_environment()
        
        # Create model
        self.model = self._create_model()
        
        # Setup safety shield
        self.safety_shield = SafetyShield(config.get("safety", {}))
        
        logger.info(f"PPOTrainer initialized on {self.device}")
        logger.info(f"Mock environment: {mock}")
    
    def _create_environment(self):
        """Create training environment."""
        if self.mock:
            # Use mock environment
            env = MockSuctionEnv(
                image_size=tuple(self.config.get("image_size", [128, 128])),
                max_steps=self.config.get("max_episode_steps", 1000)
            )
            # Return DummyVecEnv directly (VecTransposeImage is applied automatically)
            return DummyVecEnv([lambda: env])
        else:
            # Use Unity environment (when available)
            from ..envs.unity_env import UnitySuctionEnv
            env = UnitySuctionEnv(
                unity_build_path=self.config.get("unity_build_path"),
                image_size=tuple(self.config.get("image_size", [128, 128])),
                max_steps=self.config.get("max_episode_steps", 1000)
            )
            return DummyVecEnv([lambda: env])
    
    def _create_model(self):
        """Create PPO model."""
        # PPO hyperparameters
        ppo_config = self.config.get("ppo", {})
        
        model = PPO(
            "CnnPolicy",  # Use CNN policy for image inputs
            self.env,
            learning_rate=float(ppo_config.get("learning_rate", 1e-4)),
            n_steps=int(ppo_config.get("n_steps", 4096)),
            batch_size=int(ppo_config.get("batch_size", 128)),
            n_epochs=int(ppo_config.get("n_epochs", 8)),
            gamma=float(ppo_config.get("gamma", 0.995)),
            gae_lambda=float(ppo_config.get("gae_lambda", 0.95)),
            clip_range=float(ppo_config.get("clip_range", 0.15)),
            ent_coef=float(ppo_config.get("ent_coef", 0.005)),
            vf_coef=float(ppo_config.get("vf_coef", 0.3)),
            max_grad_norm=float(ppo_config.get("max_grad_norm", 0.5)),
            verbose=1,
            device=self.device,
            tensorboard_log=str(self.log_dir / "tensorboard")
        )
        
        # Load checkpoint if provided
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model = PPO.load(self.checkpoint_path, env=self.env, device=self.device)
        
        return model
    
    def train(self, total_timesteps: int = 1000000):
        """Train the PPO model."""
        logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
        
        # Setup callbacks (simplified for now)
        callbacks = []
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="ppo_suction_rl"
        )
        
        # Save final model
        final_model_path = Path(self.output_dir) / "final_ppo_model"
        self.model.save(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        logger.info("PPO training completed!")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained model."""
        logger.info(f"Evaluating model on {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Apply safety shield
                safe_action = self.safety_shield.apply_shield(obs, action)
                
                obs, reward, done, info = self.env.step(safe_action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check success (based on reward threshold)
            success = episode_reward > 0.5
            success_rates.append(success)
            
            logger.info(f"Episode {episode+1}: Reward={episode_reward:.3f}, Length={episode_length}")
        
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean(success_rates),
            'num_episodes': num_episodes
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_path = Path(self.output_dir) / f"ppo_checkpoint_step_{step}"
        self.model.save(str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    # Test PPO trainer
    config = {
        "image_size": [128, 128],
        "max_episode_steps": 1000,
        "ppo": {
            "learning_rate": 1e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 8,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.15,
            "ent_coef": 0.005,
            "vf_coef": 0.3,
            "max_grad_norm": 0.5
        },
        "safety": {
            "d_safe": 0.05,
            "segmentation_threshold": 0.5,
            "projection_scaling_factor": 0.8
        }
    }
    
    trainer = PPOTrainer(
        config=config,
        output_dir="data/checkpoints",
        mock=True
    )
    
    # Train for a short time for testing
    trainer.train(total_timesteps=10000)
    
    # Evaluate
    results = trainer.evaluate(num_episodes=5)
    print(f"Evaluation results: {results}")

