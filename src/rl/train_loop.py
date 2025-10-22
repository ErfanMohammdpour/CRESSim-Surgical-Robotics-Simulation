"""
PPO algorithm implementation with safety shield integration and GPU support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from ..envs.mock_env import MockSuctionEnv
from ..envs.unity_env import create_environment
from ..rl.safety_shield import RLSafetyShield, SafetyAwareReward
from ..vision.safety_seg import SafetySegmentation
from ..utils.seeding import set_seed
from ..utils.device import (
    get_device_from_config, log_device_info, optimize_for_gpu, 
    create_optimizer, create_grad_scaler, get_dataloader_kwargs,
    handle_oom_error, warmup_model, log_memory_usage, amp_enabled
)

logger = logging.getLogger(__name__)


class SafetyCallback(BaseCallback):
    """Callback for safety monitoring during training."""
    
    def __init__(self, safety_shield: RLSafetyShield, verbose: int = 0):
        super().__init__(verbose)
        self.safety_shield = safety_shield
        self.safety_stats = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Get safety stats
        stats = self.safety_shield.get_safety_stats()
        self.safety_stats.append(stats)
        
        # Log safety violations
        if stats['violation_count'] > 0:
            logger.warning(f"Safety violation detected: {stats}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if self.safety_stats:
            avg_stats = {
                'avg_violations': np.mean([s['violation_count'] for s in self.safety_stats]),
                'avg_distance': np.mean([s['avg_distance'] for s in self.safety_stats]),
                'total_violations': sum([s['violation_count'] for s in self.safety_stats])
            }
            
            logger.info(f"Rollout safety stats: {avg_stats}")
            self.safety_stats = []


class PPOTrainer:
    """PPO trainer with GPU support and safety integration."""
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any], 
                 safety_config: Dict[str, Any], output_dir: str):
        self.config = config
        self.env_config = env_config
        self.safety_config = safety_config
        self.output_dir = output_dir
        
        # Setup device and logging
        self.device = get_device_from_config(config)
        log_device_info(self.device, config)
        
        # Setup logging
        self.log_dir = Path(output_dir) / "logs" / "rl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment
        self.env = self._create_environment()
        
        # Create safety shield
        self.safety_shield = self._create_safety_shield()
        
        # Create PPO model
        self.model = self._create_ppo_model()
        
        # Training state
        self.training_stats = []
        
        logger.info(f"PPOTrainer initialized on {self.device}")
        logger.info(f"AMP enabled: {amp_enabled(config)}")
    
    def _create_environment(self):
        """Create training environment."""
        # Set seed for reproducibility
        set_seed(self.env_config.get("seed", 42))
        
        # Create environment
        if self.env_config.get("use_mock", False):
            env = MockSuctionEnv(self.env_config)
        else:
            env = create_environment(self.env_config)
        
        # Wrap with safety-aware reward
        if self.safety_config.get("enabled", True):
            env = SafetyAwareReward(env, self.safety_shield, self.safety_config)
        
        return env
    
    def _create_safety_shield(self):
        """Create safety shield."""
        if not self.safety_config.get("enabled", True):
            return None
        
        # Load pretrained safety segmentation model
        safety_model_path = Path(self.output_dir) / "models" / "safety_best.pth"
        
        if safety_model_path.exists():
            safety_model = SafetySegmentation(
                input_size=self.safety_config.get("segmentation", {}).get("input_size", [64, 64]),
                num_classes=self.safety_config.get("segmentation", {}).get("num_classes", 3)
            )
            
            # Load checkpoint
            checkpoint = torch.load(safety_model_path, map_location=self.device)
            safety_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move to device and optimize
            safety_model = optimize_for_gpu(safety_model, self.config)
            
            logger.info("Loaded pretrained safety segmentation model")
        else:
            logger.warning("No pretrained safety model found, using random initialization")
            safety_model = None
        
        # Create safety shield
        safety_shield = RLSafetyShield(
            safety_model=safety_model,
            config=self.safety_config,
            device=self.device
        )
        
        return safety_shield
    
    def _create_ppo_model(self):
        """Create PPO model with GPU support."""
        # PPO configuration
        ppo_config = self.config.get("ppo", {})
        
        # Create PPO model
        model = PPO(
            "CnnPolicy",
            self.env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.01),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            verbose=1,
            device=self.device,
            policy_kwargs={
                "device": self.device,
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])]
            }
        )
        
        # Move model to device
        if hasattr(model.policy, 'to'):
            model.policy = model.policy.to(self.device)
        
        logger.info(f"PPO model created on {self.device}")
        
        return model
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        try:
            self.model = PPO.load(checkpoint_path, device=self.device)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def train(self, total_timesteps: int = 100000):
        """Train the PPO model."""
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        # Setup callbacks
        callbacks = []
        if self.safety_shield is not None:
            callbacks.append(SafetyCallback(self.safety_shield))
        
        # Training loop with memory monitoring
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Log final memory usage
            log_memory_usage(self.device, "Training Complete")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM during training, adjusting configuration...")
                self.config = handle_oom_error(e, self.config)
                
                # Reduce batch size and retry
                ppo_config = self.config.get("ppo", {})
                ppo_config["batch_size"] = max(1, ppo_config.get("batch_size", 64) // 2)
                ppo_config["n_steps"] = max(256, ppo_config.get("n_steps", 2048) // 2)
                
                logger.info(f"Retrying with batch_size={ppo_config['batch_size']}, n_steps={ppo_config['n_steps']}")
                
                # Recreate model with smaller batch size
                self.model = self._create_ppo_model()
                
                # Retry training
                self.model.learn(
                    total_timesteps=total_timesteps,
                    callback=callbacks,
                    progress_bar=True
                )
            else:
                raise e
        
        # Save final model
        self.save_model()
        
        logger.info("PPO training completed")
    
    def save_model(self):
        """Save the trained model."""
        model_path = Path(self.output_dir) / "rl_best_model"
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained model."""
        logger.info(f"Evaluating model for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        safety_violations = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_violations = 0
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Apply safety shield if enabled
                if self.safety_shield is not None:
                    safe_action = self.safety_shield.check_safety(obs, action)
                    if not np.array_equal(action, safe_action):
                        episode_violations += 1
                    action = safe_action
                
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            safety_violations.append(episode_violations)
            
            logger.info(
                f"Episode {episode}: Reward={episode_reward:.2f}, "
                f"Length={episode_length}, Violations={episode_violations}"
            )
        
        # Calculate statistics
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'safety_violations': safety_violations,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'avg_violations': np.mean(safety_violations),
            'success_rate': np.mean([r > 0 for r in episode_rewards])
        }
        
        logger.info(f"Evaluation results: {results}")
        return results


def train_rl_model(config: Dict[str, Any], env_config: Dict[str, Any], 
                   safety_config: Dict[str, Any], output_dir: str,
                   checkpoint_path: Optional[str] = None) -> str:
    """Train RL model with PPO."""
    trainer = PPOTrainer(config, env_config, safety_config, output_dir)
    
    # Load checkpoint if provided
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)
    
    # Train model
    total_timesteps = config.get("total_timesteps", 100000)
    trainer.train(total_timesteps)
    
    # Evaluate model
    results = trainer.evaluate()
    
    # Save results
    results_path = Path(output_dir) / "rl_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Return path to saved model
    return str(Path(output_dir) / "rl_best_model")