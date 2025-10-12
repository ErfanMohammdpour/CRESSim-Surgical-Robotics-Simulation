"""
Main RL training loop with safety shield integration.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import json

from .algo import PPO, SAC, PPOPolicy, SACPolicy
from .buffers import PPOBuffer, ReplayBuffer
from .safety_shield import RLSafetyShield, SafetyAwareReward, SafetyCurriculum
from ..vision.encoders import create_encoder
from ..vision.safety_seg import create_safety_network, create_safety_shield
from ..envs.unity_env import UnityEnv, MockUnityEnv
from ..envs.wrappers import FrameStackWrapper, RewardShapingWrapper, DomainRandomizationWrapper
from ..utils.log import Logger
from ..utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Main RL trainer with safety shield integration.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        output_dir: str = "data/checkpoints"
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        # Training parameters
        self.algorithm = config.get('algorithm', 'ppo')
        self.total_timesteps = config.get('total_timesteps', 1000000)
        self.eval_freq = config.get('eval_freq', 10000)
        self.save_freq = config.get('save_freq', 50000)
        self.log_interval = config.get('log_interval', 10)
        
        # Environment parameters
        self.env_config = config.get('env_config', {})
        self.max_episode_steps = self.env_config.get('max_episode_steps', 1000)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._initialize_models()
        self._initialize_environment()
        self._initialize_safety_components()
        self._initialize_training_components()
        
        # Logger
        self.logger = Logger(str(self.output_dir), "rl_training")
        
        # Training state
        self.timestep = 0
        self.episode = 0
        self.best_performance = float('-inf')
        self.training_history = []
    
    def _initialize_models(self) -> None:
        """Initialize RL models."""
        # Create encoder
        self.encoder = create_encoder(self.config.get('model', {}))
        
        # Create policy based on algorithm
        if self.algorithm == 'ppo':
            self.policy = PPOPolicy(
                encoder=self.encoder,
                action_dim=5,
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 256)
            )
            self.algorithm_instance = PPO(
                policy=self.policy,
                **self.config.get('ppo', {})
            )
        elif self.algorithm == 'sac':
            self.policy = SACPolicy(
                encoder=self.encoder,
                action_dim=5,
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 256)
            )
            self.algorithm_instance = SAC(
                policy=self.policy,
                **self.config.get('sac', {})
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Load checkpoint if provided
        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)
    
    def _initialize_environment(self) -> None:
        """Initialize training environment."""
        # Load paths config
        from ..utils.io import load_config
        paths_config = load_config("configs/paths.yaml")
        unity_build_path = paths_config.get('unity_build')
        
        # Create environment
        if Path(unity_build_path).exists():
            self.env = UnityEnv(unity_build_path)
        else:
            logger.warning("Unity build not found, using mock environment")
            self.env = MockUnityEnv()
        
        # Wrap environment
        self.env = self._wrap_environment(self.env)
        
        # Get observation and action spaces
        obs_space = self.env.observation_space
        self.obs_shape = {
            'image': obs_space['image'].shape,
            'aux': obs_space['aux'].shape
        }
        self.action_dim = self.env.action_space.shape[0]
    
    def _wrap_environment(self, env):
        """Wrap environment with necessary wrappers."""
        # Frame stacking
        env = FrameStackWrapper(env, num_frames=4)
        
        # Reward shaping
        env = RewardShapingWrapper(env, self.env_config)
        
        # Domain randomization
        env = DomainRandomizationWrapper(env, self.env_config)
        
        return env
    
    def _initialize_safety_components(self) -> None:
        """Initialize safety components."""
        # Create safety network
        safety_config = self.config.get('safety_config', {})
        self.safety_net = create_safety_network(safety_config)
        
        # Create safety shield
        self.safety_shield = RLSafetyShield(
            safety_net=self.safety_net,
            config=safety_config,
            device=self.device
        )
        
        # Create safety-aware reward
        self.safety_reward = SafetyAwareReward(self.env_config.get('reward_weights', {}))
        
        # Create safety curriculum
        self.safety_curriculum = SafetyCurriculum(safety_config.get('curriculum', {}))
    
    def _initialize_training_components(self) -> None:
        """Initialize training components."""
        # Create buffer
        if self.algorithm == 'ppo':
            self.buffer = PPOBuffer(
                buffer_size=self.config.get('ppo', {}).get('n_steps', 2048),
                obs_shape=self.obs_shape,
                action_dim=self.action_dim,
                device=self.device
            )
        else:  # SAC
            self.buffer = ReplayBuffer(
                capacity=self.config.get('sac', {}).get('buffer_size', 1000000),
                obs_shape=self.obs_shape,
                action_dim=self.action_dim,
                device=self.device
            )
    
    def train(self) -> None:
        """Main training loop."""
        logger.info(f"Starting RL training with {self.algorithm.upper()}")
        logger.info(f"Total timesteps: {self.total_timesteps}")
        
        # Training loop
        while self.timestep < self.total_timesteps:
            # Collect rollouts
            rollout_info = self._collect_rollout()
            
            # Update policy
            if self.algorithm == 'ppo':
                update_info = self._update_ppo()
            else:  # SAC
                update_info = self._update_sac()
            
            # Logging
            if self.timestep % self.log_interval == 0:
                self._log_training_info(rollout_info, update_info)
            
            # Evaluation
            if self.timestep % self.eval_freq == 0:
                eval_info = self._evaluate()
                self._log_evaluation_info(eval_info)
                
                # Save checkpoint if performance improved
                if eval_info['mean_reward'] > self.best_performance:
                    self.best_performance = eval_info['mean_reward']
                    self._save_checkpoint(is_best=True)
            
            # Save checkpoint
            if self.timestep % self.save_freq == 0:
                self._save_checkpoint(is_best=False)
            
            # Update safety curriculum
            self.safety_curriculum.update(rollout_info.get('episode_success', False))
            
            # Update safety thresholds
            safety_thresholds = self.safety_curriculum.get_safety_thresholds()
            self.safety_shield.d_safe = safety_thresholds['d_safe']
            self.safety_shield.d_warning = safety_thresholds['d_warning']
        
        # Final evaluation and save
        final_eval = self._evaluate()
        self._log_evaluation_info(final_eval)
        self._save_checkpoint(is_best=True)
        
        self.logger.close()
        logger.info("RL training completed!")
    
    def _collect_rollout(self) -> Dict[str, Any]:
        """Collect rollout data."""
        if self.algorithm == 'ppo':
            return self._collect_ppo_rollout()
        else:
            return self._collect_sac_rollout()
    
    def _collect_ppo_rollout(self) -> Dict[str, Any]:
        """Collect PPO rollout."""
        self.policy.eval()
        
        # Reset environment
        obs, info = self.env.reset()
        self.safety_shield.reset()
        
        episode_reward = 0
        episode_length = 0
        episode_success = False
        
        # Collect rollout
        for step in range(self.config.get('ppo', {}).get('n_steps', 2048)):
            # Get action from policy
            with torch.no_grad():
                image_tensor = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
                aux_tensor = torch.FloatTensor(obs['aux']).unsqueeze(0).to(self.device)
                
                action, log_prob, value = self.policy.get_action(image_tensor, aux_tensor)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
                value = value.cpu().numpy()[0]
            
            # Apply safety shield
            projected_action, safety_info = self.safety_shield.apply_safety_shield(
                image_tensor, torch.FloatTensor(action).unsqueeze(0)
            )
            projected_action = projected_action.cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(projected_action)
            
            # Calculate safety-aware reward
            safety_aware_reward = self.safety_reward.calculate_reward(
                obs, projected_action, next_obs, info, safety_info
            )
            
            # Add to buffer
            self.buffer.add(
                obs=obs,
                action=projected_action,
                reward=safety_aware_reward,
                value=value,
                log_prob=log_prob,
                done=terminated or truncated
            )
            
            # Update episode info
            episode_reward += safety_aware_reward
            episode_length += 1
            
            # Check if episode ended
            if terminated or truncated:
                # Finish path for PPO
                self.buffer.finish_path()
                
                # Check episode success
                episode_success = self._is_episode_successful(info)
                
                # Reset environment
                obs, info = self.env.reset()
                self.safety_shield.reset()
                
                # Log episode
                self.logger.log_scalar('episode/reward', episode_reward, self.timestep)
                self.logger.log_scalar('episode/length', episode_length, self.timestep)
                self.logger.log_scalar('episode/success', float(episode_success), self.timestep)
                
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            self.timestep += 1
        
        return {
            'episode_success': episode_success,
            'episode_reward': episode_reward,
            'episode_length': episode_length
        }
    
    def _collect_sac_rollout(self) -> Dict[str, Any]:
        """Collect SAC rollout."""
        # Reset environment
        obs, info = self.env.reset()
        self.safety_shield.reset()
        
        episode_reward = 0
        episode_length = 0
        episode_success = False
        
        # Collect steps
        for step in range(1000):  # SAC collects one step at a time
            # Get action from policy
            with torch.no_grad():
                image_tensor = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
                aux_tensor = torch.FloatTensor(obs['aux']).unsqueeze(0).to(self.device)
                
                action, _ = self.policy.get_action(image_tensor, aux_tensor)
                action = action.cpu().numpy()[0]
            
            # Apply safety shield
            projected_action, safety_info = self.safety_shield.apply_safety_shield(
                image_tensor, torch.FloatTensor(action).unsqueeze(0)
            )
            projected_action = projected_action.cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(projected_action)
            
            # Calculate safety-aware reward
            safety_aware_reward = self.safety_reward.calculate_reward(
                obs, projected_action, next_obs, info, safety_info
            )
            
            # Add to buffer
            self.buffer.add(
                obs=obs,
                action=projected_action,
                reward=safety_aware_reward,
                next_obs=next_obs,
                done=terminated or truncated
            )
            
            # Update episode info
            episode_reward += safety_aware_reward
            episode_length += 1
            
            # Check if episode ended
            if terminated or truncated:
                episode_success = self._is_episode_successful(info)
                
                # Log episode
                self.logger.log_scalar('episode/reward', episode_reward, self.timestep)
                self.logger.log_scalar('episode/length', episode_length, self.timestep)
                self.logger.log_scalar('episode/success', float(episode_success), self.timestep)
                
                # Reset environment
                obs, info = self.env.reset()
                self.safety_shield.reset()
                
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            self.timestep += 1
        
        return {
            'episode_success': episode_success,
            'episode_reward': episode_reward,
            'episode_length': episode_length
        }
    
    def _update_ppo(self) -> Dict[str, float]:
        """Update PPO policy."""
        # Get rollout data
        rollout_data = self.buffer.get()
        
        # Update policy
        update_info = self.algorithm_instance.update(
            obs=rollout_data,
            actions=rollout_data['actions'],
            rewards=rollout_data['rewards'],
            dones=rollout_data['dones'],
            old_log_probs=rollout_data['log_probs']
        )
        
        # Clear buffer
        self.buffer.clear()
        
        return update_info
    
    def _update_sac(self) -> Dict[str, float]:
        """Update SAC policy."""
        # Sample batch from buffer
        if len(self.buffer) < 1000:  # Need enough data
            return {}
        
        batch = self.buffer.sample(256)  # Batch size
        
        # Update policy
        update_info = self.algorithm_instance.update(
            obs=batch,
            actions=batch['actions'],
            rewards=batch['rewards'],
            next_obs={key.replace('next_', ''): value for key, value in batch.items() if key.startswith('next_')},
            dones=batch['dones']
        )
        
        return update_info
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        self.policy.eval()
        
        eval_episodes = 10
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
        for _ in range(eval_episodes):
            obs, info = self.env.reset()
            self.safety_shield.reset()
            
            episode_reward = 0
            episode_length = 0
            
            for _ in range(self.max_episode_steps):
                with torch.no_grad():
                    image_tensor = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
                    aux_tensor = torch.FloatTensor(obs['aux']).unsqueeze(0).to(self.device)
                    
                    action, _, _ = self.policy.get_action(image_tensor, aux_tensor, deterministic=True)
                    action = action.cpu().numpy()[0]
                
                # Apply safety shield
                projected_action, safety_info = self.safety_shield.apply_safety_shield(
                    image_tensor, torch.FloatTensor(action).unsqueeze(0)
                )
                projected_action = projected_action.cpu().numpy()[0]
                
                obs, reward, terminated, truncated, info = self.env.step(projected_action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(self._is_episode_successful(info))
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths),
            'success_rate': np.mean(eval_successes)
        }
    
    def _is_episode_successful(self, info: Dict[str, Any]) -> bool:
        """Check if episode was successful."""
        liquid_mass = info.get('liquid_mass_remaining', 1.0)
        contaminant_mass = info.get('contaminant_mass_remaining', 1.0)
        
        return liquid_mass < 0.1 and contaminant_mass < 0.05
    
    def _log_training_info(self, rollout_info: Dict[str, Any], update_info: Dict[str, float]) -> None:
        """Log training information."""
        for key, value in update_info.items():
            self.logger.log_scalar(f'train/{key}', value, self.timestep)
        
        # Log safety metrics
        safety_metrics = self.safety_shield.get_safety_metrics()
        for key, value in safety_metrics.items():
            self.logger.log_scalar(f'safety/{key}', value, self.timestep)
        
        # Log curriculum info
        curriculum_info = self.safety_curriculum.get_curriculum_info()
        for key, value in curriculum_info.items():
            self.logger.log_scalar(f'curriculum/{key}', value, self.timestep)
    
    def _log_evaluation_info(self, eval_info: Dict[str, float]) -> None:
        """Log evaluation information."""
        for key, value in eval_info.items():
            self.logger.log_scalar(f'eval/{key}', value, self.timestep)
        
        logger.info(f"Evaluation - Reward: {eval_info['mean_reward']:.2f} ± {eval_info['std_reward']:.2f}, "
                   f"Success Rate: {eval_info['success_rate']:.2f}")
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'timestep': self.timestep,
            'episode': self.episode,
            'policy_state_dict': self.policy.state_dict(),
            'safety_net_state_dict': self.safety_net.state_dict(),
            'config': self.config,
            'best_performance': self.best_performance
        }
        
        # Save regular checkpoint
        checkpoint_file = self.output_dir / f"checkpoint_{self.timestep}.pth"
        torch.save(checkpoint, checkpoint_file)
        
        # Save best checkpoint
        if is_best:
            best_file = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_file)
            logger.info(f"Saved best model at timestep {self.timestep}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.safety_net.load_state_dict(checkpoint['safety_net_state_dict'])
        self.timestep = checkpoint.get('timestep', 0)
        self.episode = checkpoint.get('episode', 0)
        self.best_performance = checkpoint.get('best_performance', float('-inf'))
        
        logger.info(f"Loaded checkpoint from timestep {self.timestep}")
