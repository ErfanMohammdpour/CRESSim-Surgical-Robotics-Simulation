"""
Rollout collection for reinforcement learning.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import time


class RolloutCollector:
    """Collects rollouts from environment using current policy."""
    
    def __init__(self, env, policy, config: Dict[str, Any]):
        self.env = env
        self.policy = policy
        self.config = config
        
        # Rollout parameters
        self.n_steps = config.get('n_steps', 2048)
        self.n_envs = config.get('n_envs', 1)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        
        # Storage for rollouts
        self.reset_storage()
    
    def reset_storage(self) -> None:
        """Reset rollout storage."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.infos = []
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_stats = defaultdict(list)
    
    def collect_rollouts(self) -> Dict[str, Any]:
        """Collect rollouts from environment."""
        self.reset_storage()
        
        # Initialize environments
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Convert to batch format if single environment
        if obs.ndim == 3:  # Single env
            obs = obs[np.newaxis, ...]
        
        episode_rewards = np.zeros(self.n_envs)
        episode_lengths = np.zeros(self.n_envs)
        
        for step in range(self.n_steps):
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action_and_value(obs)
            
            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Handle single environment case
            if not isinstance(done, np.ndarray):
                done = np.array([done])
                truncated = np.array([truncated])
                reward = np.array([reward])
            
            # Store rollout data
            self.observations.append(obs.copy())
            self.actions.append(action.copy())
            self.rewards.append(reward.copy())
            self.values.append(value.copy())
            self.log_probs.append(log_prob.copy())
            self.dones.append(done.copy())
            self.infos.append(info.copy())
            
            # Update episode statistics
            episode_rewards += reward
            episode_lengths += 1
            
            # Handle episode completion
            for i in range(self.n_envs):
                if done[i] or truncated[i]:
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    
                    # Store episode statistics
                    if isinstance(info, dict):
                        for key, value in info.items():
                            if isinstance(value, (int, float)):
                                self.episode_stats[key].append(value)
                    
                    # Reset episode counters
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
            
            # Update observation
            obs = next_obs
            if obs.ndim == 3:  # Single env
                obs = obs[np.newaxis, ...]
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns()
        
        # Convert to numpy arrays
        rollout_data = {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'advantages': advantages,
            'returns': returns,
            'dones': np.array(self.dones),
            'infos': self.infos
        }
        
        # Episode statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_stats': dict(self.episode_stats),
            'total_steps': len(self.observations),
            'num_episodes': len(self.episode_rewards)
        }
        
        return rollout_data, stats
    
    def _compute_advantages_and_returns(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using GAE."""
        # Get final value for bootstrapping
        with torch.no_grad():
            final_obs = self.observations[-1]
            if final_obs.ndim == 3:  # Single env
                final_obs = final_obs[np.newaxis, ...]
            final_value = self.policy.get_value(final_obs)
        
        # Convert to numpy
        values = np.array(self.values)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        # Add final value
        values = np.append(values, final_value, axis=0)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantages[t + 1]
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns


class SafetyRolloutCollector(RolloutCollector):
    """Rollout collector with safety monitoring."""
    
    def __init__(self, env, policy, safety_shield, config: Dict[str, Any]):
        super().__init__(env, policy, config)
        self.safety_shield = safety_shield
        
        # Safety-specific storage
        self.safety_violations = []
        self.safety_masks = []
        self.projected_actions = []
        self.original_actions = []
    
    def collect_rollouts(self) -> Dict[str, Any]:
        """Collect rollouts with safety monitoring."""
        self.reset_storage()
        self.safety_violations = []
        self.safety_masks = []
        self.projected_actions = []
        self.original_actions = []
        
        # Initialize environments
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        if obs.ndim == 3:  # Single env
            obs = obs[np.newaxis, ...]
        
        episode_rewards = np.zeros(self.n_envs)
        episode_lengths = np.zeros(self.n_envs)
        safety_violation_counts = np.zeros(self.n_envs)
        
        for step in range(self.n_steps):
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action_and_value(obs)
            
            # Apply safety shield
            safe_action, safety_mask, violation = self.safety_shield.check_safety(obs, action)
            
            # Store safety information
            self.safety_violations.append(violation.copy())
            self.safety_masks.append(safety_mask.copy())
            self.projected_actions.append(safe_action.copy())
            self.original_actions.append(action.copy())
            
            # Step environment with safe action
            next_obs, reward, done, truncated, info = self.env.step(safe_action)
            
            # Handle single environment case
            if not isinstance(done, np.ndarray):
                done = np.array([done])
                truncated = np.array([truncated])
                reward = np.array([reward])
            
            # Store rollout data
            self.observations.append(obs.copy())
            self.actions.append(safe_action.copy())  # Store safe action
            self.rewards.append(reward.copy())
            self.values.append(value.copy())
            self.log_probs.append(log_prob.copy())
            self.dones.append(done.copy())
            self.infos.append(info.copy())
            
            # Update episode statistics
            episode_rewards += reward
            episode_lengths += 1
            safety_violation_counts += violation.astype(float)
            
            # Handle episode completion
            for i in range(self.n_envs):
                if done[i] or truncated[i]:
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    
                    # Store safety statistics
                    self.episode_stats['safety_violations'].append(safety_violation_counts[i])
                    self.episode_stats['safety_violation_rate'].append(
                        safety_violation_counts[i] / episode_lengths[i]
                    )
                    
                    # Store other episode statistics
                    if isinstance(info, dict):
                        for key, value in info.items():
                            if isinstance(value, (int, float)):
                                self.episode_stats[key].append(value)
                    
                    # Reset episode counters
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    safety_violation_counts[i] = 0
            
            # Update observation
            obs = next_obs
            if obs.ndim == 3:  # Single env
                obs = obs[np.newaxis, ...]
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns()
        
        # Convert to numpy arrays
        rollout_data = {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'advantages': advantages,
            'returns': returns,
            'dones': np.array(self.dones),
            'infos': self.infos,
            'safety_violations': np.array(self.safety_violations),
            'safety_masks': np.array(self.safety_masks),
            'projected_actions': np.array(self.projected_actions),
            'original_actions': np.array(self.original_actions)
        }
        
        # Episode statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_stats': dict(self.episode_stats),
            'total_steps': len(self.observations),
            'num_episodes': len(self.episode_rewards),
            'total_safety_violations': np.sum(self.safety_violations),
            'safety_violation_rate': np.mean(self.safety_violations)
        }
        
        return rollout_data, stats


class AsyncRolloutCollector:
    """Asynchronous rollout collector for multiple environments."""
    
    def __init__(self, env_fns, policy, config: Dict[str, Any]):
        self.env_fns = env_fns
        self.policy = policy
        self.config = config
        
        # Create environments
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        
        # Rollout parameters
        self.n_steps = config.get('n_steps', 2048)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
    
    def collect_rollouts(self) -> Dict[str, Any]:
        """Collect rollouts from multiple environments."""
        # Initialize all environments
        observations = []
        for env in self.envs:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            observations.append(obs)
        
        observations = np.array(observations)
        
        # Storage
        all_observations = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_log_probs = []
        all_dones = []
        all_infos = []
        
        episode_rewards = np.zeros(self.n_envs)
        episode_lengths = np.zeros(self.n_envs)
        
        for step in range(self.n_steps):
            # Get actions from policy
            with torch.no_grad():
                actions, log_probs, values = self.policy.get_action_and_value(observations)
            
            # Step all environments
            next_observations = []
            rewards = []
            dones = []
            infos = []
            
            for i, env in enumerate(self.envs):
                obs, reward, done, truncated, info = env.step(actions[i])
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                next_observations.append(obs)
                rewards.append(reward)
                dones.append(done or truncated)
                infos.append(info)
            
            next_observations = np.array(next_observations)
            rewards = np.array(rewards)
            dones = np.array(dones)
            
            # Store data
            all_observations.append(observations.copy())
            all_actions.append(actions.copy())
            all_rewards.append(rewards.copy())
            all_values.append(values.copy())
            all_log_probs.append(log_probs.copy())
            all_dones.append(dones.copy())
            all_infos.append(infos.copy())
            
            # Update episode statistics
            episode_rewards += rewards
            episode_lengths += 1
            
            # Reset completed episodes
            for i in range(self.n_envs):
                if dones[i]:
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
            
            # Update observations
            observations = next_observations
        
        # Convert to numpy arrays
        rollout_data = {
            'observations': np.array(all_observations),
            'actions': np.array(all_actions),
            'rewards': np.array(all_rewards),
            'values': np.array(all_values),
            'log_probs': np.array(all_log_probs),
            'dones': np.array(all_dones),
            'infos': all_infos
        }
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns(rollout_data)
        rollout_data['advantages'] = advantages
        rollout_data['returns'] = returns
        
        return rollout_data
    
    def _compute_advantages_and_returns(self, rollout_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using GAE."""
        values = rollout_data['values']
        rewards = rollout_data['rewards']
        dones = rollout_data['dones']
        
        # Get final values
        with torch.no_grad():
            final_obs = rollout_data['observations'][-1]
            final_values = self.policy.get_value(final_obs)
        
        # Add final values
        values = np.append(values, final_values, axis=0)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantages[t + 1]
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()
