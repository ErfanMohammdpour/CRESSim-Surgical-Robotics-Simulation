"""
Experience replay buffers for RL training.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Standard experience replay buffer.
    """
    
    def __init__(
        self,
        capacity: int,
        obs_shape: Dict[str, Tuple[int, ...]],
        action_dim: int,
        device: str = "cuda"
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        
        # Initialize buffers
        self.observations = {}
        self.next_observations = {}
        for key, shape in obs_shape.items():
            self.observations[key] = np.zeros((capacity, *shape), dtype=np.float32)
            self.next_observations[key] = np.zeros((capacity, *shape), dtype=np.float32)
        
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Buffer state
        self.size = 0
        self.ptr = 0
    
    def add(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray, 
        reward: float, 
        next_obs: Dict[str, np.ndarray], 
        done: bool
    ) -> None:
        """Add experience to buffer."""
        for key in self.obs_shape.keys():
            self.observations[key][self.ptr] = obs[key]
            self.next_observations[key][self.ptr] = next_obs[key]
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {}
        for key in self.obs_shape.keys():
            batch[key] = torch.FloatTensor(self.observations[key][indices]).to(self.device)
            batch[f'next_{key}'] = torch.FloatTensor(self.next_observations[key][indices]).to(self.device)
        
        batch['actions'] = torch.FloatTensor(self.actions[indices]).to(self.device)
        batch['rewards'] = torch.FloatTensor(self.rewards[indices]).to(self.device)
        batch['dones'] = torch.BoolTensor(self.dones[indices]).to(self.device)
        
        return batch
    
    def __len__(self):
        return self.size


class PPOBuffer:
    """
    Buffer for PPO training with GAE computation.
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Dict[str, Tuple[int, ...]],
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cuda"
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Initialize buffers
        self.observations = {}
        for key, shape in obs_shape.items():
            self.observations[key] = np.zeros((buffer_size, *shape), dtype=np.float32)
        
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # Buffer state
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
    
    def add(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray, 
        reward: float, 
        value: float, 
        log_prob: float, 
        done: bool
    ) -> None:
        """Add experience to buffer."""
        assert self.ptr < self.max_size, "Buffer overflow"
        
        for key in self.obs_shape.keys():
            self.observations[key][self.ptr] = obs[key]
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0) -> None:
        """Finish current path and compute advantages."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = self._compute_gae(deltas)
        
        # Store advantages and returns
        self.advantages = advantages
        self.returns = advantages + self.values[path_slice]
        
        self.path_start_idx = self.ptr
    
    def _compute_gae(self, deltas: np.ndarray) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(deltas)
        last_advantage = 0
        
        for t in reversed(range(len(deltas))):
            last_advantage = deltas[t] + self.gamma * self.gae_lambda * last_advantage
            advantages[t] = last_advantage
        
        return advantages
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data from buffer."""
        assert self.ptr == self.max_size, "Buffer not full"
        
        # Normalize advantages
        advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        
        batch = {}
        for key in self.obs_shape.keys():
            batch[key] = torch.FloatTensor(self.observations[key]).to(self.device)
        
        batch['actions'] = torch.FloatTensor(self.actions).to(self.device)
        batch['rewards'] = torch.FloatTensor(self.rewards).to(self.device)
        batch['values'] = torch.FloatTensor(self.values).to(self.device)
        batch['log_probs'] = torch.FloatTensor(self.log_probs).to(self.device)
        batch['advantages'] = torch.FloatTensor(advantages).to(self.device)
        batch['returns'] = torch.FloatTensor(self.returns).to(self.device)
        batch['dones'] = torch.BoolTensor(self.dones).to(self.device)
        
        return batch
    
    def clear(self) -> None:
        """Clear buffer."""
        self.ptr = 0
        self.path_start_idx = 0


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    """
    
    def __init__(
        self,
        capacity: int,
        obs_shape: Dict[str, Tuple[int, ...]],
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        device: str = "cuda"
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        
        # Initialize buffers
        self.observations = {}
        self.next_observations = {}
        for key, shape in obs_shape.items():
            self.observations[key] = np.zeros((capacity, *shape), dtype=np.float32)
            self.next_observations[key] = np.zeros((capacity, *shape), dtype=np.float32)
        
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Priority buffer
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        # Buffer state
        self.size = 0
        self.ptr = 0
    
    def add(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray, 
        reward: float, 
        next_obs: Dict[str, np.ndarray], 
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """Add experience to buffer."""
        if priority is None:
            priority = self.max_priority
        
        for key in self.obs_shape.keys():
            self.observations[key][self.ptr] = obs[key]
            self.next_observations[key][self.ptr] = next_obs[key]
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = priority
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample batch from buffer with importance sampling weights."""
        if self.size == 0:
            return {}, np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Create batch
        batch = {}
        for key in self.obs_shape.keys():
            batch[key] = torch.FloatTensor(self.observations[key][indices]).to(self.device)
            batch[f'next_{key}'] = torch.FloatTensor(self.next_observations[key][indices]).to(self.device)
        
        batch['actions'] = torch.FloatTensor(self.actions[indices]).to(self.device)
        batch['rewards'] = torch.FloatTensor(self.rewards[indices]).to(self.device)
        batch['dones'] = torch.BoolTensor(self.dones[indices]).to(self.device)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.size


class RolloutBuffer:
    """
    Buffer for collecting rollouts during training.
    """
    
    def __init__(
        self,
        obs_shape: Dict[str, Tuple[int, ...]],
        action_dim: int,
        device: str = "cuda"
    ):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        
        # Current rollout
        self.observations = {key: [] for key in obs_shape.keys()}
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.infos = []
        
        # Rollout state
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def add(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray, 
        reward: float, 
        value: float, 
        log_prob: float, 
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        """Add step to current rollout."""
        for key in self.obs_shape.keys():
            self.observations[key].append(obs[key])
        
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.infos.append(info)
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
    
    def get_rollout_data(self) -> Dict[str, torch.Tensor]:
        """Get current rollout data as tensors."""
        rollout_data = {}
        
        for key in self.obs_shape.keys():
            rollout_data[key] = torch.FloatTensor(np.array(self.observations[key])).to(self.device)
        
        rollout_data['actions'] = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rollout_data['rewards'] = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        rollout_data['values'] = torch.FloatTensor(np.array(self.values)).to(self.device)
        rollout_data['log_probs'] = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rollout_data['dones'] = torch.BoolTensor(np.array(self.dones)).to(self.device)
        
        return rollout_data
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'std_episode_length': np.std(self.episode_lengths),
            'num_episodes': len(self.episode_rewards)
        }
    
    def clear(self) -> None:
        """Clear current rollout."""
        for key in self.obs_shape.keys():
            self.observations[key].clear()
        
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.infos.clear()
    
    def clear_episodes(self) -> None:
        """Clear episode statistics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.current_episode_reward = 0
        self.current_episode_length = 0
