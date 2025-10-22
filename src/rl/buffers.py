"""
Experience replay buffers for reinforcement learning.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader


class ReplayBuffer:
    """Standard experience replay buffer."""
    
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Storage arrays
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Additional info
        self.infos = [None] * capacity
    
    def add(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_obs: np.ndarray, 
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add experience to buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.infos[self.ptr] = info
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of experiences."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'obs': self.obs[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_obs': self.next_obs[indices],
            'dones': self.dones[indices],
            'infos': [self.infos[i] for i in indices]
        }
    
    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    
    def __init__(
        self, 
        capacity: int, 
        obs_shape: Tuple[int, ...], 
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.ptr = 0
        self.size = 0
        
        # Storage arrays
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        # Additional info
        self.infos = [None] * capacity
    
    def add(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_obs: np.ndarray, 
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add experience to buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.infos[self.ptr] = info
        
        # Set priority to max priority
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        if self.size == 0:
            return {}, np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = {
            'obs': self.obs[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_obs': self.next_obs[indices],
            'dones': self.dones[indices],
            'infos': [self.infos[i] for i in indices]
        }
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        return self.size


class SafetyBuffer:
    """Buffer specifically for safety-related experiences."""
    
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Storage arrays
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.safety_masks = np.zeros((capacity, obs_shape[0], obs_shape[1]), dtype=np.float32)
        self.distances = np.zeros(capacity, dtype=np.float32)
        self.safety_violations = np.zeros(capacity, dtype=np.bool_)
        
        # Safety labels
        self.safety_labels = np.zeros(capacity, dtype=np.int32)  # 0: safe, 1: warning, 2: critical
    
    def add(
        self, 
        obs: np.ndarray, 
        safety_mask: np.ndarray, 
        distance: float, 
        violation: bool,
        safety_label: int
    ) -> None:
        """Add safety experience to buffer."""
        self.obs[self.ptr] = obs
        self.safety_masks[self.ptr] = safety_mask
        self.distances[self.ptr] = distance
        self.safety_violations[self.ptr] = violation
        self.safety_labels[self.ptr] = safety_label
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of safety experiences."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'obs': self.obs[indices],
            'safety_masks': self.safety_masks[indices],
            'distances': self.distances[indices],
            'violations': self.safety_violations[indices],
            'labels': self.safety_labels[indices]
        }
    
    def get_violation_ratio(self) -> float:
        """Get ratio of safety violations."""
        if self.size == 0:
            return 0.0
        return np.sum(self.safety_violations[:self.size]) / self.size
    
    def __len__(self) -> int:
        return self.size


class DemonstrationBuffer:
    """Buffer for storing and sampling demonstrations."""
    
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Storage arrays
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.weights = np.zeros(capacity, dtype=np.float32)  # Quality weights
        
        # Episode boundaries
        self.episode_starts = []
        self.episode_lengths = []
    
    def add_episode(
        self, 
        obs: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> None:
        """Add complete episode to buffer."""
        episode_length = len(obs)
        
        if self.ptr + episode_length > self.capacity:
            # Wrap around
            remaining = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:remaining]
            self.actions[self.ptr:] = actions[:remaining]
            self.rewards[self.ptr:] = rewards[:remaining]
            if weights is not None:
                self.weights[self.ptr:] = weights[:remaining]
            
            self.obs[:episode_length - remaining] = obs[remaining:]
            self.actions[:episode_length - remaining] = actions[remaining:]
            self.rewards[:episode_length - remaining] = rewards[remaining:]
            if weights is not None:
                self.weights[:episode_length - remaining] = weights[remaining:]
            
            self.ptr = episode_length - remaining
        else:
            self.obs[self.ptr:self.ptr + episode_length] = obs
            self.actions[self.ptr:self.ptr + episode_length] = actions
            self.rewards[self.ptr:self.ptr + episode_length] = rewards
            if weights is not None:
                self.weights[self.ptr:self.ptr + episode_length] = weights
            
            self.ptr = (self.ptr + episode_length) % self.capacity
        
        self.size = min(self.size + episode_length, self.capacity)
        self.episode_starts.append(self.ptr - episode_length)
        self.episode_lengths.append(episode_length)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of demonstration experiences."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'obs': self.obs[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'weights': self.weights[indices]
        }
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics about stored episodes."""
        if not self.episode_lengths:
            return {}
        
        return {
            'num_episodes': len(self.episode_lengths),
            'avg_episode_length': np.mean(self.episode_lengths),
            'total_transitions': self.size,
            'avg_reward': np.mean(self.rewards[:self.size]),
            'avg_weight': np.mean(self.weights[:self.size])
        }
    
    def __len__(self) -> int:
        return self.size


class TorchDataset(Dataset):
    """PyTorch dataset wrapper for replay buffer."""
    
    def __init__(self, buffer: ReplayBuffer):
        self.buffer = buffer
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index."""
        return {
            'obs': torch.from_numpy(self.buffer.obs[idx]),
            'action': torch.from_numpy(self.buffer.actions[idx]),
            'reward': torch.tensor(self.buffer.rewards[idx]),
            'next_obs': torch.from_numpy(self.buffer.next_obs[idx]),
            'done': torch.tensor(self.buffer.dones[idx], dtype=torch.bool)
        }


def create_dataloader(
    buffer: ReplayBuffer, 
    batch_size: int, 
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create PyTorch DataLoader from replay buffer."""
    dataset = TorchDataset(buffer)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
