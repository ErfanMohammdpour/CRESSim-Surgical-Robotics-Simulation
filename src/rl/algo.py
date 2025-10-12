"""
Reinforcement learning algorithms (PPO, SAC).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PPOPolicy(nn.Module):
    """
    PPO policy network with actor-critic architecture.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int = 5,
        hidden_dim: int = 256,
        activation: str = "relu",
        log_std_init: float = -0.5
    ):
        super().__init__()
        
        self.encoder = encoder
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Log standard deviation for continuous actions
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, image: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Encode observation
        features = self.encoder(image, aux)
        
        # Actor output
        action_mean = self.actor(features)
        action_std = torch.exp(self.log_std)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, image: torch.Tensor, aux: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        action_mean, action_std, value = self.forward(image, aux)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            # Create normal distribution
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_actions(self, image: torch.Tensor, aux: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_mean, action_std, value = self.forward(image, aux)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class SACPolicy(nn.Module):
    """
    SAC policy network with actor-critic architecture.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int = 5,
        hidden_dim: int = 256,
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        
        self.encoder = encoder
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU() if activation == "relu" else nn.ELU(),
        )
        
        # Action mean and log std
        self.action_mean = nn.Linear(hidden_dim // 2, action_dim)
        self.action_log_std = nn.Linear(hidden_dim // 2, action_dim)
        
        # Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU() if activation == "relu" else nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, image: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for actor."""
        # Encode observation
        features = self.encoder(image, aux)
        
        # Actor output
        actor_features = self.actor(features)
        action_mean = self.action_mean(actor_features)
        action_log_std = self.action_log_std(actor_features)
        
        # Clamp log std
        action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        
        return action_mean, action_log_std
    
    def get_action(self, image: torch.Tensor, aux: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        action_mean, action_log_std = self.forward(image, aux)
        action_std = torch.exp(action_log_std)
        
        if deterministic:
            action = torch.tanh(action_mean)
            log_prob = None
        else:
            # Create normal distribution
            dist = Normal(action_mean, action_std)
            action_raw = dist.sample()
            action = torch.tanh(action_raw)
            
            # Calculate log probability with tanh transformation
            log_prob = dist.log_prob(action_raw).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self, image: torch.Tensor, aux: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions for SAC update."""
        action_mean, action_log_std = self.forward(image, aux)
        action_std = torch.exp(action_log_std)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        action_raw = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(action_raw).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return log_prob, dist.entropy().sum(dim=-1, keepdim=True)
    
    def get_q_values(self, image: torch.Tensor, aux: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values for given state-action pairs."""
        # Encode observation
        features = self.encoder(image, aux)
        
        # Concatenate features and action
        q_input = torch.cat([features, action], dim=-1)
        
        # Get Q-values
        q1_value = self.q1(q_input)
        q2_value = self.q2(q_input)
        
        return q1_value, q2_value


class PPO:
    """
    Proximal Policy Optimization algorithm.
    """
    
    def __init__(
        self,
        policy: PPOPolicy,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda"
    ):
        self.policy = policy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Compute advantages in reverse order
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def update(
        self, 
        obs: Dict[str, torch.Tensor], 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
        n_epochs: int = 10
    ) -> Dict[str, float]:
        """Update policy using PPO."""
        # Move to device
        image = obs['image'].to(self.device)
        aux = obs['aux'].to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # Get current policy outputs
        log_probs, values, entropy = self.policy.evaluate_actions(image, aux, actions)
        
        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values.squeeze(), dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        for _ in range(n_epochs):
            # Policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1)).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss / n_epochs,
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item()
        }


class SAC:
    """
    Soft Actor-Critic algorithm.
    """
    
    def __init__(
        self,
        policy: SACPolicy,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cuda"
    ):
        self.policy = policy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        
        # Target networks
        self.target_policy = SACPolicy(
            policy.encoder, 
            policy.action_dim, 
            policy.hidden_dim
        ).to(device)
        self.target_policy.load_state_dict(policy.state_dict())
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.q1_optimizer = torch.optim.Adam(policy.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = torch.optim.Adam(policy.q2.parameters(), lr=learning_rate)
        
        # Automatic entropy tuning
        self.target_entropy = -policy.action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
    
    def update(
        self, 
        obs: Dict[str, torch.Tensor], 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        next_obs: Dict[str, torch.Tensor], 
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update SAC networks."""
        # Move to device
        image = obs['image'].to(self.device)
        aux = obs['aux'].to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_image = next_obs['image'].to(self.device)
        next_aux = next_obs['aux'].to(self.device)
        dones = dones.to(self.device)
        
        # Update Q-networks
        with torch.no_grad():
            next_actions, next_log_probs = self.target_policy.get_action(next_image, next_aux)
            next_q1, next_q2 = self.target_policy.get_q_values(next_image, next_aux, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * next_q.squeeze()
        
        # Q1 loss
        current_q1, _ = self.policy.get_q_values(image, aux, actions)
        q1_loss = F.mse_loss(current_q1.squeeze(), target_q)
        
        # Q2 loss
        current_q2, _ = self.policy.get_q_values(image, aux, actions)
        q2_loss = F.mse_loss(current_q2.squeeze(), target_q)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.get_action(image, aux)
        q1_new, q2_new = self.policy.get_q_values(image, aux, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        self._soft_update(self.policy, self.target_policy, self.tau)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
