"""
Off-Policy Evaluation utilities.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class OffPolicyEvaluator:
    """Base class for off-policy evaluation methods."""
    
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
    
    def evaluate(
        self,
        trajectories: List[Dict[str, Any]],
        policy: Any,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate policy using off-policy methods."""
        raise NotImplementedError


class ImportanceSampling(OffPolicyEvaluator):
    """Importance Sampling for off-policy evaluation."""
    
    def __init__(self, gamma: float = 0.99):
        super().__init__(gamma)
    
    def evaluate(
        self,
        trajectories: List[Dict[str, Any]],
        policy: Any,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate using Importance Sampling."""
        logger.info("Running Importance Sampling evaluation")
        
        estimates = []
        
        for traj in trajectories:
            # Calculate importance weights
            weights = self._calculate_importance_weights(traj, policy)
            
            # Calculate weighted return
            returns = traj['rewards']
            weighted_return = np.sum(weights * returns)
            
            estimates.append(weighted_return)
        
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        
        return {
            'mean_estimate': mean_estimate,
            'std_estimate': std_estimate,
            'num_trajectories': len(trajectories),
            'confidence_interval': (mean_estimate - 1.96 * std_estimate, 
                                  mean_estimate + 1.96 * std_estimate)
        }
    
    def _calculate_importance_weights(
        self,
        trajectory: Dict[str, Any],
        policy: Any
    ) -> np.ndarray:
        """Calculate importance weights for trajectory."""
        states = trajectory['observations']
        actions = trajectory['actions']
        
        weights = np.ones(len(actions))
        
        for t in range(len(actions)):
            # Get action probabilities from policy
            state = states[t]
            action = actions[t]
            
            # Calculate importance weight
            # This is a simplified version - in practice, you'd need
            # the behavior policy probabilities
            policy_prob = self._get_action_probability(policy, state, action)
            behavior_prob = 1.0  # Assume uniform behavior policy for now
            
            weights[t] = policy_prob / behavior_prob
        
        return weights
    
    def _get_action_probability(self, policy: Any, state: np.ndarray, action: np.ndarray) -> float:
        """Get action probability from policy."""
        # This is a placeholder - in practice, you'd need to implement
        # the actual policy probability calculation
        return 0.1  # Placeholder


class DoublyRobust(OffPolicyEvaluator):
    """Doubly Robust estimator for off-policy evaluation."""
    
    def __init__(self, gamma: float = 0.99):
        super().__init__(gamma)
    
    def evaluate(
        self,
        trajectories: List[Dict[str, Any]],
        policy: Any,
        value_function: Any = None,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate using Doubly Robust estimator."""
        logger.info("Running Doubly Robust evaluation")
        
        estimates = []
        
        for traj in trajectories:
            # Calculate importance weights
            weights = self._calculate_importance_weights(traj, policy)
            
            # Calculate DR estimate
            dr_estimate = self._calculate_dr_estimate(traj, weights, value_function)
            estimates.append(dr_estimate)
        
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        
        return {
            'mean_estimate': mean_estimate,
            'std_estimate': std_estimate,
            'num_trajectories': len(trajectories),
            'confidence_interval': (mean_estimate - 1.96 * std_estimate, 
                                  mean_estimate + 1.96 * std_estimate)
        }
    
    def _calculate_dr_estimate(
        self,
        trajectory: Dict[str, Any],
        weights: np.ndarray,
        value_function: Any
    ) -> float:
        """Calculate Doubly Robust estimate."""
        states = trajectory['observations']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        
        dr_estimate = 0.0
        
        for t in range(len(rewards)):
            # Direct method component
            if value_function is not None:
                state_value = self._get_state_value(value_function, states[t])
                dr_estimate += state_value
            
            # Importance sampling component
            if t < len(weights):
                dr_estimate += weights[t] * (rewards[t] - self._get_state_value(value_function, states[t]))
        
        return dr_estimate
    
    def _get_state_value(self, value_function: Any, state: np.ndarray) -> float:
        """Get state value from value function."""
        # This is a placeholder - in practice, you'd need to implement
        # the actual value function evaluation
        return 0.0  # Placeholder
    
    def _calculate_importance_weights(
        self,
        trajectory: Dict[str, Any],
        policy: Any
    ) -> np.ndarray:
        """Calculate importance weights for trajectory."""
        # Same as ImportanceSampling
        states = trajectory['observations']
        actions = trajectory['actions']
        
        weights = np.ones(len(actions))
        
        for t in range(len(actions)):
            state = states[t]
            action = actions[t]
            
            policy_prob = self._get_action_probability(policy, state, action)
            behavior_prob = 1.0  # Assume uniform behavior policy for now
            
            weights[t] = policy_prob / behavior_prob
        
        return weights
    
    def _get_action_probability(self, policy: Any, state: np.ndarray, action: np.ndarray) -> float:
        """Get action probability from policy."""
        return 0.1  # Placeholder


def run_off_policy_evaluation(
    trajectories: List[Dict[str, Any]],
    policy: Any,
    methods: List[str] = ['is', 'dr'],
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """Run multiple off-policy evaluation methods."""
    results = {}
    
    for method in methods:
        if method.lower() == 'is':
            evaluator = ImportanceSampling(**kwargs)
        elif method.lower() == 'dr':
            evaluator = DoublyRobust(**kwargs)
        else:
            logger.warning(f"Unknown evaluation method: {method}")
            continue
        
        try:
            results[method] = evaluator.evaluate(trajectories, policy, **kwargs)
        except Exception as e:
            logger.error(f"Failed to run {method} evaluation: {e}")
            results[method] = {'error': str(e)}
    
    return results
