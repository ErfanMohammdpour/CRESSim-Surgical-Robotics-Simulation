"""
Evaluation metrics and utilities.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculate evaluation metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_episode_metrics(
        self,
        episode_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate metrics for single episode."""
        metrics = {}
        
        # Basic episode metrics
        metrics['episode_length'] = episode_data['length']
        metrics['total_reward'] = np.sum(episode_data['rewards'])
        metrics['mean_reward'] = np.mean(episode_data['rewards'])
        
        # Task-specific metrics
        final_info = episode_data['infos'][-1]
        metrics['liquid_reduction'] = final_info.get('liquid_reduction', 0)
        metrics['contaminant_reduction'] = final_info.get('contaminant_reduction', 0)
        metrics['collision_count'] = final_info.get('collision_count', 0)
        metrics['safety_violations'] = final_info.get('safety_violations', 0)
        
        # Success criteria
        metrics['success'] = (
            metrics['liquid_reduction'] > 0.8 and
            metrics['contaminant_reduction'] > 0.8 and
            metrics['safety_violations'] < 5
        )
        
        # Action metrics
        actions = episode_data['actions']
        metrics['action_variance'] = np.var(actions, axis=0).mean()
        metrics['action_smoothness'] = 1.0 / (1.0 + metrics['action_variance'])
        
        return metrics
    
    def calculate_aggregate_metrics(
        self,
        episode_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across episodes."""
        if not episode_metrics:
            return {}
        
        # Extract arrays for calculation
        episode_lengths = [m['episode_length'] for m in episode_metrics]
        total_rewards = [m['total_reward'] for m in episode_metrics]
        liquid_reductions = [m['liquid_reduction'] for m in episode_metrics]
        contaminant_reductions = [m['contaminant_reduction'] for m in episode_metrics]
        collision_counts = [m['collision_count'] for m in episode_metrics]
        safety_violations = [m['safety_violations'] for m in episode_metrics]
        successes = [m['success'] for m in episode_metrics]
        
        # Calculate aggregate metrics
        aggregate = {
            'num_episodes': len(episode_metrics),
            'success_rate': np.mean(successes),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'mean_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'mean_liquid_reduction': np.mean(liquid_reductions),
            'std_liquid_reduction': np.std(liquid_reductions),
            'mean_contaminant_reduction': np.mean(contaminant_reductions),
            'std_contaminant_reduction': np.std(contaminant_reductions),
            'mean_collision_count': np.mean(collision_counts),
            'std_collision_count': np.std(collision_counts),
            'mean_safety_violations': np.mean(safety_violations),
            'std_safety_violations': np.std(safety_violations),
        }
        
        return aggregate


class ModelEvaluator:
    """Evaluate trained models."""
    
    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        device: str = "cuda"
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"ModelEvaluator initialized: {checkpoint_path}")
    
    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Determine model type and load accordingly
        if 'model_state_dict' in checkpoint:
            # PyTorch model
            from ..il.bc_trainer import PolicyNetwork
            from ..vision.encoders import CNNEncoder
            
            encoder = CNNEncoder(
                input_channels=3,
                output_dim=512,
                channels=[32, 64, 128, 256],
                kernel_sizes=[3, 3, 3, 3],
                strides=[2, 2, 2, 2],
                padding=[1, 1, 1, 1]
            )
            
            model = PolicyNetwork(encoder=encoder, hidden_dim=512, action_dim=5)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            return model
        else:
            # Stable-Baselines3 model
            from stable_baselines3 import PPO
            model = PPO.load(str(self.checkpoint_path))
            return model
    
    def evaluate(
        self,
        num_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on environment."""
        logger.info(f"Evaluating model for {num_episodes} episodes")
        
        # Create environment
        from ..envs.mock_env import MockSuctionEnv
        env = MockSuctionEnv(image_size=(128, 128), max_steps=1000)
        
        # Evaluation metrics
        metrics_calc = EvaluationMetrics()
        episode_metrics = []
        episode_data = []
        
        # Video recording
        if render:
            from ..utils.video import VideoRecorder
            video_recorder = VideoRecorder(
                self.output_dir / "evaluation_video.mp4",
                fps=30
            )
        
        for episode_id in range(num_episodes):
            logger.info(f"Evaluating episode {episode_id + 1}/{num_episodes}")
            
            # Reset environment
            obs, info = env.reset()
            
            # Episode data
            observations = []
            actions = []
            rewards = []
            infos = []
            
            # Video recording
            if render:
                video_recorder.start_recording()
            
            # Run episode
            done = False
            step = 0
            
            while not done and step < 1000:
                # Get action from model
                if hasattr(self.model, 'predict'):
                    # Stable-Baselines3 model
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    # PyTorch model
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
                        action_tensor = self.model(obs_tensor)
                        action = action_tensor.cpu().numpy()[0]
                
                # Step environment
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                
                # Store data
                observations.append(obs.copy())
                actions.append(action.copy())
                rewards.append(reward)
                infos.append(info.copy())
                
                # Update state
                obs = next_obs
                info = next_info
                done = terminated or truncated
                step += 1
                
                # Record frame
                if render:
                    video_recorder.add_frame(obs)
            
            # Stop video recording
            if render:
                video_recorder.stop_recording()
            
            # Calculate episode metrics
            episode_dict = {
                'episode_id': episode_id,
                'observations': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'infos': infos,
                'length': len(observations)
            }
            
            episode_metrics.append(metrics_calc.calculate_episode_metrics(episode_dict))
            episode_data.append(episode_dict)
        
        # Calculate aggregate metrics
        aggregate_metrics = metrics_calc.calculate_aggregate_metrics(episode_metrics)
        
        # Save results
        results = {
            'episode_metrics': episode_metrics,
            'aggregate_metrics': aggregate_metrics,
            'episode_data': episode_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to files
        self._save_results(results)
        
        logger.info(f"Evaluation completed. Success rate: {aggregate_metrics.get('success_rate', 0):.2%}")
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results['aggregate_metrics'], f, indent=2)
        
        # Save per-episode data
        per_episode_file = self.output_dir / "per_episode.csv"
        import pandas as pd
        df = pd.DataFrame(results['episode_metrics'])
        df.to_csv(per_episode_file, index=False)
        
        # Save detailed results
        detailed_file = self.output_dir / "detailed_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
