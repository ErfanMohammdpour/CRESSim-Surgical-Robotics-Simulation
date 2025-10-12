"""
Model evaluator for testing trained models.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import cv2
from datetime import datetime

from ..utils.gpu import get_device
from ..envs.mock_env import MockSuctionEnv
from ..safety.safety_shield import SafetyShield

logger = logging.getLogger(__name__)


class Evaluator:
    """Model evaluator for testing trained models."""
    
    def __init__(self, checkpoint_path: str, output_dir: str):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = get_device()
        
        # Load model
        self.model = self._load_model()
        
        # Setup environment
        self.env = MockSuctionEnv(
            image_size=(128, 128),
            max_steps=1000
        )
        
        # Setup safety shield
        self.safety_shield = SafetyShield({
            "d_safe": 0.05,
            "segmentation_threshold": 0.5,
            "projection_scaling_factor": 0.8
        })
        
        logger.info(f"Evaluator initialized with model: {checkpoint_path}")
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        try:
            # Try loading as stable-baselines3 model
            from stable_baselines3 import PPO
            model = PPO.load(self.checkpoint_path, device=self.device)
            logger.info("Loaded PPO model from stable-baselines3")
            return model
        except Exception as e:
            logger.warning(f"Failed to load as PPO model: {e}")
            # Try loading as PyTorch model
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                logger.info("Loaded PyTorch model")
                return checkpoint
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """Evaluate model on multiple episodes."""
        logger.info(f"Starting evaluation on {num_episodes} episodes...")
        
        episode_results = []
        all_rewards = []
        all_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            
            # Run episode
            episode_result = self._run_episode(episode, render)
            episode_results.append(episode_result)
            
            # Collect metrics
            all_rewards.append(episode_result['total_reward'])
            all_lengths.append(episode_result['episode_length'])
            
            if episode_result['success']:
                success_count += 1
            
            logger.info(f"Episode {episode + 1}: Reward={episode_result['total_reward']:.3f}, "
                       f"Length={episode_result['episode_length']}, Success={episode_result['success']}")
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(
            episode_results, all_rewards, all_lengths, success_count, num_episodes
        )
        
        # Save results
        self._save_results(episode_results, summary)
        
        logger.info(f"Evaluation completed! Success rate: {summary['success_rate']:.2%}")
        return summary
    
    def _run_episode(self, episode: int, render: bool = False) -> Dict[str, Any]:
        """Run a single episode."""
        obs, info = self.env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        # Video recording
        frames = [] if render else None
        
        while not done:
            # Get action from model
            if hasattr(self.model, 'predict'):
                # Stable-baselines3 model
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                # PyTorch model (simplified)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.model(obs_tensor).cpu().numpy().squeeze()
            
            # Apply safety shield
            safe_action = self.safety_shield.apply_shield(obs, action)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(safe_action)
            total_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Record frame if rendering
            if render and frames is not None:
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
        
        # Save video if rendering
        if render and frames:
            self._save_video(frames, episode)
        
        # Determine success
        success = self._is_success(info)
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'success': success,
            'final_info': info,
            'safety_violations': self.safety_shield.get_safety_stats()['violation_count']
        }
    
    def _is_success(self, info: Dict[str, Any]) -> bool:
        """Determine if episode was successful."""
        liquid_reduction = info.get('liquid_reduction', 0)
        contaminant_reduction = info.get('contaminant_reduction', 0)
        
        # Success if both liquid and contaminant are mostly removed
        return liquid_reduction > 0.8 and contaminant_reduction > 0.8
    
    def _calculate_summary_stats(self, episode_results: List[Dict], all_rewards: List[float], 
                                all_lengths: List[int], success_count: int, num_episodes: int) -> Dict[str, Any]:
        """Calculate summary statistics."""
        return {
            'num_episodes': num_episodes,
            'success_count': success_count,
            'success_rate': success_count / num_episodes,
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'mean_length': np.mean(all_lengths),
            'std_length': np.std(all_lengths),
            'min_length': np.min(all_lengths),
            'max_length': np.max(all_lengths),
            'total_safety_violations': sum(r['safety_violations'] for r in episode_results),
            'episode_results': episode_results
        }
    
    def _save_results(self, episode_results: List[Dict], summary: Dict[str, Any]):
        """Save evaluation results."""
        # Save detailed results
        results_file = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'episode_results': episode_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save summary CSV
        csv_file = self.output_dir / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        import pandas as pd
        df = pd.DataFrame([summary])
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {csv_file}")
    
    def _save_video(self, frames: List[np.ndarray], episode: int):
        """Save video of episode."""
        if not frames:
            return
        
        video_file = self.output_dir / f"episode_{episode:03d}.mp4"
        
        # Get video properties
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        logger.info(f"Video saved to {video_file}")


if __name__ == "__main__":
    # Test evaluator
    evaluator = Evaluator(
        checkpoint_path="data/checkpoints/final_ppo_model.pth",
        output_dir="data/videos"
    )
    
    results = evaluator.evaluate(num_episodes=5, render=True)
    print(f"Evaluation results: {results}")