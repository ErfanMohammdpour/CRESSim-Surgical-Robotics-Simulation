"""
Main evaluator for trained models with GPU support.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime

from .metrics import ModelEvaluator, EvaluationMetrics
from .report import generate_evaluation_report
from ..envs.mock_env import MockSuctionEnv
from ..envs.unity_env import create_environment
from ..utils.video import VideoRecorder
from ..utils.seeding import set_seed
from ..utils.device import (
    get_device_from_config, log_device_info, optimize_for_gpu, 
    amp_enabled, log_memory_usage
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator for trained models with GPU support."""
    
    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Setup device
        self.device = get_device_from_config(self.config)
        log_device_info(self.device, self.config)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model evaluator
        self.model_evaluator = ModelEvaluator(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=self.device,
            config=self.config
        )
        
        logger.info(f"Evaluator initialized: {checkpoint_path}")
        logger.info(f"AMP enabled: {amp_enabled(self.config)}")
    
    def evaluate(
        self,
        num_episodes: int = 10,
        render: bool = False,
        mock: bool = False
    ) -> Dict[str, Any]:
        """Evaluate the model."""
        logger.info(f"Starting evaluation: {num_episodes} episodes, render={render}, mock={mock}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create environment
        env_config = self.config.get("env", {})
        env_config["use_mock"] = mock
        
        if mock:
            env = MockSuctionEnv(env_config)
        else:
            env = create_environment(env_config)
        
        # Setup video recording if requested
        video_recorder = None
        if render:
            video_dir = self.output_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            video_recorder = VideoRecorder(video_dir)
        
        # Evaluation results
        episode_results = []
        all_observations = []
        all_actions = []
        all_rewards = []
        
        # Run episodes
        for episode in range(num_episodes):
            logger.info(f"Running episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_observations = []
            episode_actions = []
            episode_rewards = []
            
            # Start video recording
            if video_recorder:
                video_recorder.start_recording(f"episode_{episode}")
            
            done = False
            while not done:
                # Get action from model
                action = self.model_evaluator.predict(obs)
                
                # Store data
                episode_observations.append(obs.copy())
                episode_actions.append(action.copy())
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_rewards.append(reward)
                
                # Record frame
                if video_recorder:
                    video_recorder.record_frame(obs)
            
            # Stop video recording
            if video_recorder:
                video_recorder.stop_recording()
            
            # Store episode results
            episode_result = {
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'success': episode_reward > 0,
                'info': info
            }
            episode_results.append(episode_result)
            
            # Store data for analysis
            all_observations.extend(episode_observations)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            
            logger.info(
                f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                f"Length={episode_length}, Success={episode_result['success']}"
            )
            
            # Log memory usage periodically
            if episode % 5 == 0:
                log_memory_usage(self.device, f"Episode {episode}")
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(episode_results)
        
        # Create evaluation results
        results = {
            'episode_results': episode_results,
            'aggregate_metrics': aggregate_metrics,
            'observations': all_observations,
            'actions': all_actions,
            'rewards': all_rewards,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(results)
        
        # Generate report
        report_path = self._generate_report(results)
        
        logger.info(f"Evaluation completed. Results saved to {self.output_dir}")
        logger.info(f"Report generated: {report_path}")
        
        return results
    
    def _calculate_aggregate_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from episode results."""
        rewards = [r['reward'] for r in episode_results]
        lengths = [r['length'] for r in episode_results]
        successes = [r['success'] for r in episode_results]
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': np.mean(successes),
            'total_episodes': len(episode_results)
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        # Save JSON results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save episode CSV
        episode_data = []
        for result in results['episode_results']:
            episode_data.append({
                'episode': result['episode'],
                'reward': result['reward'],
                'length': result['length'],
                'success': result['success']
            })
        
        import pandas as pd
        df = pd.DataFrame(episode_data)
        csv_path = self.output_dir / "per_episode.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {results_path} and {csv_path}")
    
    def _generate_report(self, results: Dict[str, Any]) -> Path:
        """Generate evaluation report."""
        report_path = self.output_dir / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Evaluation Report\n\n")
            f.write(f"**Timestamp:** {results['timestamp']}\n")
            f.write(f"**Checkpoint:** {self.checkpoint_path}\n")
            f.write(f"**Device:** {self.device}\n\n")
            
            # Aggregate metrics
            f.write("## Aggregate Metrics\n\n")
            metrics = results['aggregate_metrics']
            f.write(f"- **Average Reward:** {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}\n")
            f.write(f"- **Success Rate:** {metrics['success_rate']:.2%}\n")
            f.write(f"- **Average Length:** {metrics['avg_length']:.1f} ± {metrics['std_length']:.1f}\n")
            f.write(f"- **Total Episodes:** {metrics['total_episodes']}\n\n")
            
            # Episode details
            f.write("## Episode Details\n\n")
            f.write("| Episode | Reward | Length | Success |\n")
            f.write("|---------|--------|--------|----------|\n")
            
            for result in results['episode_results']:
                f.write(f"| {result['episode']} | {result['reward']:.2f} | {result['length']} | {result['success']} |\n")
            
            f.write("\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write("```yaml\n")
            import yaml
            yaml.dump(results['config'], f, default_flow_style=False)
            f.write("```\n")
        
        logger.info(f"Report generated: {report_path}")
        return report_path


def evaluate_model(
    checkpoint_path: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    num_episodes: int = 10,
    render: bool = False,
    mock: bool = False
) -> Dict[str, Any]:
    """Evaluate a trained model."""
    evaluator = Evaluator(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        config=config
    )
    
    return evaluator.evaluate(
        num_episodes=num_episodes,
        render=render,
        mock=mock
    )