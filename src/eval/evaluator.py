"""
Model evaluator for comprehensive performance assessment.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import cv2
from tqdm import tqdm

from ..vision.encoders import create_encoder
from ..vision.safety_seg import create_safety_network, create_safety_shield
from ..envs.unity_env import UnityEnv, MockUnityEnv
from ..envs.wrappers import FrameStackWrapper, RewardShapingWrapper, DomainRandomizationWrapper
from ..rl.safety_shield import RLSafetyShield
from .metrics import MetricsCalculator
from .plots import PlotGenerator
from ..utils.video import VideoRecorder
from ..utils.io import ensure_dir, load_config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        # Load configuration
        if config is None:
            config = load_config("configs/train.yaml")
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._load_model()
        self._initialize_environment()
        self._initialize_safety_components()
        
        # Evaluation components
        self.metrics_calculator = MetricsCalculator()
        self.plot_generator = PlotGenerator(str(self.output_dir))
        self.video_recorder = VideoRecorder(str(self.output_dir / "videos"))
    
    def _load_model(self) -> None:
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Create model architecture
        self.encoder = create_encoder(self.config.get('model', {}))
        
        # Load policy
        if self.config.get('algorithm', 'ppo') == 'ppo':
            from ..rl.algo import PPOPolicy
            self.policy = PPOPolicy(
                encoder=self.encoder,
                action_dim=5,
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 256)
            )
        else:
            from ..rl.algo import SACPolicy
            self.policy = SACPolicy(
                encoder=self.encoder,
                action_dim=5,
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 256)
            )
        
        # Load state dict
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        
        logger.info("Model loaded successfully")
    
    def _initialize_environment(self) -> None:
        """Initialize evaluation environment."""
        # Load paths config
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
        
        logger.info("Environment initialized")
    
    def _wrap_environment(self, env):
        """Wrap environment with necessary wrappers."""
        # Frame stacking
        env = FrameStackWrapper(env, num_frames=4)
        
        # Reward shaping
        env_config = load_config("configs/env.yaml")
        env = RewardShapingWrapper(env, env_config)
        
        # Domain randomization (disabled for evaluation)
        env = DomainRandomizationWrapper(env, env_config)
        
        return env
    
    def _initialize_safety_components(self) -> None:
        """Initialize safety components."""
        # Create safety network
        safety_config = load_config("configs/safety.yaml")
        self.safety_net = create_safety_network(safety_config)
        
        # Load safety network weights if available
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if 'safety_net_state_dict' in checkpoint:
            self.safety_net.load_state_dict(checkpoint['safety_net_state_dict'])
        
        # Create safety shield
        self.safety_shield = RLSafetyShield(
            safety_net=self.safety_net,
            config=safety_config,
            device=self.device
        )
        
        logger.info("Safety components initialized")
    
    def evaluate(
        self, 
        num_episodes: int = 10, 
        render: bool = False,
        save_videos: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model performance."""
        logger.info(f"Starting evaluation with {num_episodes} episodes")
        
        # Reset metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Evaluation loop
        for episode_idx in tqdm(range(num_episodes), desc="Evaluating"):
            episode_data = self._run_episode(episode_idx, render, save_videos)
            self.metrics_calculator.add_episode(episode_data)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics()
        
        # Generate plots
        self._generate_plots()
        
        # Save results
        self._save_results(metrics)
        
        logger.info("Evaluation completed")
        return metrics
    
    def _run_episode(
        self, 
        episode_idx: int, 
        render: bool = False, 
        save_video: bool = True
    ) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        # Reset environment
        obs, info = self.env.reset()
        self.safety_shield.reset()
        
        # Episode data
        episode_data = {
            'episode_id': episode_idx,
            'actions': [],
            'rewards': [],
            'observations': [],
            'safety_info': [],
            'total_reward': 0.0,
            'episode_length': 0,
            'initial_liquid_mass': info.get('liquid_mass_remaining', 1.0),
            'initial_contaminant_mass': info.get('contaminant_mass_remaining', 0.5),
            'total_collisions': 0,
            'total_safety_violations': 0,
            'violation_duration': 0
        }
        
        # Video recording
        if save_video:
            video_path = self.output_dir / "videos" / f"episode_{episode_idx:03d}.mp4"
            self.video_recorder.start_recording(str(video_path))
        
        # Episode loop
        for step in range(1000):  # Max episode length
            # Get action from policy
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
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(projected_action)
            
            # Record data
            episode_data['actions'].append(projected_action.copy())
            episode_data['rewards'].append(reward)
            episode_data['observations'].append(obs.copy())
            episode_data['safety_info'].append(safety_info.copy())
            episode_data['total_reward'] += reward
            episode_data['episode_length'] += 1
            
            # Update safety metrics
            if safety_info.get('safety_level') in ['warning', 'critical']:
                episode_data['total_safety_violations'] += 1
                episode_data['violation_duration'] += 1
            
            if info.get('collisions', 0) > 0:
                episode_data['total_collisions'] += 1
            
            # Render if requested
            if render:
                self._render_frame(obs, action, projected_action, safety_info)
            
            # Record video frame
            if save_video:
                self._record_video_frame(obs, action, projected_action, safety_info)
            
            # Check termination
            if terminated or truncated:
                episode_data['final_liquid_mass'] = info.get('liquid_mass_remaining', 1.0)
                episode_data['final_contaminant_mass'] = info.get('contaminant_mass_remaining', 0.5)
                break
            
            obs = next_obs
        
        # Stop video recording
        if save_video:
            self.video_recorder.stop_recording()
        
        return episode_data
    
    def _render_frame(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray, 
        projected_action: np.ndarray,
        safety_info: Dict[str, Any]
    ) -> None:
        """Render current frame."""
        # Convert image to displayable format
        image = obs['image']
        if image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        
        # Resize for display
        image = cv2.resize(image, (400, 400))
        
        # Add text overlay
        cv2.putText(image, f"Action: {action[:3]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Projected: {projected_action[:3]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Safety: {safety_info.get('safety_level', 'unknown')}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display image
        cv2.imshow('Evaluation', image)
        cv2.waitKey(1)
    
    def _record_video_frame(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray, 
        projected_action: np.ndarray,
        safety_info: Dict[str, Any]
    ) -> None:
        """Record frame for video."""
        # Convert image to displayable format
        image = obs['image']
        if image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        
        # Resize for video
        image = cv2.resize(image, (640, 480))
        
        # Add text overlay
        cv2.putText(image, f"Action: {action[:3]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Projected: {projected_action[:3]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Safety: {safety_info.get('safety_level', 'unknown')}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Record frame
        self.video_recorder.record_frame(image)
    
    def _generate_plots(self) -> None:
        """Generate evaluation plots."""
        episode_breakdown = self.metrics_calculator.get_episode_breakdown()
        
        # Reward progression plot
        rewards = [ep['total_reward'] for ep in episode_breakdown]
        self.plot_generator.plot_reward_progression(rewards)
        
        # Success rate plot
        successes = [ep['success'] for ep in episode_breakdown]
        self.plot_generator.plot_success_rate(successes)
        
        # Safety metrics plot
        collisions = [ep['collisions'] for ep in episode_breakdown]
        safety_violations = [ep['safety_violations'] for ep in episode_breakdown]
        self.plot_generator.plot_safety_metrics(collisions, safety_violations)
        
        # Performance distribution plot
        liquid_reductions = [ep['liquid_reduction'] for ep in episode_breakdown]
        contaminant_reductions = [ep['contaminant_reduction'] for ep in episode_breakdown]
        self.plot_generator.plot_performance_distribution(liquid_reductions, contaminant_reductions)
    
    def _save_results(self, metrics: Dict[str, Any]) -> None:
        """Save evaluation results."""
        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        self.metrics_calculator.save_metrics(str(metrics_file))
        
        # Save episode breakdown
        episode_breakdown = self.metrics_calculator.get_episode_breakdown()
        breakdown_file = self.output_dir / "episode_breakdown.json"
        with open(breakdown_file, 'w') as f:
            json.dump(episode_breakdown, f, indent=2)
        
        # Save summary
        summary = {
            'checkpoint_path': self.checkpoint_path,
            'evaluation_config': self.config,
            'metrics': metrics,
            'num_episodes': len(episode_breakdown)
        }
        
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")


class AblationEvaluator:
    """
    Evaluator for ablation studies.
    """
    
    def __init__(self, base_config: Dict[str, Any], output_dir: str):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        self.results = {}
    
    def run_ablation_study(
        self, 
        variations: Dict[str, Dict[str, Any]], 
        num_episodes: int = 5
    ) -> Dict[str, Any]:
        """Run ablation study with different configurations."""
        logger.info(f"Running ablation study with {len(variations)} variations")
        
        for name, config in variations.items():
            logger.info(f"Evaluating variation: {name}")
            
            # Create evaluator with modified config
            modified_config = self.base_config.copy()
            modified_config.update(config)
            
            # Create temporary checkpoint (would need to be trained separately)
            # For now, use the same checkpoint for all variations
            checkpoint_path = "data/checkpoints/best_model.pth"
            
            # Evaluate
            evaluator = ModelEvaluator(
                checkpoint_path=checkpoint_path,
                output_dir=str(self.output_dir / name),
                config=modified_config
            )
            
            metrics = evaluator.evaluate(num_episodes=num_episodes, render=False, save_videos=False)
            self.results[name] = metrics
        
        # Generate comparison
        self._generate_comparison()
        
        return self.results
    
    def _generate_comparison(self) -> None:
        """Generate comparison plots and tables."""
        from .metrics import ComparativeMetrics
        
        # Create comparison
        comparison = ComparativeMetrics()
        for name, metrics in self.results.items():
            comparison.add_result(name, metrics)
        
        # Save comparison
        comparison_file = self.output_dir / "ablation_comparison.json"
        comparison.save_comparison(str(comparison_file))
        
        # Generate comparison plots
        plot_generator = PlotGenerator(str(self.output_dir))
        plot_generator.plot_ablation_comparison(self.results)
        
        logger.info(f"Ablation study results saved to {self.output_dir}")


def evaluate_model(
    checkpoint_path: str,
    output_dir: str,
    num_episodes: int = 10,
    render: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate a trained model."""
    evaluator = ModelEvaluator(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        config=config
    )
    
    return evaluator.evaluate(
        num_episodes=num_episodes,
        render=render,
        save_videos=True
    )
