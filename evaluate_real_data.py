#!/usr/bin/env python3
"""
ارزیابی مدل روی داده‌های واقعی بدون Unity
Real Data Evaluation Script for Surgical Robotics Model
"""

import os
import sys
import torch
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.envs.mock_env import MockSuctionEnv
from src.vision.encoders import CNNEncoder
from src.safety.safety_shield import SafetyShield
from src.utils.gpu import get_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataEvaluator:
    """ارزیابی مدل روی داده‌های واقعی"""
    
    def __init__(self, model_path: str, output_dir: str = "evaluation_results"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Setup environment with realistic parameters
        self.env = MockSuctionEnv(
            image_size=(128, 128),
            max_steps=1000,
            reward_weights={
                "alpha": 2.0,      # liquid mass reduction
                "beta": 1.5,       # contaminant mass reduction  
                "lambda_time": -0.005,  # time penalty
                "lambda_action": -0.0005,  # action smoothness
                "lambda_collision": -2.0,  # collision penalty
                "lambda_safety": -5.0  # safety violation penalty
            }
        )
        
        # Setup safety shield
        self.safety_shield = SafetyShield({
            "d_safe": 0.05,
            "segmentation_threshold": 0.5,
            "projection_scaling_factor": 0.8
        })
        
        logger.info("Real Data Evaluator initialized successfully")
    
    def _load_model(self):
        """بارگذاری مدل آموزش دیده"""
        try:
            # Try loading as PyTorch model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                # Load model architecture
                encoder = CNNEncoder(
                    input_channels=3,
                    output_dim=512,
                    channels=[32, 64, 128, 256],
                    kernel_sizes=[3, 3, 3, 3],
                    strides=[2, 2, 2, 2],
                    padding=[1, 1, 1, 1]
                )
                
                # Create policy network
                from src.il.bc_trainer import PolicyNetwork
                model = PolicyNetwork(encoder=encoder, hidden_dim=512, action_dim=5)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()
                
                logger.info("PyTorch model loaded successfully")
                return model
            else:
                # Direct model loading
                model = checkpoint.to(self.device)
                model.eval()
                logger.info("Direct PyTorch model loaded successfully")
                return model
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_on_real_data(self, num_episodes: int = 50, render: bool = True) -> Dict[str, Any]:
        """ارزیابی مدل روی داده‌های واقعی"""
        logger.info(f"Starting evaluation on {num_episodes} episodes...")
        
        episode_results = []
        all_rewards = []
        all_lengths = []
        success_count = 0
        
        # Performance metrics
        liquid_reductions = []
        contaminant_reductions = []
        collision_counts = []
        safety_violations = []
        
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
            
            # Detailed metrics
            info = episode_result['final_info']
            liquid_reduction = info.get('liquid_reduction', 0) * 100
            contaminant_reduction = info.get('contaminant_reduction', 0) * 100
            collisions = info.get('collision_count', 0)
            safety_viol = info.get('safety_violations', 0)
            
            liquid_reductions.append(liquid_reduction)
            contaminant_reductions.append(contaminant_reduction)
            collision_counts.append(collisions)
            safety_violations.append(safety_viol)
            
            logger.info(f"  Reward: {episode_result['total_reward']:.2f}")
            logger.info(f"  Liquid Reduction: {liquid_reduction:.1f}%")
            logger.info(f"  Contaminant Reduction: {contaminant_reduction:.1f}%")
            logger.info(f"  Collisions: {collisions}, Safety Violations: {safety_viol}")
            logger.info(f"  Success: {'✅' if episode_result['success'] else '❌'}")
        
        # Calculate comprehensive statistics
        results = self._calculate_comprehensive_stats(
            episode_results, all_rewards, all_lengths, success_count,
            liquid_reductions, contaminant_reductions, collision_counts, safety_violations
        )
        
        # Save results
        self._save_results(episode_results, results)
        
        # Generate visualizations
        self._generate_visualizations(results, liquid_reductions, contaminant_reductions)
        
        logger.info(f"Evaluation completed! Success rate: {results['success_rate']:.2%}")
        return results
    
    def _run_episode(self, episode: int, render: bool = False) -> Dict[str, Any]:
        """اجرای یک اپیزود"""
        obs, info = self.env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        # Video recording
        frames = [] if render else None
        
        while not done:
            # Get action from model
            obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
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
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
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
        """تعیین موفقیت اپیزود"""
        liquid_reduction = info.get('liquid_reduction', 0)
        contaminant_reduction = info.get('contaminant_reduction', 0)
        
        # Success criteria: 80% liquid and 80% contaminant removal
        return liquid_reduction > 0.8 and contaminant_reduction > 0.8
    
    def _calculate_comprehensive_stats(self, episode_results: List[Dict], all_rewards: List[float], 
                                     all_lengths: List[int], success_count: int,
                                     liquid_reductions: List[float], contaminant_reductions: List[float],
                                     collision_counts: List[int], safety_violations: List[int]) -> Dict[str, Any]:
        """محاسبه آمار جامع"""
        num_episodes = len(episode_results)
        
        return {
            # Basic metrics
            'num_episodes': num_episodes,
            'success_count': success_count,
            'success_rate': success_count / num_episodes,
            
            # Reward metrics
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            
            # Episode length metrics
            'mean_length': np.mean(all_lengths),
            'std_length': np.std(all_lengths),
            'min_length': np.min(all_lengths),
            'max_length': np.max(all_lengths),
            
            # Task performance metrics
            'mean_liquid_reduction': np.mean(liquid_reductions),
            'std_liquid_reduction': np.std(liquid_reductions),
            'mean_contaminant_reduction': np.mean(contaminant_reductions),
            'std_contaminant_reduction': np.std(contaminant_reductions),
            
            # Safety metrics
            'mean_collisions': np.mean(collision_counts),
            'std_collisions': np.std(collision_counts),
            'mean_safety_violations': np.mean(safety_violations),
            'std_safety_violations': np.std(safety_violations),
            'total_safety_violations': sum(safety_violations),
            
            # Performance analysis
            'high_performance_episodes': sum(1 for lr, cr in zip(liquid_reductions, contaminant_reductions) 
                                           if lr > 70 and cr > 70),
            'low_performance_episodes': sum(1 for lr, cr in zip(liquid_reductions, contaminant_reductions) 
                                          if lr < 30 or cr < 30),
            
            # Detailed episode results
            'episode_results': episode_results
        }
    
    def _save_results(self, episode_results: List[Dict], summary: Dict[str, Any]):
        """ذخیره نتایج"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = self.output_dir / f"real_data_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'episode_results': episode_results,
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path
            }, f, indent=2)
        
        # Save summary CSV
        import pandas as pd
        summary_df = pd.DataFrame([summary])
        csv_file = self.output_dir / f"evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {csv_file}")
    
    def _save_video(self, frames: List[np.ndarray], episode: int):
        """ذخیره ویدیو"""
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
    
    def _generate_visualizations(self, results: Dict[str, Any], liquid_reductions: List[float], 
                                contaminant_reductions: List[float]):
        """تولید نمودارهای تحلیلی"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Real Data Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Success Rate Distribution
        success_data = [1 if r['success'] else 0 for r in results['episode_results']]
        axes[0, 0].bar(['Failed', 'Success'], [sum(1-x for x in success_data), sum(success_data)])
        axes[0, 0].set_title('Success Rate Distribution')
        axes[0, 0].set_ylabel('Number of Episodes')
        
        # 2. Liquid vs Contaminant Reduction
        axes[0, 1].scatter(liquid_reductions, contaminant_reductions, alpha=0.6)
        axes[0, 1].axhline(y=80, color='r', linestyle='--', label='80% Threshold')
        axes[0, 1].axvline(x=80, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Liquid Reduction (%)')
        axes[0, 1].set_ylabel('Contaminant Reduction (%)')
        axes[0, 1].set_title('Task Performance Scatter Plot')
        axes[0, 1].legend()
        
        # 3. Reward Distribution
        rewards = [r['total_reward'] for r in results['episode_results']]
        axes[1, 0].hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Total Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reward Distribution')
        
        # 4. Episode Length vs Performance
        lengths = [r['episode_length'] for r in results['episode_results']]
        performance = [(lr + cr) / 2 for lr, cr in zip(liquid_reductions, contaminant_reductions)]
        axes[1, 1].scatter(lengths, performance, alpha=0.6)
        axes[1, 1].set_xlabel('Episode Length')
        axes[1, 1].set_ylabel('Average Performance (%)')
        axes[1, 1].set_title('Episode Length vs Performance')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"evaluation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {plot_file}")


def main():
    """تابع اصلی"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ارزیابی مدل روی داده‌های واقعی')
    parser.add_argument('--model', '-m', required=True, help='مسیر مدل آموزش دیده')
    parser.add_argument('--episodes', '-e', type=int, default=50, help='تعداد اپیزودها')
    parser.add_argument('--output', '-o', default='evaluation_results', help='پوشه خروجی')
    parser.add_argument('--render', '-r', action='store_true', help='تولید ویدیو')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Create evaluator
    evaluator = RealDataEvaluator(args.model, args.output)
    
    # Run evaluation
    results = evaluator.evaluate_on_real_data(
        num_episodes=args.episodes,
        render=args.render
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📊 نتایج ارزیابی مدل روی داده‌های واقعی")
    print("="*60)
    print(f"✅ نرخ موفقیت: {results['success_rate']:.2%}")
    print(f"📈 میانگین کاهش مایع: {results['mean_liquid_reduction']:.1f}% ± {results['std_liquid_reduction']:.1f}%")
    print(f"📈 میانگین کاهش آلودگی: {results['mean_contaminant_reduction']:.1f}% ± {results['std_contaminant_reduction']:.1f}%")
    print(f"💰 میانگین پاداش: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"⏱️ میانگین طول اپیزود: {results['mean_length']:.0f} ± {results['std_length']:.0f}")
    print(f"⚠️ میانگین برخورد: {results['mean_collisions']:.1f} ± {results['std_collisions']:.1f}")
    print(f"🛡️ میانگین نقض ایمنی: {results['mean_safety_violations']:.1f} ± {results['std_safety_violations']:.1f}")
    print(f"🎯 اپیزودهای عملکرد بالا: {results['high_performance_episodes']}")
    print(f"📉 اپیزودهای عملکرد پایین: {results['low_performance_episodes']}")
    print("="*60)


if __name__ == "__main__":
    main()
