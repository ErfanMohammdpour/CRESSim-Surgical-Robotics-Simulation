"""
Plotting utilities for evaluation and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os
from pathlib import Path


class EvaluationPlotter:
    """Creates plots for evaluation results."""
    
    def __init__(self, output_dir: str = "data/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, metrics: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
        """Plot training curves for different metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(metrics.get('episode_rewards', []))
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(metrics.get('episode_lengths', []))
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Safety violations
        axes[1, 0].plot(metrics.get('safety_violations', []))
        axes[1, 0].set_title('Safety Violations')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Violations')
        axes[1, 0].grid(True)
        
        # Success rate
        if 'success_rate' in metrics:
            axes[1, 1].plot(metrics['success_rate'])
            axes[1, 1].set_title('Success Rate')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_evaluation_metrics(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot evaluation metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evaluation Results', fontsize=16)
        
        # Success rate by episode
        episodes = range(len(results.get('episode_rewards', [])))
        success_rates = results.get('success_rates', [])
        
        if success_rates:
            axes[0, 0].bar(episodes, success_rates)
            axes[0, 0].set_title('Success Rate by Episode')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].set_ylim(0, 1)
        
        # Episode rewards distribution
        episode_rewards = results.get('episode_rewards', [])
        if episode_rewards:
            axes[0, 1].hist(episode_rewards, bins=20, alpha=0.7)
            axes[0, 1].set_title('Episode Rewards Distribution')
            axes[0, 1].set_xlabel('Reward')
            axes[0, 1].set_ylabel('Frequency')
        
        # Safety violations by episode
        safety_violations = results.get('safety_violations', [])
        if safety_violations:
            axes[1, 0].bar(episodes, safety_violations)
            axes[1, 0].set_title('Safety Violations by Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Violations')
        
        # Task completion time
        completion_times = results.get('completion_times', [])
        if completion_times:
            axes[1, 1].hist(completion_times, bins=20, alpha=0.7)
            axes[1, 1].set_title('Task Completion Time')
            axes[1, 1].set_xlabel('Time (steps)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_ablation_study(self, ablation_results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
        """Plot ablation study results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Ablation Study Results', fontsize=16)
        
        methods = list(ablation_results.keys())
        
        # Success rate comparison
        success_rates = [ablation_results[method].get('success_rate', 0) for method in methods]
        axes[0, 0].bar(methods, success_rates)
        axes[0, 0].set_title('Success Rate Comparison')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Average reward comparison
        avg_rewards = [ablation_results[method].get('avg_reward', 0) for method in methods]
        axes[0, 1].bar(methods, avg_rewards)
        axes[0, 1].set_title('Average Reward Comparison')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Safety violation rate comparison
        violation_rates = [ablation_results[method].get('safety_violation_rate', 0) for method in methods]
        axes[1, 0].bar(methods, violation_rates)
        axes[1, 0].set_title('Safety Violation Rate Comparison')
        axes[1, 0].set_ylabel('Violation Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Episode length comparison
        avg_lengths = [ablation_results[method].get('avg_episode_length', 0) for method in methods]
        axes[1, 1].bar(methods, avg_lengths)
        axes[1, 1].set_title('Average Episode Length Comparison')
        axes[1, 1].set_ylabel('Episode Length')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_safety_analysis(self, safety_data: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot safety analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Safety Analysis', fontsize=16)
        
        # Safety violation timeline
        violation_timeline = safety_data.get('violation_timeline', [])
        if violation_timeline:
            axes[0, 0].plot(violation_timeline)
            axes[0, 0].set_title('Safety Violations Over Time')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Violation (0/1)')
            axes[0, 0].grid(True)
        
        # Distance to obstacles
        distances = safety_data.get('distances', [])
        if distances:
            axes[0, 1].hist(distances, bins=30, alpha=0.7)
            axes[0, 1].set_title('Distance to Obstacles Distribution')
            axes[0, 1].set_xlabel('Distance (m)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Safety mask visualization
        safety_masks = safety_data.get('safety_masks', [])
        if safety_masks:
            # Show average safety mask
            avg_mask = np.mean(safety_masks, axis=0)
            im = axes[1, 0].imshow(avg_mask, cmap='hot')
            axes[1, 0].set_title('Average Safety Mask')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Y')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Safety level distribution
        safety_levels = safety_data.get('safety_levels', [])
        if safety_levels:
            level_counts = np.bincount(safety_levels)
            axes[1, 1].bar(range(len(level_counts)), level_counts)
            axes[1, 1].set_title('Safety Level Distribution')
            axes[1, 1].set_xlabel('Safety Level')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xticks(range(len(level_counts)))
            axes[1, 1].set_xticklabels(['Safe', 'Warning', 'Critical', 'Emergency'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'safety_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_learning_curves(self, learning_data: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
        """Plot learning curves with moving averages."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Curves', fontsize=16)
        
        # Episode rewards with moving average
        episode_rewards = learning_data.get('episode_rewards', [])
        if episode_rewards:
            window = min(50, len(episode_rewards) // 10)
            moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
            
            axes[0, 0].plot(episode_rewards, alpha=0.3, label='Raw')
            axes[0, 0].plot(moving_avg, label=f'Moving Avg ({window})')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Loss curves
        policy_loss = learning_data.get('policy_loss', [])
        value_loss = learning_data.get('value_loss', [])
        
        if policy_loss:
            axes[0, 1].plot(policy_loss, label='Policy Loss')
        if value_loss:
            axes[0, 1].plot(value_loss, label='Value Loss')
        
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Entropy
        entropy = learning_data.get('entropy', [])
        if entropy:
            axes[1, 0].plot(entropy)
            axes[1, 0].set_title('Policy Entropy')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True)
        
        # KL divergence
        kl_div = learning_data.get('kl_divergence', [])
        if kl_div:
            axes[1, 1].plot(kl_div)
            axes[1, 1].set_title('KL Divergence')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('KL Divergence')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_action_distribution(self, actions: np.ndarray, save_path: Optional[str] = None) -> None:
        """Plot action distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Action Distribution Analysis', fontsize=16)
        
        # Action dimensions
        action_names = ['Δx', 'Δy', 'Δz', 'Δyaw']
        
        for i in range(min(4, actions.shape[1])):
            row, col = i // 2, i % 2
            
            axes[row, col].hist(actions[:, i], bins=50, alpha=0.7)
            axes[row, col].set_title(f'{action_names[i]} Distribution')
            axes[row, col].set_xlabel('Action Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_reward_components(self, reward_data: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
        """Plot individual reward components."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reward Components Analysis', fontsize=16)
        
        components = ['liquid_removal', 'contaminant_removal', 'time_penalty', 'action_penalty']
        
        for i, component in enumerate(components):
            if component in reward_data:
                row, col = i // 2, i % 2
                
                axes[row, col].plot(reward_data[component])
                axes[row, col].set_title(f'{component.replace("_", " ").title()}')
                axes[row, col].set_xlabel('Step')
                axes[row, col].set_ylabel('Reward')
                axes[row, col].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'reward_components.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_summary_plot(self, all_results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Create a comprehensive summary plot."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Training curves
        ax1 = fig.add_subplot(gs[0, :2])
        episode_rewards = all_results.get('episode_rewards', [])
        if episode_rewards:
            ax1.plot(episode_rewards, alpha=0.7)
            ax1.set_title('Training Progress - Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
        
        # Evaluation metrics
        ax2 = fig.add_subplot(gs[0, 2:])
        eval_rewards = all_results.get('eval_episode_rewards', [])
        if eval_rewards:
            ax2.bar(range(len(eval_rewards)), eval_rewards)
            ax2.set_title('Evaluation Episode Rewards')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
        
        # Safety analysis
        ax3 = fig.add_subplot(gs[1, :2])
        safety_violations = all_results.get('safety_violations', [])
        if safety_violations:
            ax3.plot(safety_violations)
            ax3.set_title('Safety Violations Over Time')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Violations')
            ax3.grid(True)
        
        # Action distribution
        ax4 = fig.add_subplot(gs[1, 2:])
        actions = all_results.get('actions', [])
        if len(actions) > 0:
            actions = np.array(actions)
            ax4.hist(actions[:, 0], bins=30, alpha=0.7, label='Δx')
            ax4.hist(actions[:, 1], bins=30, alpha=0.7, label='Δy')
            ax4.set_title('Action Distribution')
            ax4.set_xlabel('Action Value')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        # Performance metrics
        ax5 = fig.add_subplot(gs[2, :])
        metrics = all_results.get('summary_metrics', {})
        if metrics:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            ax5.bar(metric_names, metric_values)
            ax5.set_title('Summary Metrics')
            ax5.set_ylabel('Value')
            ax5.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Comprehensive Evaluation Summary', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'summary_plot.png', dpi=300, bbox_inches='tight')
        
        plt.close()


def create_plots_from_results(results: Dict[str, Any], output_dir: str = "data/plots") -> None:
    """Create all plots from evaluation results."""
    plotter = EvaluationPlotter(output_dir)
    
    # Training curves
    if 'training_metrics' in results:
        plotter.plot_training_curves(results['training_metrics'])
    
    # Evaluation metrics
    plotter.plot_evaluation_metrics(results)
    
    # Safety analysis
    if 'safety_data' in results:
        plotter.plot_safety_analysis(results['safety_data'])
    
    # Learning curves
    if 'learning_data' in results:
        plotter.plot_learning_curves(results['learning_data'])
    
    # Action distribution
    if 'actions' in results:
        plotter.plot_action_distribution(np.array(results['actions']))
    
    # Reward components
    if 'reward_components' in results:
        plotter.plot_reward_components(results['reward_components'])
    
    # Summary plot
    plotter.create_summary_plot(results)
    
    print(f"📊 All plots saved to {output_dir}")
