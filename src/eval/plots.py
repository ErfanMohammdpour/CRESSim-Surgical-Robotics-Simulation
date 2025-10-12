"""
Plot generation for evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PlotGenerator:
    """
    Generate various plots for evaluation results.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_reward_progression(self, rewards: List[float], title: str = "Reward Progression") -> None:
        """Plot reward progression over episodes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = range(1, len(rewards) + 1)
        ax.plot(episodes, rewards, 'b-', linewidth=2, alpha=0.7)
        ax.scatter(episodes, rewards, c='blue', s=30, alpha=0.8)
        
        # Add trend line
        if len(rewards) > 1:
            z = np.polyfit(episodes, rewards, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=2)
        
        # Add statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        ax.axhline(y=mean_reward, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_reward:.2f}')
        ax.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2, color='green')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "reward_progression.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_success_rate(self, successes: List[bool], title: str = "Success Rate") -> None:
        """Plot success rate over episodes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Episode-by-episode success
        episodes = range(1, len(successes) + 1)
        ax1.plot(episodes, successes, 'go-', linewidth=2, markersize=8)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success (1) / Failure (0)')
        ax1.set_title('Episode Success')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative success rate
        cumulative_success_rate = np.cumsum(successes) / np.arange(1, len(successes) + 1)
        ax2.plot(episodes, cumulative_success_rate, 'b-', linewidth=2)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Success Rate')
        ax2.set_title('Cumulative Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "success_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_safety_metrics(
        self, 
        collisions: List[int], 
        safety_violations: List[int],
        title: str = "Safety Metrics"
    ) -> None:
        """Plot safety metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(1, len(collisions) + 1)
        
        # Collisions per episode
        ax1.bar(episodes, collisions, color='red', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Collisions')
        ax1.set_title('Collisions per Episode')
        ax1.grid(True, alpha=0.3)
        
        # Safety violations per episode
        ax2.bar(episodes, safety_violations, color='orange', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Safety Violations')
        ax2.set_title('Safety Violations per Episode')
        ax2.grid(True, alpha=0.3)
        
        # Cumulative collisions
        cumulative_collisions = np.cumsum(collisions)
        ax3.plot(episodes, cumulative_collisions, 'r-', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Cumulative Collisions')
        ax3.set_title('Cumulative Collisions')
        ax3.grid(True, alpha=0.3)
        
        # Safety score (inverse of violations)
        safety_scores = [1.0 / (1.0 + violations) for violations in safety_violations]
        ax4.plot(episodes, safety_scores, 'g-', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Safety Score')
        ax4.set_title('Safety Score per Episode')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "safety_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_distribution(
        self, 
        liquid_reductions: List[float], 
        contaminant_reductions: List[float],
        title: str = "Performance Distribution"
    ) -> None:
        """Plot performance distribution."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Liquid reduction histogram
        ax1.hist(liquid_reductions, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(liquid_reductions), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(liquid_reductions):.3f}')
        ax1.set_xlabel('Liquid Reduction')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Liquid Reduction Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Contaminant reduction histogram
        ax2.hist(contaminant_reductions, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(contaminant_reductions), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(contaminant_reductions):.3f}')
        ax2.set_xlabel('Contaminant Reduction')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Contaminant Reduction Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Scatter plot
        ax3.scatter(liquid_reductions, contaminant_reductions, alpha=0.7, s=50)
        ax3.set_xlabel('Liquid Reduction')
        ax3.set_ylabel('Contaminant Reduction')
        ax3.set_title('Liquid vs Contaminant Reduction')
        ax3.grid(True, alpha=0.3)
        
        # Box plots
        data = [liquid_reductions, contaminant_reductions]
        labels = ['Liquid Reduction', 'Contaminant Reduction']
        ax4.boxplot(data, labels=labels)
        ax4.set_ylabel('Reduction')
        ax4.set_title('Performance Box Plots')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ablation_comparison(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Plot ablation study comparison."""
        # Extract metrics for comparison
        metrics_to_plot = ['success_rate', 'mean_reward', 'mean_liquid_reduction', 'mean_contaminant_reduction']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            methods = list(results.keys())
            values = [results[method].get(metric, 0) for method in methods]
            
            bars = ax.bar(methods, values, alpha=0.7)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "ablation_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curves(
        self, 
        train_rewards: List[float], 
        eval_rewards: List[float],
        title: str = "Learning Curves"
    ) -> None:
        """Plot learning curves."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training rewards
        train_episodes = range(1, len(train_rewards) + 1)
        ax.plot(train_episodes, train_rewards, 'b-', label='Training', alpha=0.7)
        
        # Plot evaluation rewards
        eval_episodes = range(1, len(eval_rewards) + 1)
        ax.plot(eval_episodes, eval_rewards, 'r-', label='Evaluation', linewidth=2)
        
        # Add smoothed curves
        if len(train_rewards) > 10:
            train_smooth = self._smooth_curve(train_rewards, window=10)
            ax.plot(train_episodes, train_smooth, 'b--', alpha=0.8, linewidth=2)
        
        if len(eval_rewards) > 5:
            eval_smooth = self._smooth_curve(eval_rewards, window=5)
            ax.plot(eval_episodes, eval_smooth, 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_action_analysis(self, actions: List[np.ndarray], title: str = "Action Analysis") -> None:
        """Plot action analysis."""
        if not actions:
            return
        
        actions_array = np.array(actions)
        action_names = ['dx', 'dy', 'dz', 'dyaw', 'suction']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Plot each action dimension
        for i in range(min(5, actions_array.shape[1])):
            ax = axes[i]
            
            # Time series
            ax.plot(actions_array[:, i], alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel(f'{action_names[i]}')
            ax.set_title(f'{action_names[i]} over Time')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(actions_array[:, i])
            std_val = np.std(actions_array[:, i])
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7)
            ax.fill_between(range(len(actions_array)), 
                           mean_val - std_val, mean_val + std_val, 
                           alpha=0.2, color='red')
        
        # Action magnitude
        ax = axes[5]
        action_magnitudes = np.linalg.norm(actions_array, axis=1)
        ax.plot(action_magnitudes, alpha=0.7, color='purple')
        ax.set_xlabel('Step')
        ax.set_ylabel('Action Magnitude')
        ax.set_title('Action Magnitude over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "action_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_safety_shield_analysis(
        self, 
        safety_info: List[Dict[str, Any]], 
        title: str = "Safety Shield Analysis"
    ) -> None:
        """Plot safety shield analysis."""
        if not safety_info:
            return
        
        # Extract data
        safety_levels = [info.get('safety_level', 'unknown') for info in safety_info]
        distances = [info.get('distance', 0) for info in safety_info]
        action_scales = [info.get('action_scale', 1.0) for info in safety_info]
        violation_counts = [info.get('violation_count', 0) for info in safety_info]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Safety level distribution
        safety_counts = pd.Series(safety_levels).value_counts()
        ax1.pie(safety_counts.values, labels=safety_counts.index, autopct='%1.1f%%')
        ax1.set_title('Safety Level Distribution')
        
        # Distance over time
        ax2.plot(distances, alpha=0.7, color='blue')
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Safe threshold')
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Distance')
        ax2.set_title('Distance over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Action scale over time
        ax3.plot(action_scales, alpha=0.7, color='green')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Action Scale')
        ax3.set_title('Action Scale over Time')
        ax3.grid(True, alpha=0.3)
        
        # Violation count
        ax4.plot(violation_counts, alpha=0.7, color='red')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Cumulative Violations')
        ax4.set_title('Cumulative Violations')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "safety_shield_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _smooth_curve(self, data: List[float], window: int = 10) -> List[float]:
        """Smooth curve using moving average."""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        
        return smoothed
    
    def create_summary_dashboard(self, metrics: Dict[str, Any]) -> None:
        """Create a summary dashboard with key metrics."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Key metrics text
        ax_text = fig.add_subplot(gs[0, :2])
        ax_text.axis('off')
        
        key_metrics = [
            f"Success Rate: {metrics.get('success_rate', 0):.3f}",
            f"Mean Reward: {metrics.get('mean_reward', 0):.2f} ± {metrics.get('std_reward', 0):.2f}",
            f"Mean Liquid Reduction: {metrics.get('mean_liquid_reduction', 0):.3f}",
            f"Mean Contaminant Reduction: {metrics.get('mean_contaminant_reduction', 0):.3f}",
            f"Mean Collisions: {metrics.get('mean_collisions_per_episode', 0):.2f}",
            f"Safety Violations: {metrics.get('mean_safety_violations_per_episode', 0):.2f}"
        ]
        
        ax_text.text(0.1, 0.9, "Key Metrics", fontsize=16, fontweight='bold', transform=ax_text.transAxes)
        for i, metric in enumerate(key_metrics):
            ax_text.text(0.1, 0.8 - i*0.1, metric, fontsize=12, transform=ax_text.transAxes)
        
        # Performance distribution
        ax_perf = fig.add_subplot(gs[0, 2:])
        performance_data = [
            metrics.get('mean_liquid_reduction', 0),
            metrics.get('mean_contaminant_reduction', 0),
            metrics.get('success_rate', 0)
        ]
        performance_labels = ['Liquid Reduction', 'Contaminant Reduction', 'Success Rate']
        bars = ax_perf.bar(performance_labels, performance_data, alpha=0.7)
        ax_perf.set_title('Performance Metrics')
        ax_perf.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, performance_data):
            height = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Safety metrics
        ax_safety = fig.add_subplot(gs[1, :2])
        safety_data = [
            metrics.get('mean_collisions_per_episode', 0),
            metrics.get('mean_safety_violations_per_episode', 0),
            metrics.get('episodes_with_violations_rate', 0)
        ]
        safety_labels = ['Collisions', 'Safety Violations', 'Violation Rate']
        bars = ax_safety.bar(safety_labels, safety_data, alpha=0.7, color=['red', 'orange', 'yellow'])
        ax_safety.set_title('Safety Metrics')
        ax_safety.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, safety_data):
            height = bar.get_height()
            ax_safety.text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # Efficiency metrics
        ax_eff = fig.add_subplot(gs[1, 2:])
        eff_data = [
            metrics.get('mean_path_length', 0),
            metrics.get('mean_action_smoothness', 0),
            metrics.get('mean_energy_consumption', 0)
        ]
        eff_labels = ['Path Length', 'Action Smoothness', 'Energy Consumption']
        bars = ax_eff.bar(eff_labels, eff_data, alpha=0.7, color=['blue', 'green', 'purple'])
        ax_eff.set_title('Efficiency Metrics')
        ax_eff.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, eff_data):
            height = bar.get_height()
            ax_eff.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Overall performance score
        ax_score = fig.add_subplot(gs[2, :])
        ax_score.axis('off')
        
        # Calculate overall score
        success_score = metrics.get('success_rate', 0) * 100
        safety_score = max(0, 100 - metrics.get('mean_safety_violations_per_episode', 0) * 20)
        efficiency_score = min(100, metrics.get('mean_action_smoothness', 0) * 100)
        
        overall_score = (success_score + safety_score + efficiency_score) / 3
        
        ax_score.text(0.5, 0.7, f"Overall Performance Score", 
                     fontsize=20, fontweight='bold', ha='center', transform=ax_score.transAxes)
        ax_score.text(0.5, 0.5, f"{overall_score:.1f}/100", 
                     fontsize=48, fontweight='bold', ha='center', transform=ax_score.transAxes,
                     color='green' if overall_score >= 70 else 'orange' if overall_score >= 50 else 'red')
        
        ax_score.text(0.5, 0.3, f"Success: {success_score:.1f} | Safety: {safety_score:.1f} | Efficiency: {efficiency_score:.1f}", 
                     fontsize=14, ha='center', transform=ax_score.transAxes)
        
        plt.suptitle('Evaluation Summary Dashboard', fontsize=20, fontweight='bold')
        plt.savefig(self.output_dir / "summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
