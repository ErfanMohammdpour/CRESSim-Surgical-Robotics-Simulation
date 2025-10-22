"""
Evaluation utilities and report generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationReport:
    """Generate evaluation reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"EvaluationReport initialized: {output_dir}")
    
    def generate_report(
        self,
        results: Dict[str, Any],
        model_name: str = "Model"
    ) -> Path:
        """Generate comprehensive evaluation report."""
        logger.info(f"Generating evaluation report for {model_name}")
        
        # Create plots
        self._create_metrics_plots(results)
        self._create_episode_plots(results)
        self._create_safety_plots(results)
        
        # Generate markdown report
        report_file = self._generate_markdown_report(results, model_name)
        
        logger.info(f"Evaluation report generated: {report_file}")
        return report_file
    
    def _create_metrics_plots(self, results: Dict[str, Any]):
        """Create metrics visualization plots."""
        episode_metrics = results['episode_metrics']
        aggregate_metrics = results['aggregate_metrics']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Evaluation Metrics', fontsize=16)
        
        # Episode length distribution
        episode_lengths = [m['episode_length'] for m in episode_metrics]
        axes[0, 0].hist(episode_lengths, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].set_xlabel('Episode Length')
        axes[0, 0].set_ylabel('Frequency')
        
        # Total reward distribution
        total_rewards = [m['total_reward'] for m in episode_metrics]
        axes[0, 1].hist(total_rewards, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Total Reward Distribution')
        axes[0, 1].set_xlabel('Total Reward')
        axes[0, 1].set_ylabel('Frequency')
        
        # Liquid reduction distribution
        liquid_reductions = [m['liquid_reduction'] for m in episode_metrics]
        axes[0, 2].hist(liquid_reductions, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Liquid Reduction Distribution')
        axes[0, 2].set_xlabel('Liquid Reduction')
        axes[0, 2].set_ylabel('Frequency')
        
        # Contaminant reduction distribution
        contaminant_reductions = [m['contaminant_reduction'] for m in episode_metrics]
        axes[1, 0].hist(contaminant_reductions, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Contaminant Reduction Distribution')
        axes[1, 0].set_xlabel('Contaminant Reduction')
        axes[1, 0].set_ylabel('Frequency')
        
        # Safety violations distribution
        safety_violations = [m['safety_violations'] for m in episode_metrics]
        axes[1, 1].hist(safety_violations, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Safety Violations Distribution')
        axes[1, 1].set_xlabel('Safety Violations')
        axes[1, 1].set_ylabel('Frequency')
        
        # Success rate pie chart
        successes = [m['success'] for m in episode_metrics]
        success_count = sum(successes)
        failure_count = len(successes) - success_count
        
        axes[1, 2].pie([success_count, failure_count], 
                      labels=['Success', 'Failure'],
                      autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 2].set_title('Success Rate')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_episode_plots(self, results: Dict[str, Any]):
        """Create episode-level plots."""
        episode_data = results['episode_data']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Episode Analysis', fontsize=16)
        
        # Reward over time for each episode
        for i, episode in enumerate(episode_data[:5]):  # Show first 5 episodes
            rewards = episode['rewards']
            axes[0, 0].plot(rewards, label=f'Episode {i+1}', alpha=0.7)
        axes[0, 0].set_title('Reward Over Time (First 5 Episodes)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        
        # Action variance over episodes
        action_variances = []
        for episode in episode_data:
            actions = episode['actions']
            action_var = np.var(actions, axis=0).mean()
            action_variances.append(action_var)
        
        axes[0, 1].plot(action_variances, marker='o')
        axes[0, 1].set_title('Action Variance Over Episodes')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Action Variance')
        
        # Liquid reduction over episodes
        liquid_reductions = [episode['infos'][-1].get('liquid_reduction', 0) for episode in episode_data]
        axes[1, 0].plot(liquid_reductions, marker='o', color='blue')
        axes[1, 0].set_title('Liquid Reduction Over Episodes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Liquid Reduction')
        
        # Contaminant reduction over episodes
        contaminant_reductions = [episode['infos'][-1].get('contaminant_reduction', 0) for episode in episode_data]
        axes[1, 1].plot(contaminant_reductions, marker='o', color='orange')
        axes[1, 1].set_title('Contaminant Reduction Over Episodes')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Contaminant Reduction')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'episode_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_safety_plots(self, results: Dict[str, Any]):
        """Create safety-related plots."""
        episode_data = results['episode_data']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Safety Analysis', fontsize=16)
        
        # Safety violations over episodes
        safety_violations = [episode['infos'][-1].get('safety_violations', 0) for episode in episode_data]
        axes[0, 0].bar(range(len(safety_violations)), safety_violations, alpha=0.7)
        axes[0, 0].set_title('Safety Violations Per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Safety Violations')
        
        # Collision count over episodes
        collision_counts = [episode['infos'][-1].get('collision_count', 0) for episode in episode_data]
        axes[0, 1].bar(range(len(collision_counts)), collision_counts, alpha=0.7, color='red')
        axes[0, 1].set_title('Collision Count Per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Collision Count')
        
        # Safety violation rate over time
        violation_rates = []
        for episode in episode_data:
            violations = sum(1 for info in episode['infos'] if info.get('safety_violations', 0) > 0)
            violation_rate = violations / episode['length']
            violation_rates.append(violation_rate)
        
        axes[1, 0].plot(violation_rates, marker='o', color='red')
        axes[1, 0].set_title('Safety Violation Rate Over Episodes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Violation Rate')
        
        # Safety compliance histogram
        safety_compliance = [1 - vr for vr in violation_rates]
        axes[1, 1].hist(safety_compliance, bins=20, alpha=0.7, color='green')
        axes[1, 1].set_title('Safety Compliance Distribution')
        axes[1, 1].set_xlabel('Safety Compliance')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'safety_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(
        self,
        results: Dict[str, Any],
        model_name: str
    ) -> Path:
        """Generate markdown report."""
        aggregate_metrics = results['aggregate_metrics']
        episode_metrics = results['episode_metrics']
        
        report_file = self.output_dir / "final_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# {model_name} Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Episodes:** {aggregate_metrics.get('num_episodes', 0)}\n")
            f.write(f"- **Success Rate:** {aggregate_metrics.get('success_rate', 0):.2%}\n")
            f.write(f"- **Mean Episode Length:** {aggregate_metrics.get('mean_episode_length', 0):.1f} steps\n")
            f.write(f"- **Mean Total Reward:** {aggregate_metrics.get('mean_total_reward', 0):.2f}\n\n")
            
            # Task Performance
            f.write("## Task Performance\n\n")
            f.write(f"- **Mean Liquid Reduction:** {aggregate_metrics.get('mean_liquid_reduction', 0):.2%}\n")
            f.write(f"- **Mean Contaminant Reduction:** {aggregate_metrics.get('mean_contaminant_reduction', 0):.2%}\n")
            f.write(f"- **Mean Collision Count:** {aggregate_metrics.get('mean_collision_count', 0):.1f}\n")
            f.write(f"- **Mean Safety Violations:** {aggregate_metrics.get('mean_safety_violations', 0):.1f}\n\n")
            
            # Safety Analysis
            f.write("## Safety Analysis\n\n")
            f.write(f"- **Safety Violation Rate:** {aggregate_metrics.get('mean_safety_violations', 0) / aggregate_metrics.get('mean_episode_length', 1):.3f}\n")
            f.write(f"- **Collision Rate:** {aggregate_metrics.get('mean_collision_count', 0) / aggregate_metrics.get('mean_episode_length', 1):.3f}\n\n")
            
            # Plots
            f.write("## Visualizations\n\n")
            f.write("![Metrics Plots](metrics_plots.png)\n\n")
            f.write("![Episode Analysis](episode_plots.png)\n\n")
            f.write("![Safety Analysis](safety_plots.png)\n\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write("| Episode | Length | Total Reward | Liquid Reduction | Contaminant Reduction | Safety Violations | Success |\n")
            f.write("|---------|--------|--------------|------------------|----------------------|-------------------|----------|\n")
            
            for i, metrics in enumerate(episode_metrics):
                f.write(f"| {i+1} | {metrics['episode_length']} | {metrics['total_reward']:.2f} | {metrics['liquid_reduction']:.2%} | {metrics['contaminant_reduction']:.2%} | {metrics['safety_violations']} | {'✓' if metrics['success'] else '✗'} |\n")
        
        return report_file


def generate_evaluation_report(
    results: Dict[str, Any],
    output_dir: Path,
    model_name: str = "Model"
) -> Path:
    """Generate evaluation report."""
    report_generator = EvaluationReport(output_dir)
    return report_generator.generate_report(results, model_name)
