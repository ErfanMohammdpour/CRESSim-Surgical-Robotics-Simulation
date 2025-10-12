"""
Benchmark system for ablation studies and performance comparison.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

from .evaluator import ModelEvaluator, AblationEvaluator
from .metrics import ComparativeMetrics, MetricsCalculator
from .plots import PlotGenerator
from ..utils.io import ensure_dir, load_config

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Main benchmark runner for ablation studies and comparisons.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        # Load base configuration
        self.base_config = load_config("configs/train.yaml")
        self.env_config = load_config("configs/env.yaml")
        self.safety_config = load_config("configs/safety.yaml")
        
        # Results storage
        self.results = {}
        self.comparison_metrics = ComparativeMetrics()
        
        # Plot generator
        self.plot_generator = PlotGenerator(str(self.output_dir))
    
    def run_ablation_study(self, num_episodes: int = 5) -> Dict[str, Any]:
        """Run comprehensive ablation study."""
        logger.info("Starting ablation study...")
        
        # Define ablation variations
        variations = self._define_ablation_variations()
        
        # Run each variation
        for name, config in variations.items():
            logger.info(f"Running ablation: {name}")
            
            try:
                # Create evaluator with modified config
                evaluator = self._create_evaluator_with_config(config)
                
                # Run evaluation
                metrics = evaluator.evaluate(
                    num_episodes=num_episodes,
                    render=False,
                    save_videos=False
                )
                
                # Store results
                self.results[name] = metrics
                self.comparison_metrics.add_result(name, metrics)
                
                logger.info(f"Completed ablation: {name}")
                
            except Exception as e:
                logger.error(f"Failed to run ablation {name}: {e}")
                continue
        
        # Generate comparison analysis
        self._generate_ablation_analysis()
        
        logger.info("Ablation study completed")
        return self.results
    
    def _define_ablation_variations(self) -> Dict[str, Dict[str, Any]]:
        """Define ablation study variations."""
        variations = {}
        
        # Baseline (RL-only)
        variations['RL_Only'] = {
            'algorithm': 'ppo',
            'safety_config': {
                'enabled': False
            }
        }
        
        # IL→RL
        variations['IL_to_RL'] = {
            'algorithm': 'ppo',
            'checkpoint_path': 'data/checkpoints/il_best_model.pth',  # Would need to be trained
            'safety_config': {
                'enabled': False
            }
        }
        
        # IL→RL+Safety
        variations['IL_to_RL_Safety'] = {
            'algorithm': 'ppo',
            'checkpoint_path': 'data/checkpoints/il_best_model.pth',
            'safety_config': {
                'enabled': True,
                'd_safe': 0.05,
                'd_warning': 0.1
            }
        }
        
        # RL+Safety (no IL)
        variations['RL_Safety'] = {
            'algorithm': 'ppo',
            'safety_config': {
                'enabled': True,
                'd_safe': 0.05,
                'd_warning': 0.1
            }
        }
        
        # Different safety thresholds
        variations['RL_Safety_Strict'] = {
            'algorithm': 'ppo',
            'safety_config': {
                'enabled': True,
                'd_safe': 0.03,
                'd_warning': 0.06
            }
        }
        
        variations['RL_Safety_Loose'] = {
            'algorithm': 'ppo',
            'safety_config': {
                'enabled': True,
                'd_safe': 0.08,
                'd_warning': 0.15
            }
        }
        
        # Different algorithms
        variations['SAC_Only'] = {
            'algorithm': 'sac',
            'safety_config': {
                'enabled': False
            }
        }
        
        variations['SAC_Safety'] = {
            'algorithm': 'sac',
            'safety_config': {
                'enabled': True,
                'd_safe': 0.05,
                'd_warning': 0.1
            }
        }
        
        return variations
    
    def _create_evaluator_with_config(self, config: Dict[str, Any]) -> ModelEvaluator:
        """Create evaluator with modified configuration."""
        # Merge with base config
        merged_config = self.base_config.copy()
        merged_config.update(config)
        
        # Use best available checkpoint
        checkpoint_path = config.get('checkpoint_path', 'data/checkpoints/best_model.pth')
        
        return ModelEvaluator(
            checkpoint_path=checkpoint_path,
            output_dir=str(self.output_dir / "temp"),
            config=merged_config
        )
    
    def _generate_ablation_analysis(self) -> None:
        """Generate comprehensive ablation analysis."""
        # Generate comparison plots
        self.plot_generator.plot_ablation_comparison(self.results)
        
        # Generate detailed comparison table
        self._generate_comparison_table()
        
        # Generate statistical analysis
        self._generate_statistical_analysis()
        
        # Save results
        self._save_ablation_results()
    
    def _generate_comparison_table(self) -> None:
        """Generate detailed comparison table."""
        # Extract key metrics
        key_metrics = [
            'success_rate',
            'mean_reward',
            'std_reward',
            'mean_liquid_reduction',
            'mean_contaminant_reduction',
            'mean_collisions_per_episode',
            'mean_safety_violations_per_episode',
            'mean_action_smoothness',
            'mean_path_length'
        ]
        
        # Create DataFrame
        data = []
        for method, metrics in self.results.items():
            row = {'Method': method}
            for metric in key_metrics:
                row[metric] = metrics.get(metric, 0.0)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by success rate
        df = df.sort_values('success_rate', ascending=False)
        
        # Save table
        table_file = self.output_dir / "comparison_table.csv"
        df.to_csv(table_file, index=False)
        
        # Save markdown table
        markdown_file = self.output_dir / "comparison_table.md"
        with open(markdown_file, 'w') as f:
            f.write("# Ablation Study Results\n\n")
            f.write(df.to_markdown(index=False))
        
        logger.info(f"Comparison table saved to {table_file}")
    
    def _generate_statistical_analysis(self) -> None:
        """Generate statistical analysis of results."""
        if len(self.results) < 2:
            return
        
        # Extract data for statistical tests
        success_rates = []
        mean_rewards = []
        liquid_reductions = []
        
        for method, metrics in self.results.items():
            success_rates.append(metrics.get('success_rate', 0.0))
            mean_rewards.append(metrics.get('mean_reward', 0.0))
            liquid_reductions.append(metrics.get('mean_liquid_reduction', 0.0))
        
        # Calculate statistics
        stats = {
            'success_rate': {
                'mean': float(np.mean(success_rates)),
                'std': float(np.std(success_rates)),
                'min': float(np.min(success_rates)),
                'max': float(np.max(success_rates))
            },
            'mean_reward': {
                'mean': float(np.mean(mean_rewards)),
                'std': float(np.std(mean_rewards)),
                'min': float(np.min(mean_rewards)),
                'max': float(np.max(mean_rewards))
            },
            'liquid_reduction': {
                'mean': float(np.mean(liquid_reductions)),
                'std': float(np.std(liquid_reductions)),
                'min': float(np.min(liquid_reductions)),
                'max': float(np.max(liquid_reductions))
            }
        }
        
        # Find best performing methods
        best_success = max(self.results.items(), key=lambda x: x[1].get('success_rate', 0))
        best_reward = max(self.results.items(), key=lambda x: x[1].get('mean_reward', 0))
        best_liquid = max(self.results.items(), key=lambda x: x[1].get('mean_liquid_reduction', 0))
        
        stats['best_methods'] = {
            'success_rate': best_success[0],
            'mean_reward': best_reward[0],
            'liquid_reduction': best_liquid[0]
        }
        
        # Save statistics
        stats_file = self.output_dir / "statistical_analysis.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistical analysis saved to {stats_file}")
    
    def _save_ablation_results(self) -> None:
        """Save complete ablation results."""
        # Save raw results
        results_file = self.output_dir / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save comparison metrics
        comparison_file = self.output_dir / "comparison_metrics.json"
        self.comparison_metrics.save_comparison(str(comparison_file))
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info(f"Ablation results saved to {self.output_dir}")
    
    def _generate_summary_report(self) -> None:
        """Generate summary report."""
        report = {
            'ablation_study_summary': {
                'total_variations': len(self.results),
                'evaluation_date': str(pd.Timestamp.now()),
                'base_config': self.base_config
            },
            'key_findings': self._extract_key_findings(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_file = self.output_dir / "summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(report)
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from results."""
        findings = []
        
        if not self.results:
            return findings
        
        # Find best overall method
        best_method = max(self.results.items(), key=lambda x: x[1].get('success_rate', 0))
        findings.append(f"Best performing method: {best_method[0]} (success rate: {best_method[1].get('success_rate', 0):.3f})")
        
        # Compare IL vs RL
        if 'IL_to_RL' in self.results and 'RL_Only' in self.results:
            il_success = self.results['IL_to_RL'].get('success_rate', 0)
            rl_success = self.results['RL_Only'].get('success_rate', 0)
            if il_success > rl_success:
                findings.append(f"Imitation learning provides {il_success - rl_success:.3f} improvement in success rate")
            else:
                findings.append(f"RL-only performs {rl_success - il_success:.3f} better than IL→RL")
        
        # Compare safety impact
        if 'RL_Safety' in self.results and 'RL_Only' in self.results:
            safety_success = self.results['RL_Safety'].get('success_rate', 0)
            no_safety_success = self.results['RL_Only'].get('success_rate', 0)
            safety_violations = self.results['RL_Safety'].get('mean_safety_violations_per_episode', 0)
            no_safety_violations = self.results['RL_Only'].get('mean_safety_violations_per_episode', 0)
            
            if safety_violations < no_safety_violations:
                findings.append(f"Safety shield reduces violations by {no_safety_violations - safety_violations:.2f} per episode")
            else:
                findings.append("Safety shield does not significantly reduce violations")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # Find best method
        best_method = max(self.results.items(), key=lambda x: x[1].get('success_rate', 0))
        
        recommendations.append(f"Recommended approach: {best_method[0]}")
        
        # Safety recommendations
        if 'RL_Safety' in self.results:
            safety_violations = self.results['RL_Safety'].get('mean_safety_violations_per_episode', 0)
            if safety_violations < 1.0:
                recommendations.append("Safety shield is effective and should be used in production")
            else:
                recommendations.append("Safety shield needs tuning to be more effective")
        
        # IL recommendations
        if 'IL_to_RL' in self.results and 'RL_Only' in self.results:
            il_success = self.results['IL_to_RL'].get('success_rate', 0)
            rl_success = self.results['RL_Only'].get('success_rate', 0)
            if il_success > rl_success:
                recommendations.append("Imitation learning provides good initialization for RL training")
            else:
                recommendations.append("RL-only training may be sufficient without imitation learning")
        
        return recommendations
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> None:
        """Generate markdown summary report."""
        markdown_file = self.output_dir / "summary_report.md"
        
        with open(markdown_file, 'w') as f:
            f.write("# Ablation Study Summary Report\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write(f"**Total Variations:** {report['ablation_study_summary']['total_variations']}\n")
            f.write(f"**Evaluation Date:** {report['ablation_study_summary']['evaluation_date']}\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            for i, finding in enumerate(report['key_findings'], 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # Results table
            f.write("## Results Summary\n\n")
            f.write("See `comparison_table.md` for detailed results.\n\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("- `comparison_table.csv` - Detailed results table\n")
            f.write("- `comparison_table.md` - Markdown results table\n")
            f.write("- `ablation_results.json` - Raw results data\n")
            f.write("- `statistical_analysis.json` - Statistical analysis\n")
            f.write("- `ablation_comparison.png` - Comparison plots\n")
            f.write("- `summary_dashboard.png` - Summary dashboard\n")
        
        logger.info(f"Markdown report saved to {markdown_file}")


def run_benchmark(output_dir: str, num_episodes: int = 5) -> Dict[str, Any]:
    """Run complete benchmark suite."""
    runner = BenchmarkRunner(output_dir)
    return runner.run_ablation_study(num_episodes)
