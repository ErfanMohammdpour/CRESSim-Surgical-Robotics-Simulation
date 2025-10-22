"""
Benchmark runner for ablation studies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

from .evaluator import Evaluator
from ..utils.seeding import set_seed

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run benchmark experiments and ablation studies."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BenchmarkRunner initialized: {output_dir}")
    
    def run_ablation_study(
        self,
        checkpoint_paths: Dict[str, Path],
        num_episodes: int = 5,
        mock: bool = True
    ) -> Dict[str, Any]:
        """Run ablation study comparing different configurations."""
        logger.info("Running ablation study")
        
        # Define ablation configurations
        ablation_configs = {
            'RL_only': {
                'name': 'RL Only',
                'description': 'No IL warm-start, no safety shield',
                'checkpoint': checkpoint_paths.get('rl_only')
            },
            'IL_RL': {
                'name': 'IL → RL',
                'description': 'IL warm-start, no safety shield',
                'checkpoint': checkpoint_paths.get('il_rl')
            },
            'IL_RL_Safety': {
                'name': 'IL → RL + Safety',
                'description': 'IL warm-start with safety shield',
                'checkpoint': checkpoint_paths.get('il_rl_safety')
            }
        }
        
        ablation_results = {}
        
        for config_name, config_info in ablation_configs.items():
            checkpoint_path = config_info['checkpoint']
            
            if checkpoint_path is None or not checkpoint_path.exists():
                logger.warning(f"No checkpoint found for {config_name}, skipping")
                continue
            
            logger.info(f"Running ablation: {config_name}")
            
            # Create evaluator
            evaluator = Evaluator(
                checkpoint_path=checkpoint_path,
                output_dir=self.output_dir / f"ablation_{config_name}",
                device="cuda"
            )
            
            # Evaluate model
            results = evaluator.evaluate(
                num_episodes=num_episodes,
                render=False,
                mock=mock
            )
            
            ablation_results[config_name] = {
                'config_info': config_info,
                'results': results
            }
        
        # Generate ablation report
        ablation_report = self._generate_ablation_report(ablation_results)
        
        # Save results
        self._save_ablation_results(ablation_results, ablation_report)
        
        return {
            'ablation_results': ablation_results,
            'ablation_report': ablation_report
        }
    
    def _generate_ablation_report(self, ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ablation study report."""
        report = {
            'summary': {},
            'detailed_comparison': {},
            'best_configs': {}
        }
        
        # Extract key metrics for each configuration
        for config_name, config_data in ablation_results.items():
            results = config_data['results']
            aggregate_metrics = results['aggregate_metrics']
            
            report['summary'][config_name] = {
                'success_rate': aggregate_metrics.get('success_rate', 0),
                'mean_reward': aggregate_metrics.get('mean_total_reward', 0),
                'mean_liquid_reduction': aggregate_metrics.get('mean_liquid_reduction', 0),
                'mean_contaminant_reduction': aggregate_metrics.get('mean_contaminant_reduction', 0),
                'mean_safety_violations': aggregate_metrics.get('mean_safety_violations', 0),
                'mean_collision_count': aggregate_metrics.get('mean_collision_count', 0)
            }
        
        # Find best configuration for each metric
        for metric in ['success_rate', 'mean_reward', 'mean_liquid_reduction', 'mean_contaminant_reduction']:
            best_config = max(
                report['summary'].keys(),
                key=lambda k: report['summary'][k][metric]
            )
            report['best_configs'][metric] = best_config
        
        # Find best configuration for safety (lowest violations)
        best_safety_config = min(
            report['summary'].keys(),
            key=lambda k: report['summary'][k]['mean_safety_violations']
        )
        report['best_configs']['safety'] = best_safety_config
        
        # Find best configuration for collision avoidance
        best_collision_config = min(
            report['summary'].keys(),
            key=lambda k: report['summary'][k]['mean_collision_count']
        )
        report['best_configs']['collision'] = best_collision_config
        
        return report
    
    def _save_ablation_results(
        self,
        ablation_results: Dict[str, Any],
        ablation_report: Dict[str, Any]
    ):
        """Save ablation study results."""
        # Save detailed results
        results_file = self.output_dir / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(ablation_results, f, indent=2)
        
        # Save summary report
        report_file = self.output_dir / "ablation_report.json"
        with open(report_file, 'w') as f:
            json.dump(ablation_report, f, indent=2)
        
        # Save CSV summary
        csv_file = self.output_dir / "ablation_summary.csv"
        summary_data = []
        
        for config_name, metrics in ablation_report['summary'].items():
            row = {'configuration': config_name}
            row.update(metrics)
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
        
        # Generate markdown report
        self._generate_ablation_markdown(ablation_report)
        
        logger.info(f"Ablation results saved to {self.output_dir}")
    
    def _generate_ablation_markdown(self, ablation_report: Dict[str, Any]):
        """Generate markdown report for ablation study."""
        report_file = self.output_dir / "ablation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Ablation Study Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Configuration | Success Rate | Mean Reward | Liquid Reduction | Contaminant Reduction | Safety Violations | Collision Count |\n")
            f.write("|---------------|--------------|-------------|------------------|----------------------|-------------------|-----------------|\n")
            
            for config_name, metrics in ablation_report['summary'].items():
                f.write(f"| {config_name} | {metrics['success_rate']:.2%} | {metrics['mean_reward']:.2f} | {metrics['mean_liquid_reduction']:.2%} | {metrics['mean_contaminant_reduction']:.2%} | {metrics['mean_safety_violations']:.1f} | {metrics['mean_collision_count']:.1f} |\n")
            
            # Best configurations
            f.write("\n## Best Configurations\n\n")
            for metric, best_config in ablation_report['best_configs'].items():
                f.write(f"- **{metric}**: {best_config}\n")
            
            # Analysis
            f.write("\n## Analysis\n\n")
            f.write("### Key Findings\n\n")
            
            # Find overall best configuration
            overall_scores = {}
            for config_name, metrics in ablation_report['summary'].items():
                # Weighted score combining all metrics
                score = (
                    metrics['success_rate'] * 0.3 +
                    metrics['mean_reward'] * 0.2 +
                    metrics['mean_liquid_reduction'] * 0.2 +
                    metrics['mean_contaminant_reduction'] * 0.2 +
                    (1 - metrics['mean_safety_violations']) * 0.1
                )
                overall_scores[config_name] = score
            
            best_overall = max(overall_scores.keys(), key=lambda k: overall_scores[k])
            f.write(f"- **Overall Best Configuration**: {best_overall}\n")
            f.write(f"- **Overall Score**: {overall_scores[best_overall]:.3f}\n\n")
            
            # Safety analysis
            safety_config = ablation_report['best_configs']['safety']
            f.write(f"- **Best Safety Performance**: {safety_config}\n")
            f.write(f"- **Safety Violations**: {ablation_report['summary'][safety_config]['mean_safety_violations']:.1f}\n\n")
            
            # Task performance analysis
            task_config = ablation_report['best_configs']['success_rate']
            f.write(f"- **Best Task Performance**: {task_config}\n")
            f.write(f"- **Success Rate**: {ablation_report['summary'][task_config]['success_rate']:.2%}\n\n")
        
        logger.info(f"Ablation markdown report saved: {report_file}")
    
    def run_hyperparameter_sweep(
        self,
        base_config: Dict[str, Any],
        hyperparameter_ranges: Dict[str, List[Any]],
        num_episodes: int = 3,
        mock: bool = True
    ) -> Dict[str, Any]:
        """Run hyperparameter sweep."""
        logger.info("Running hyperparameter sweep")
        
        # Generate all combinations
        import itertools
        
        param_names = list(hyperparameter_ranges.keys())
        param_values = list(hyperparameter_ranges.values())
        
        combinations = list(itertools.product(*param_values))
        
        sweep_results = {}
        
        for i, combination in enumerate(combinations):
            # Create config with this combination
            config = base_config.copy()
            for param_name, param_value in zip(param_names, combination):
                config[param_name] = param_value
            
            # Create configuration name
            config_name = f"config_{i:03d}"
            for param_name, param_value in zip(param_names, combination):
                config_name += f"_{param_name}_{param_value}"
            
            logger.info(f"Running sweep: {config_name}")
            
            # Train model (simplified - in practice, you'd train here)
            # For now, we'll just evaluate with the base checkpoint
            base_checkpoint = Path("data/checkpoints/rl_best_model")
            
            if base_checkpoint.exists():
                evaluator = Evaluator(
                    checkpoint_path=base_checkpoint,
                    output_dir=self.output_dir / f"sweep_{config_name}",
                    device="cuda"
                )
                
                results = evaluator.evaluate(
                    num_episodes=num_episodes,
                    render=False,
                    mock=mock
                )
                
                sweep_results[config_name] = {
                    'config': config,
                    'results': results
                }
        
        # Save sweep results
        self._save_sweep_results(sweep_results)
        
        return sweep_results
    
    def _save_sweep_results(self, sweep_results: Dict[str, Any]):
        """Save hyperparameter sweep results."""
        # Save detailed results
        results_file = self.output_dir / "sweep_results.json"
        with open(results_file, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        
        # Save CSV summary
        csv_file = self.output_dir / "sweep_summary.csv"
        summary_data = []
        
        for config_name, config_data in sweep_results.items():
            config = config_data['config']
            results = config_data['results']
            aggregate_metrics = results['aggregate_metrics']
            
            row = {'configuration': config_name}
            row.update(config)
            row.update({
                'success_rate': aggregate_metrics.get('success_rate', 0),
                'mean_reward': aggregate_metrics.get('mean_total_reward', 0),
                'mean_liquid_reduction': aggregate_metrics.get('mean_liquid_reduction', 0),
                'mean_contaminant_reduction': aggregate_metrics.get('mean_contaminant_reduction', 0),
                'mean_safety_violations': aggregate_metrics.get('mean_safety_violations', 0)
            })
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Sweep results saved to {self.output_dir}")


def run_ablation_study(
    checkpoint_paths: Dict[str, Path],
    output_dir: Path,
    num_episodes: int = 5,
    mock: bool = True
) -> Dict[str, Any]:
    """Run ablation study."""
    runner = BenchmarkRunner(output_dir)
    return runner.run_ablation_study(
        checkpoint_paths=checkpoint_paths,
        num_episodes=num_episodes,
        mock=mock
    )
