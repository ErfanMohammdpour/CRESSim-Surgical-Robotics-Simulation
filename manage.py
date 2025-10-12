#!/usr/bin/env python3
"""
CRESSim Suction RL - Main CLI
Windows-first command-line interface for training vision-based RL policies.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List
import yaml
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.io import load_config, ensure_dir

console = Console()


def check_python_version() -> None:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        console.print("[red]Error: Python 3.10+ is required[/red]")
        sys.exit(1)


def load_paths_config() -> dict:
    """Load paths configuration."""
    return load_config("configs/paths.yaml")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config-dir", default="configs", help="Configuration directory")
def cli(verbose: bool, config_dir: str) -> None:
    """CRESSim Suction RL - Vision-based RL for autonomous endoscopic suction."""
    check_python_version()
    
    # Set up logging
    from src.utils.log import setup_logging
    log_dir = Path("data/logs")
    setup_logging(log_dir, level="DEBUG" if verbose else "INFO")
    
    # Store config dir globally
    os.environ["CONFIG_DIR"] = config_dir


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Force re-download/re-clone")
@click.option("--skip-tests", is_flag=True, help="Skip running tests")
def bootstrap(force: bool, skip_tests: bool) -> None:
    """Bootstrap the environment: install deps, download datasets, clone simulator."""
    console.print(Panel.fit("Bootstrapping CRESSim Suction RL", style="bold blue"))
    
    paths = load_paths_config()
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Install Python dependencies
        task1 = progress.add_task("Installing Python dependencies...", total=None)
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True)
            progress.update(task1, description="Dependencies installed")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to install dependencies: {e}[/red]")
            sys.exit(1)
        
        # Download Kvasir-SEG dataset
        task2 = progress.add_task("Downloading Kvasir-SEG dataset...", total=None)
        try:
            from utils.data import download_kvasir_seg
            download_kvasir_seg(paths["kvasir_seg_dir"], force=force)
            progress.update(task2, description="Kvasir-SEG downloaded")
        except Exception as e:
            console.print(f"[red]Failed to download dataset: {e}[/red]")
            console.print(f"[yellow]Please download manually from: https://datasets.simula.no/downloads/kvasir-seg.zip[/yellow]")
            console.print(f"[yellow]Save it to: {paths['kvasir_seg_dir']}/Kvasir-SEG.zip[/yellow]")
            progress.update(task2, description="Kvasir-SEG download failed")
        
        # Clone CRESSim simulator
        task3 = progress.add_task("Cloning CRESSim simulator...", total=None)
        try:
            from utils.sim import clone_cressim
            clone_cressim(paths["sim_dir"], force=force)
            progress.update(task3, description="CRESSim cloned")
        except Exception as e:
            console.print(f"[red]Failed to clone simulator: {e}[/red]")
            console.print(f"[yellow]Please clone manually from: https://github.com/tbs-ualberta/CRESSim[/yellow]")
            console.print(f"[yellow]Clone it to: {paths['sim_dir']}[/yellow]")
            progress.update(task3, description="CRESSim clone failed")
        
        # Check Unity build
        task4 = progress.add_task("Checking Unity build...", total=None)
        unity_build_path = Path(paths["unity_build"])
        if unity_build_path.exists():
            progress.update(task4, description="Unity build found")
        else:
            progress.update(task4, description="Unity build missing - see README for build instructions")
        
        # Run tests
        if not skip_tests:
            task5 = progress.add_task("Running tests...", total=None)
            try:
                subprocess.run([
                    sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"
                ], check=True, capture_output=True)
                progress.update(task5, description="Tests passed")
            except subprocess.CalledProcessError as e:
                console.print(f"[yellow]Some tests failed: {e}[/yellow]")
                progress.update(task5, description="Some tests failed")
    
    console.print("\n[green]Bootstrap complete![/green]")
    console.print("\nNext steps:")
    console.print("1. Build Unity environment (see README)")
    console.print("2. Generate demos: [cyan]python manage.py demos[/cyan]")
    console.print("3. Train IL: [cyan]python manage.py train-il[/cyan]")
    console.print("4. Train RL: [cyan]python manage.py train-rl[/cyan]")


@cli.command()
@click.option("--num-episodes", "-n", default=None, help="Number of episodes to generate")
@click.option("--workers", "-w", default=None, help="Number of worker processes")
@click.option("--mock", is_flag=True, help="Generate mock demos (no Unity required)")
def demos(num_episodes: Optional[int], workers: Optional[int], mock: bool) -> None:
    """Generate imperfect scripted demonstrations."""
    console.print(Panel.fit("Generating Demonstrations", style="bold green"))
    
    paths = load_paths_config()
    demos_config = load_config("configs/demos.yaml")
    
    # Override config with CLI args
    if num_episodes is not None:
        demos_config["num_episodes"] = int(num_episodes)
    if workers is not None:
        demos_config["num_workers"] = int(workers)
    
    try:
        from src.il.demos import generate_demos
        generate_demos(
            output_dir=paths["demos_dir"],
            config=demos_config,
            mock=mock
        )
        console.print("[green]Demos generated successfully![/green]")
    except Exception as e:
        console.print(f"[red]Failed to generate demos: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", default="configs/train.yaml", help="Training config file")
@click.option("--demos-dir", default=None, help="Directory containing demos")
@click.option("--output-dir", default=None, help="Output directory for checkpoints")
@click.option("--dagger", is_flag=True, help="Enable DAgger-lite")
def train_il(config: str, demos_dir: Optional[str], output_dir: Optional[str], dagger: bool) -> None:
    """Train imitation learning (behavior cloning) model."""
    console.print(Panel.fit("Training Imitation Learning Model", style="bold yellow"))
    
    paths = load_paths_config()
    train_config = load_config(config)
    
    # Set default paths
    if demos_dir is None:
        demos_dir = paths["demos_dir"]
    if output_dir is None:
        output_dir = paths["checkpoints_dir"]
    
    try:
        from src.il.bc_trainer import BCTrainer
        trainer = BCTrainer(config=train_config, demos_dir=demos_dir, output_dir=output_dir)
        trainer.train(dagger=dagger)
        console.print("[green]IL training completed![/green]")
    except Exception as e:
        console.print(f"[red]Failed to train IL model: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", default="configs/train.yaml", help="Training config file")
@click.option("--checkpoint", default=None, help="IL checkpoint to start from")
@click.option("--output-dir", default=None, help="Output directory for checkpoints")
@click.option("--timesteps", "-t", default=None, type=int, help="Total training timesteps")
@click.option("--mock", is_flag=True, help="Use mock environment (no Unity required)")
def train_rl(config: str, checkpoint: Optional[str], output_dir: Optional[str], timesteps: Optional[int], mock: bool) -> None:
    """Train reinforcement learning (PPO) model with safety shield."""
    console.print(Panel.fit("Training RL Model with Safety Shield", style="bold red"))
    
    paths = load_paths_config()
    train_config = load_config(config)
    
    # Check Unity build
    unity_build_path = Path(paths["unity_build"])
    if not mock and unity_build_path.exists():
        console.print("[green]Using Unity environment for RL training[/green]")
    elif not mock and not unity_build_path.exists():
        console.print("[red]Error: Unity build not found![/red]")
        console.print(f"Expected at: {unity_build_path}")
        console.print("Please build the Unity environment first (see README)")
        console.print("Or use --mock flag to use mock environment")
        sys.exit(1)
    else:
        console.print("[yellow]Using mock environment for RL training[/yellow]")
    
    # Set default paths
    if output_dir is None:
        output_dir = paths["checkpoints_dir"]
    if timesteps is not None:
        train_config["total_timesteps"] = timesteps
    
    # Get total timesteps from config
    total_timesteps = train_config.get("total_timesteps", 1000000)
    
    try:
        from src.rl.ppo_trainer import PPOTrainer
        trainer = PPOTrainer(
            config=train_config,
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            mock=mock
        )
        trainer.train(total_timesteps=total_timesteps)
        console.print("[green]RL training completed![/green]")
    except Exception as e:
        console.print(f"[red]Failed to train RL model: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--checkpoint", "-c", required=True, help="Model checkpoint to evaluate")
@click.option("--num-episodes", "-n", default=10, help="Number of evaluation episodes")
@click.option("--output-dir", default=None, help="Output directory for results")
@click.option("--render", is_flag=True, help="Render videos during evaluation")
def eval(checkpoint: str, num_episodes: int, output_dir: Optional[str], render: bool) -> None:
    """Evaluate trained model and generate metrics/videos."""
    console.print(Panel.fit("📊 Evaluating Model", style="bold purple"))
    
    paths = load_paths_config()
    
    if output_dir is None:
        output_dir = paths["videos_dir"]
    
    try:
        from src.eval.evaluator import Evaluator
        evaluator = Evaluator(checkpoint_path=checkpoint, output_dir=output_dir)
        results = evaluator.evaluate(num_episodes=num_episodes, render=render)
        
        # Print summary
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for metric, value in results.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        console.print(f"\n[green]✅ Evaluation completed! Results saved to {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to evaluate model: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--output-dir", default=None, help="Output directory for benchmark results")
@click.option("--num-episodes", "-n", default=5, help="Episodes per experiment")
def bench(output_dir: Optional[str], num_episodes: int) -> None:
    """Run ablation studies and generate comparison results."""
    console.print(Panel.fit("🔬 Running Benchmark Experiments", style="bold cyan"))
    
    paths = load_paths_config()
    
    if output_dir is None:
        output_dir = paths["benchmarks_dir"]
    
    try:
        from eval.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(output_dir=output_dir)
        results = runner.run_ablation_study(num_episodes=num_episodes)
        
        # Print comparison table
        console.print("\n[bold]Benchmark Results:[/bold]")
        console.print(results)
        console.print(f"\n[green]✅ Benchmark completed! Results saved to {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to run benchmark: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
