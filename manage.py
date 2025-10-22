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


def update_config_with_cli_overrides(config: dict, device: str, no_amp: bool, compile: bool, 
                                    num_workers: Optional[int], gpu_mem_fraction: float) -> dict:
    """Update config with CLI overrides."""
    # Device setting
    if device != "auto":
        config["device"] = device
    
    # AMP setting
    if no_amp:
        config["amp"] = False
    
    # Compile setting
    if compile:
        config["compile"] = True
    
    # Number of workers
    if num_workers is not None:
        config["num_workers"] = num_workers
    
    # GPU memory fraction
    if gpu_mem_fraction != 1.0:
        config["gpu_memory_fraction"] = gpu_mem_fraction
    
    return config


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
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", help="Device to use")
@click.option("--no-amp", is_flag=True, help="Disable Automatic Mixed Precision")
@click.option("--compile", is_flag=True, help="Enable torch.compile for speed")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers")
@click.option("--gpu-mem-fraction", type=float, default=1.0, help="GPU memory fraction to use")
def bootstrap(force: bool, skip_tests: bool, device: str, no_amp: bool, compile: bool, 
              num_workers: Optional[int], gpu_mem_fraction: float) -> None:
    """Bootstrap the environment: install deps, download datasets, clone simulator."""
    console.print(Panel.fit("Bootstrapping CRESSim Suction RL", style="bold blue"))
    
    # Load and update configuration with CLI overrides
    train_config = load_config("configs/train.yaml")
    train_config = update_config_with_cli_overrides(
        train_config, device, no_amp, compile, num_workers, gpu_mem_fraction
    )
    
    # Log device configuration
    from utils.device import cuda_info_string
    console.print(f"[cyan]Device: {device}[/cyan]")
    console.print(f"[cyan]AMP: {not no_amp}[/cyan]")
    console.print(f"[cyan]Compile: {compile}[/cyan]")
    console.print(f"[cyan]Workers: {num_workers or 'auto'}[/cyan]")
    console.print(f"[cyan]GPU Memory Fraction: {gpu_mem_fraction}[/cyan]")
    
    if device in ["auto", "cuda"]:
        console.print(f"[cyan]CUDA Info:[/cyan]")
        console.print(cuda_info_string())
    
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
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", help="Device to use")
@click.option("--no-amp", is_flag=True, help="Disable Automatic Mixed Precision")
@click.option("--compile", is_flag=True, help="Enable torch.compile for speed")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers")
@click.option("--gpu-mem-fraction", type=float, default=1.0, help="GPU memory fraction to use")
def train_il(config: str, demos_dir: Optional[str], output_dir: Optional[str], dagger: bool,
             device: str, no_amp: bool, compile: bool, num_workers: Optional[int], gpu_mem_fraction: float) -> None:
    """Train imitation learning (behavior cloning) model."""
    console.print(Panel.fit("Training Imitation Learning Model", style="bold yellow"))
    
    paths = load_paths_config()
    train_config = load_config(config)
    
    # Update config with CLI overrides
    train_config = update_config_with_cli_overrides(
        train_config, device, no_amp, compile, num_workers, gpu_mem_fraction
    )
    
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
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", help="Device to use")
@click.option("--no-amp", is_flag=True, help="Disable Automatic Mixed Precision")
@click.option("--compile", is_flag=True, help="Enable torch.compile for speed")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers")
@click.option("--gpu-mem-fraction", type=float, default=1.0, help="GPU memory fraction to use")
def train_rl(config: str, checkpoint: Optional[str], output_dir: Optional[str], timesteps: Optional[int], mock: bool,
             device: str, no_amp: bool, compile: bool, num_workers: Optional[int], gpu_mem_fraction: float) -> None:
    """Train reinforcement learning (PPO) model with safety shield."""
    console.print(Panel.fit("Training RL Model with Safety Shield", style="bold red"))
    
    paths = load_paths_config()
    train_config = load_config(config)
    
    # Update config with CLI overrides
    train_config = update_config_with_cli_overrides(
        train_config, device, no_amp, compile, num_workers, gpu_mem_fraction
    )
    
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
@click.option("--mock", is_flag=True, help="Use mock environment (no Unity required)")
@click.option("--report", is_flag=True, help="Generate final acceptance report")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", help="Device to use")
@click.option("--no-amp", is_flag=True, help="Disable Automatic Mixed Precision")
@click.option("--compile", is_flag=True, help="Enable torch.compile for speed")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers")
@click.option("--gpu-mem-fraction", type=float, default=1.0, help="GPU memory fraction to use")
def eval(checkpoint: str, num_episodes: int, output_dir: Optional[str], render: bool, mock: bool, report: bool,
         device: str, no_amp: bool, compile: bool, num_workers: Optional[int], gpu_mem_fraction: float) -> None:
    """Evaluate trained model and generate metrics/videos."""
    console.print(Panel.fit("📊 Evaluating Model", style="bold purple"))
    
    paths = load_paths_config()
    
    # Load and update configuration with CLI overrides
    train_config = load_config("configs/train.yaml")
    train_config = update_config_with_cli_overrides(
        train_config, device, no_amp, compile, num_workers, gpu_mem_fraction
    )
    
    if output_dir is None:
        output_dir = paths["videos_dir"]
    
    try:
        from src.eval.evaluator import Evaluator
        evaluator = Evaluator(checkpoint_path=checkpoint, output_dir=output_dir, config=train_config)
        results = evaluator.evaluate(num_episodes=num_episodes, render=render, mock=mock)
        
        # Generate final report if requested
        if report:
            from src.eval.report import generate_final_report
            report_path = generate_final_report(results, output_dir)
            console.print(f"[green]📋 Final report generated: {report_path}[/green]")
        
        # Print summary
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        aggregate_metrics = results.get('aggregate_metrics', {})
        for metric, value in aggregate_metrics.items():
            if isinstance(value, (int, float)):
                table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        console.print(f"\n[green]✅ Evaluation completed! Results saved to {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to evaluate model: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--output-dir", default=None, help="Output directory for benchmark results")
@click.option("--num-episodes", "-n", default=5, help="Episodes per experiment")
@click.option("--mock", is_flag=True, help="Use mock environment (no Unity required)")
def bench(output_dir: Optional[str], num_episodes: int, mock: bool) -> None:
    """Run ablation studies and generate comparison results."""
    console.print(Panel.fit("🔬 Running Benchmark Experiments", style="bold cyan"))
    
    paths = load_paths_config()
    
    if output_dir is None:
        output_dir = paths["benchmarks_dir"]
    
    try:
        from src.eval.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(output_dir=output_dir)
        
        # Define checkpoint paths for ablation study
        checkpoint_paths = {
            'rl_only': Path(paths["checkpoints_dir"]) / "rl_best_model",
            'il_rl': Path(paths["checkpoints_dir"]) / "rl_best_model",  # Same for now
            'il_rl_safety': Path(paths["checkpoints_dir"]) / "rl_best_model"  # Same for now
        }
        
        results = runner.run_ablation_study(
            checkpoint_paths=checkpoint_paths,
            num_episodes=num_episodes,
            mock=mock
        )
        
        # Print comparison table
        console.print("\n[bold]Benchmark Results:[/bold]")
        ablation_report = results.get('ablation_report', {})
        summary = ablation_report.get('summary', {})
        
        table = Table(title="Ablation Study Results")
        table.add_column("Configuration", style="cyan")
        table.add_column("Success Rate", style="green")
        table.add_column("Mean Reward", style="yellow")
        table.add_column("Safety Violations", style="red")
        
        for config_name, metrics in summary.items():
            table.add_row(
                config_name,
                f"{metrics.get('success_rate', 0):.2%}",
                f"{metrics.get('mean_reward', 0):.2f}",
                f"{metrics.get('mean_safety_violations', 0):.1f}"
            )
        
        console.print(table)
        console.print(f"\n[green]✅ Benchmark completed! Results saved to {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to run benchmark: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", default="configs/train.yaml", help="Training config file")
@click.option("--output-dir", default=None, help="Output directory for models")
@click.option("--epochs", "-e", default=20, help="Number of training epochs")
def pretrain_vision(config: str, output_dir: Optional[str], epochs: int) -> None:
    """Pretrain vision models on Kvasir-SEG dataset."""
    console.print(Panel.fit("👁️ Pretraining Vision Models", style="bold blue"))
    
    paths = load_paths_config()
    train_config = load_config(config)
    
    if output_dir is None:
        output_dir = paths["models_dir"]
    
    # Check if Kvasir-SEG dataset exists
    kvasir_dir = Path(paths["kvasir_seg_dir"])
    if not kvasir_dir.exists():
        console.print("[red]Error: Kvasir-SEG dataset not found![/red]")
        console.print("Please run 'python manage.py bootstrap' first")
        sys.exit(1)
    
    try:
        from src.vision.pretrain import pretrain_safety_segmentation
        
        # Create pretraining config
        pretrain_config = {
            "image_size": [64, 64],
            "num_classes": 3,
            "base_channels": 32,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "num_epochs": epochs,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        metrics = pretrain_safety_segmentation(
            data_dir=kvasir_dir,
            output_dir=Path(output_dir),
            config=pretrain_config
        )
        
        # Print results
        table = Table(title="Pretraining Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        console.print(f"\n[green]✅ Vision pretraining completed! Model saved to {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to pretrain vision models: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--build-path", default=None, help="Path to Unity headless build")
@click.option("--port", default=5005, help="Port for Unity communication")
@click.option("--log-file", default=None, help="Log file for Unity output")
def headless_run(build_path: Optional[str], port: int, log_file: Optional[str]) -> None:
    """Launch Unity headless build for server deployment."""
    console.print(Panel.fit("🖥️ Launching Unity Headless Build", style="bold magenta"))
    
    paths = load_paths_config()
    
    if build_path is None:
        build_path = paths["unity_build"]
    
    build_path = Path(build_path)
    
    if not build_path.exists():
        console.print(f"[red]Error: Unity build not found: {build_path}[/red]")
        console.print("Please build the Unity environment first (see README)")
        sys.exit(1)
    
    try:
        from src.utils.sim import launch_unity_headless, wait_for_unity_connection
        
        # Launch Unity headless
        log_path = Path(log_file) if log_file else None
        process = launch_unity_headless(
            build_path=build_path,
            log_file=log_path,
            port=port
        )
        
        console.print(f"[green]Unity launched with PID: {process.pid}[/green]")
        console.print(f"[green]Listening on port: {port}[/green]")
        
        # Wait for connection
        if wait_for_unity_connection(port, timeout=60):
            console.print("[green]✅ Unity connection established![/green]")
            console.print("[yellow]Press Ctrl+C to stop Unity[/yellow]")
            
            try:
                # Keep running until interrupted
                process.wait()
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping Unity...[/yellow]")
                process.terminate()
                process.wait()
                console.print("[green]✅ Unity stopped[/green]")
        else:
            console.print("[red]❌ Failed to establish Unity connection[/red]")
            process.terminate()
            sys.exit(1)
        
    except Exception as e:
        console.print(f"[red]Failed to launch Unity: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--logs", is_flag=True, help="Remove log files")
@click.option("--models", is_flag=True, help="Remove model checkpoints")
@click.option("--videos", is_flag=True, help="Remove video files")
@click.option("--all", is_flag=True, help="Remove all temporary files")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def clean(logs: bool, models: bool, videos: bool, all: bool, confirm: bool) -> None:
    """Clean up temporary files and logs."""
    console.print(Panel.fit("🧹 Cleaning Up Files", style="bold yellow"))
    
    paths = load_paths_config()
    
    # Determine what to clean
    if all:
        clean_logs = clean_models = clean_videos = True
    else:
        clean_logs = logs
        clean_models = models
        clean_videos = videos
    
    if not any([clean_logs, clean_models, clean_videos]):
        console.print("[yellow]No cleanup options specified. Use --help for options.[/yellow]")
        return
    
    # Show what will be cleaned
    cleanup_items = []
    if clean_logs:
        cleanup_items.append("Log files")
    if clean_models:
        cleanup_items.append("Model checkpoints")
    if clean_videos:
        cleanup_items.append("Video files")
    
    console.print(f"[yellow]Will clean: {', '.join(cleanup_items)}[/yellow]")
    
    # Confirm unless --confirm flag
    if not confirm:
        if not click.confirm("Are you sure you want to continue?"):
            console.print("[yellow]Cleanup cancelled[/yellow]")
            return
    
    try:
        import shutil
        
        # Clean logs
        if clean_logs:
            logs_dir = Path(paths["logs_dir"])
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
                logs_dir.mkdir(parents=True, exist_ok=True)
                console.print(f"[green]✅ Cleaned logs: {logs_dir}[/green]")
        
        # Clean models
        if clean_models:
            models_dir = Path(paths["checkpoints_dir"])
            if models_dir.exists():
                for model_file in models_dir.glob("*.pth"):
                    model_file.unlink()
                console.print(f"[green]✅ Cleaned models: {models_dir}[/green]")
        
        # Clean videos
        if clean_videos:
            videos_dir = Path(paths["videos_dir"])
            if videos_dir.exists():
                for video_file in videos_dir.glob("*.mp4"):
                    video_file.unlink()
                console.print(f"[green]✅ Cleaned videos: {videos_dir}[/green]")
        
        console.print("[green]✅ Cleanup completed![/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to clean files: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
