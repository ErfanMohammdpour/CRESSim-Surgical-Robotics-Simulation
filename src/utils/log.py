"""
Logging utilities for the project.
"""

import logging
import os
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    log_file = log_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def log_metrics(metrics: dict, log_dir: Path, step: Optional[int] = None) -> None:
    """Log metrics to JSON file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp and step
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "metrics": metrics
    }
    
    # Append to metrics file
    metrics_file = log_dir / "metrics.jsonl"
    with open(metrics_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def log_config(config: dict, log_dir: Path) -> None:
    """Log configuration to JSON file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = log_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)