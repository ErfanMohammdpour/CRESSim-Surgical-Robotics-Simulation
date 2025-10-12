"""
Logging utilities for training and evaluation.
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logging(verbose: bool = False, log_dir: Optional[str] = None) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup file handler
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "data" / "logs"
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Logger:
    """Unified logger for training metrics."""
    
    def __init__(self, log_dir: str, name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.tb_writer = SummaryWriter(self.log_dir / "tensorboard" / name)
        
        # CSV logger
        self.csv_file = self.log_dir / f"{name}.csv"
        self.csv_headers_written = False
        
        # Logger
        self.logger = logging.getLogger(name)
    
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log scalar value to TensorBoard and CSV."""
        # TensorBoard
        self.tb_writer.add_scalar(key, value, step)
        
        # CSV
        self._write_csv_row({key: value, "step": step})
    
    def log_scalars(self, metrics: dict, step: int) -> None:
        """Log multiple scalars."""
        for key, value in metrics.items():
            self.log_scalar(key, value, step)
    
    def log_image(self, key: str, image: torch.Tensor, step: int) -> None:
        """Log image to TensorBoard."""
        self.tb_writer.add_image(key, image, step)
    
    def log_histogram(self, key: str, values: torch.Tensor, step: int) -> None:
        """Log histogram to TensorBoard."""
        self.tb_writer.add_histogram(key, values, step)
    
    def _write_csv_row(self, row: dict) -> None:
        """Write row to CSV file."""
        import csv
        
        # Write headers if first time
        if not self.csv_headers_written:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            self.csv_headers_written = True
        
        # Append row
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def close(self) -> None:
        """Close all writers."""
        self.tb_writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
