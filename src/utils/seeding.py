"""
Seeding utilities for reproducible experiments.
"""

import random
import numpy as np
import torch
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


def get_random_state() -> Dict[str, Any]:
    """Get current random state."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_random_state(state: Dict[str, Any]) -> None:
    """Set random state from saved state."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if torch.cuda.is_available() and state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    
    logger.info("Restored random state")


class SeededRandom:
    """Context manager for seeded random operations."""
    
    def __init__(self, seed: int):
        self.seed = seed
        self.old_state = None
    
    def __enter__(self):
        self.old_state = get_random_state()
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_state is not None:
            set_random_state(self.old_state)
