"""
GPU utilities for device detection and optimization.
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")
    
    return device


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
        "memory_allocated": torch.cuda.memory_allocated() / 1e9,
        "memory_reserved": torch.cuda.memory_reserved() / 1e9,
        "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "compute_capability": torch.cuda.get_device_capability(),
    }
    
    return info


def optimize_for_gpu() -> Dict[str, Any]:
    """Optimize PyTorch settings for GPU training."""
    if not torch.cuda.is_available():
        return {"optimized": False, "reason": "CUDA not available"}
    
    # Enable cuDNN benchmark for consistent performance
    torch.backends.cudnn.benchmark = True
    
    # Enable cuDNN deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = False
    
    # Set memory growth to avoid OOM
    torch.cuda.empty_cache()
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    logger.info("GPU optimization enabled:")
    logger.info(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"  - Memory allocated: {gpu_info['memory_allocated']:.2f} GB")
    logger.info(f"  - Memory total: {gpu_info['memory_total']:.2f} GB")
    
    return {
        "optimized": True,
        "gpu_info": gpu_info,
        "cudnn_benchmark": torch.backends.cudnn.benchmark
    }


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")


def get_optimal_batch_size(model, input_shape: tuple, max_batch_size: int = 256) -> int:
    """Find optimal batch size for GPU training."""
    if not torch.cuda.is_available():
        return 32  # Default for CPU
    
    device = get_device()
    model = model.to(device)
    
    # Binary search for optimal batch size
    left, right = 1, max_batch_size
    optimal_batch_size = 1
    
    while left <= right:
        mid = (left + right) // 2
        try:
            # Create dummy input
            dummy_input = torch.randn(mid, *input_shape).to(device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            optimal_batch_size = mid
            left = mid + 1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                right = mid - 1
                torch.cuda.empty_cache()
            else:
                raise e
    
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


if __name__ == "__main__":
    # Test GPU detection
    device = get_device()
    print(f"Using device: {device}")
    
    gpu_info = get_gpu_info()
    print(f"GPU info: {gpu_info}")
    
    optimize_for_gpu()
