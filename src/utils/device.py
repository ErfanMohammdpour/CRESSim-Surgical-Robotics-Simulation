"""
Device utilities for automatic GPU/CPU selection and CUDA management.
"""

import torch
import os
import logging
from typing import Optional, Dict, Any
import subprocess
import platform


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device with automatic fallback.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device: The selected device
    """
    if not prefer_cuda:
        return torch.device("cpu")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    
    # Check CUDA_VISIBLE_DEVICES
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        # CUDA_VISIBLE_DEVICES is set, use device 0 (which maps to the first visible device)
        device_id = 0
    else:
        # Use the first available GPU
        device_id = 0
    
    # Verify device is accessible
    try:
        device = torch.device(f"cuda:{device_id}")
        # Test device with a small tensor
        test_tensor = torch.zeros(1, device=device)
        del test_tensor
        return device
    except Exception as e:
        logging.warning(f"Failed to initialize CUDA device {device_id}: {e}")
        return torch.device("cpu")


def cuda_info_string() -> str:
    """
    Get comprehensive CUDA information string.
    
    Returns:
        str: Formatted CUDA information
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    
    info_lines = []
    info_lines.append(f"CUDA available: {torch.cuda.is_available()}")
    info_lines.append(f"Device count: {device_count}")
    info_lines.append(f"Current device: {current_device}")
    
    # Driver and runtime versions
    try:
        driver_version = torch.cuda.get_driver_version()
        runtime_version = torch.cuda.get_runtime_version()
        info_lines.append(f"Driver version: {driver_version}")
        info_lines.append(f"Runtime version: {runtime_version}")
    except Exception:
        info_lines.append("Version info unavailable")
    
    # Device information
    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # GB
            
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            
            info_lines.append(f"GPU {i}: {props.name}")
            info_lines.append(f"  Compute capability: {props.major}.{props.minor}")
            info_lines.append(f"  Total memory: {total_memory:.1f} GB")
            info_lines.append(f"  Allocated: {allocated:.1f} GB")
            info_lines.append(f"  Cached: {cached:.1f} GB")
        except Exception as e:
            info_lines.append(f"GPU {i}: Error getting properties - {e}")
    
    return "\n".join(info_lines)


def amp_enabled(config: Dict[str, Any]) -> bool:
    """
    Check if Automatic Mixed Precision should be enabled.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        bool: True if AMP should be enabled
    """
    # AMP requires CUDA
    if not torch.cuda.is_available():
        return False
    
    # Check config setting
    amp_setting = config.get("amp", True)
    if not amp_setting:
        return False
    
    # Check device preference
    device_setting = config.get("device", "auto")
    if device_setting == "cpu":
        return False
    
    return True


def setup_cudnn(config: Dict[str, Any]) -> None:
    """
    Configure cuDNN settings based on configuration.
    
    Args:
        config: Training configuration dictionary
    """
    if not torch.cuda.is_available():
        return
    
    # Benchmark mode
    benchmark = config.get("cudnn_benchmark", True)
    torch.backends.cudnn.benchmark = benchmark
    
    # Deterministic mode
    deterministic = config.get("deterministic", False)
    torch.backends.cudnn.deterministic = deterministic
    
    # Float32 matmul precision
    precision = config.get("float32_matmul_precision", "high")
    try:
        torch.set_float32_matmul_precision(precision)
    except Exception as e:
        logging.warning(f"Failed to set float32 matmul precision: {e}")


def get_device_from_config(config: Dict[str, Any]) -> torch.device:
    """
    Get device based on configuration with CLI override support.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device: Selected device
    """
    device_setting = config.get("device", "auto")
    
    if device_setting == "cpu":
        return torch.device("cpu")
    elif device_setting == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return get_device(prefer_cuda=True)
    elif device_setting == "auto":
        return get_device(prefer_cuda=True)
    else:
        logging.warning(f"Unknown device setting '{device_setting}', using auto")
        return get_device(prefer_cuda=True)


def log_device_info(device: torch.device, config: Dict[str, Any]) -> None:
    """
    Log comprehensive device and configuration information.
    
    Args:
        device: Selected device
        config: Training configuration
    """
    logging.info("=" * 60)
    logging.info("DEVICE CONFIGURATION")
    logging.info("=" * 60)
    
    # Device information
    logging.info(f"Selected device: {device}")
    logging.info(f"Device type: {device.type}")
    
    if device.type == "cuda":
        logging.info(f"CUDA device ID: {device.index}")
        logging.info(f"GPU name: {torch.cuda.get_device_name(device.index)}")
        
        # Memory information
        total_memory = torch.cuda.get_device_properties(device.index).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(device.index) / (1024**3)
        cached_memory = torch.cuda.memory_reserved(device.index) / (1024**3)
        
        logging.info(f"Total GPU memory: {total_memory:.1f} GB")
        logging.info(f"Allocated memory: {allocated_memory:.1f} GB")
        logging.info(f"Cached memory: {cached_memory:.1f} GB")
    
    # Configuration settings
    logging.info(f"AMP enabled: {amp_enabled(config)}")
    logging.info(f"cuDNN benchmark: {config.get('cudnn_benchmark', True)}")
    logging.info(f"Deterministic: {config.get('deterministic', False)}")
    logging.info(f"Float32 matmul precision: {config.get('float32_matmul_precision', 'high')}")
    logging.info(f"Compile enabled: {config.get('compile', False)}")
    
    # CUDA_VISIBLE_DEVICES
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        logging.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
    else:
        logging.info("CUDA_VISIBLE_DEVICES: not set")
    
    logging.info("=" * 60)


def optimize_for_gpu(model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Apply GPU optimizations to a model.
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        torch.nn.Module: Optimized model
    """
    device = get_device_from_config(config)
    
    # Move to device
    model = model.to(device)
    
    # Apply torch.compile if enabled
    if config.get("compile", False) and device.type == "cuda":
        try:
            model = torch.compile(model)
            logging.info("Model compiled with torch.compile")
        except Exception as e:
            logging.warning(f"Failed to compile model: {e}")
    
    return model


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer with device-aware settings.
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    device = get_device_from_config(config)
    
    # Get optimizer parameters
    lr = config.get("learning_rate", 3e-4)
    weight_decay = config.get("weight_decay", 1e-5)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Move optimizer state to device if needed
    if device.type == "cuda":
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device, non_blocking=True)
    
    return optimizer


def create_grad_scaler(config: Dict[str, Any]) -> Optional[torch.cuda.amp.GradScaler]:
    """
    Create gradient scaler for AMP if enabled.
    
    Args:
        config: Training configuration
        
    Returns:
        Optional[torch.cuda.amp.GradScaler]: GradScaler if AMP enabled, None otherwise
    """
    if not amp_enabled(config):
        return None
    
    return torch.cuda.amp.GradScaler()


def get_dataloader_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get DataLoader kwargs optimized for the current platform and device.
    
    Args:
        config: Training configuration
        
    Returns:
        Dict[str, Any]: DataLoader kwargs
    """
    device = get_device_from_config(config)
    
    kwargs = {
        "pin_memory": device.type == "cuda",
        "persistent_workers": True,
    }
    
    # Set num_workers based on platform
    if platform.system() == "Windows":
        kwargs["num_workers"] = min(config.get("num_workers", 0), 2)
    else:
        kwargs["num_workers"] = config.get("num_workers", 4)
    
    return kwargs


def handle_oom_error(error: RuntimeError, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle CUDA out of memory errors by adjusting configuration.
    
    Args:
        error: The OOM RuntimeError
        config: Current configuration
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    if "out of memory" not in str(error).lower():
        raise error
    
    logging.warning("CUDA out of memory detected, adjusting configuration...")
    
    # Create updated config
    updated_config = config.copy()
    
    # Reduce batch size by 50%
    if "batch_size" in updated_config:
        updated_config["batch_size"] = max(1, updated_config["batch_size"] // 2)
        logging.info(f"Reduced batch size to {updated_config['batch_size']}")
    
    # Reduce gradient accumulation if present
    if "gradient_accumulation_steps" in updated_config:
        updated_config["gradient_accumulation_steps"] = max(1, updated_config["gradient_accumulation_steps"] // 2)
        logging.info(f"Reduced gradient accumulation steps to {updated_config['gradient_accumulation_steps']}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return updated_config


def warmup_model(model: torch.nn.Module, config: Dict[str, Any], input_shape: tuple) -> None:
    """
    Warm up model with a dummy forward pass to trigger kernel selection.
    
    Args:
        model: PyTorch model
        config: Training configuration
        input_shape: Input tensor shape (batch_size, ...)
    """
    device = get_device_from_config(config)
    
    if device.type != "cuda":
        return
    
    logging.info("Warming up model with dummy forward pass...")
    
    try:
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(input_shape, device=device)
            
            # Forward pass
            if amp_enabled(config):
                with torch.autocast("cuda"):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
        
        logging.info("Model warmup completed")
    except Exception as e:
        logging.warning(f"Model warmup failed: {e}")


def get_memory_usage(device: torch.device) -> Dict[str, float]:
    """
    Get current memory usage for a device.
    
    Args:
        device: Target device
        
    Returns:
        Dict[str, float]: Memory usage in GB
    """
    if device.type != "cuda":
        return {"allocated": 0.0, "cached": 0.0, "total": 0.0}
    
    allocated = torch.cuda.memory_allocated(device.index) / (1024**3)
    cached = torch.cuda.memory_reserved(device.index) / (1024**3)
    total = torch.cuda.get_device_properties(device.index).total_memory / (1024**3)
    
    return {
        "allocated": allocated,
        "cached": cached,
        "total": total
    }


def log_memory_usage(device: torch.device, stage: str = "") -> None:
    """
    Log current memory usage.
    
    Args:
        device: Target device
        stage: Optional stage description
    """
    if device.type != "cuda":
        return
    
    usage = get_memory_usage(device)
    stage_str = f" [{stage}]" if stage else ""
    
    logging.info(
        f"GPU Memory{stage_str}: "
        f"Allocated={usage['allocated']:.1f}GB, "
        f"Cached={usage['cached']:.1f}GB, "
        f"Total={usage['total']:.1f}GB"
    )
