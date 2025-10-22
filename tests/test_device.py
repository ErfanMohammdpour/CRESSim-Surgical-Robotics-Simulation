"""
Tests for device utilities and GPU support.
"""

import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from src.utils.device import (
    get_device, cuda_info_string, amp_enabled, setup_cudnn,
    get_device_from_config, log_device_info, optimize_for_gpu,
    create_optimizer, create_grad_scaler, get_dataloader_kwargs,
    handle_oom_error, warmup_model, log_memory_usage, get_memory_usage
)


class TestDeviceSelection:
    """Test device selection functionality."""
    
    def test_get_device_cpu_preference(self):
        """Test device selection with CPU preference."""
        device = get_device(prefer_cuda=False)
        assert device.type == "cpu"
    
    @patch('torch.cuda.is_available')
    def test_get_device_cuda_unavailable(self, mock_cuda_available):
        """Test device selection when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        device = get_device(prefer_cuda=True)
        assert device.type == "cpu"
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_get_device_cuda_available(self, mock_device_count, mock_cuda_available):
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        with patch('torch.device') as mock_device:
            mock_device.return_value = torch.device('cuda:0')
            device = get_device(prefer_cuda=True)
            assert device.type == "cuda"
    
    def test_get_device_from_config_cpu(self):
        """Test device selection from config with CPU."""
        config = {"device": "cpu"}
        device = get_device_from_config(config)
        assert device.type == "cpu"
    
    def test_get_device_from_config_auto(self):
        """Test device selection from config with auto."""
        config = {"device": "auto"}
        device = get_device_from_config(config)
        # Should return a valid device (CPU or CUDA depending on availability)
        assert device.type in ["cpu", "cuda"]
    
    @patch('torch.cuda.is_available')
    def test_get_device_from_config_cuda(self, mock_cuda_available):
        """Test device selection from config with CUDA."""
        mock_cuda_available.return_value = True
        config = {"device": "cuda"}
        
        with patch('torch.device') as mock_device:
            mock_device.return_value = torch.device('cuda:0')
            device = get_device_from_config(config)
            assert device.type == "cuda"


class TestAMPConfiguration:
    """Test AMP configuration functionality."""
    
    def test_amp_enabled_cpu_device(self):
        """Test AMP disabled for CPU device."""
        config = {"device": "cpu", "amp": True}
        assert not amp_enabled(config)
    
    def test_amp_enabled_amp_disabled(self):
        """Test AMP disabled when config says so."""
        config = {"device": "cuda", "amp": False}
        assert not amp_enabled(config)
    
    @patch('torch.cuda.is_available')
    def test_amp_enabled_cuda_available(self, mock_cuda_available):
        """Test AMP enabled when CUDA available and config allows."""
        mock_cuda_available.return_value = True
        config = {"device": "cuda", "amp": True}
        assert amp_enabled(config)
    
    @patch('torch.cuda.is_available')
    def test_amp_enabled_auto_device(self, mock_cuda_available):
        """Test AMP enabled with auto device when CUDA available."""
        mock_cuda_available.return_value = True
        config = {"device": "auto", "amp": True}
        assert amp_enabled(config)


class TestCUDNNConfiguration:
    """Test cuDNN configuration functionality."""
    
    @patch('torch.backends.cudnn.benchmark', False)
    @patch('torch.backends.cudnn.deterministic', True)
    def test_setup_cudnn_cpu(self):
        """Test cuDNN setup with CPU (should not modify settings)."""
        config = {"device": "cpu", "cudnn_benchmark": True, "deterministic": False}
        setup_cudnn(config)
        # Should not modify cuDNN settings for CPU
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.cudnn.benchmark')
    @patch('torch.backends.cudnn.deterministic')
    @patch('torch.set_float32_matmul_precision')
    def test_setup_cudnn_cuda(self, mock_set_precision, mock_deterministic, 
                             mock_benchmark, mock_cuda_available):
        """Test cuDNN setup with CUDA."""
        mock_cuda_available.return_value = True
        config = {
            "device": "cuda",
            "cudnn_benchmark": True,
            "deterministic": False,
            "float32_matmul_precision": "high"
        }
        
        setup_cudnn(config)
        
        # Should set cuDNN settings
        mock_benchmark.assert_called_with(True)
        mock_deterministic.assert_called_with(False)
        mock_set_precision.assert_called_with("high")


class TestModelOptimization:
    """Test model optimization functionality."""
    
    def test_optimize_for_gpu_cpu(self):
        """Test model optimization for CPU."""
        model = torch.nn.Linear(10, 1)
        config = {"device": "cpu", "compile": False}
        
        optimized_model = optimize_for_gpu(model, config)
        assert optimized_model is model  # Should return same model for CPU
    
    @patch('torch.cuda.is_available')
    def test_optimize_for_gpu_cuda_no_compile(self, mock_cuda_available):
        """Test model optimization for CUDA without compilation."""
        mock_cuda_available.return_value = True
        model = torch.nn.Linear(10, 1)
        config = {"device": "cuda", "compile": False}
        
        with patch('torch.device') as mock_device:
            mock_device.return_value = torch.device('cuda:0')
            optimized_model = optimize_for_gpu(model, config)
            assert optimized_model is model  # Should return same model
    
    @patch('torch.cuda.is_available')
    def test_optimize_for_gpu_cuda_with_compile(self, mock_cuda_available):
        """Test model optimization for CUDA with compilation."""
        mock_cuda_available.return_value = True
        model = torch.nn.Linear(10, 1)
        config = {"device": "cuda", "compile": True}
        
        with patch('torch.device') as mock_device, \
             patch('torch.compile') as mock_compile:
            mock_device.return_value = torch.device('cuda:0')
            mock_compile.return_value = model
            optimized_model = optimize_for_gpu(model, config)
            mock_compile.assert_called_once_with(model)


class TestOptimizerCreation:
    """Test optimizer creation functionality."""
    
    def test_create_optimizer_cpu(self):
        """Test optimizer creation for CPU."""
        model = torch.nn.Linear(10, 1)
        config = {"device": "cpu", "learning_rate": 1e-3, "weight_decay": 1e-4}
        
        optimizer = create_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['weight_decay'] == 1e-4


class TestGradScalerCreation:
    """Test gradient scaler creation functionality."""
    
    def test_create_grad_scaler_cpu(self):
        """Test gradient scaler creation for CPU."""
        config = {"device": "cpu", "amp": True}
        scaler = create_grad_scaler(config)
        assert scaler is None  # Should return None for CPU
    
    @patch('torch.cuda.is_available')
    def test_create_grad_scaler_cuda(self, mock_cuda_available):
        """Test gradient scaler creation for CUDA."""
        mock_cuda_available.return_value = True
        config = {"device": "cuda", "amp": True}
        
        scaler = create_grad_scaler(config)
        assert scaler is not None
        assert isinstance(scaler, torch.cuda.amp.GradScaler)


class TestDataLoaderKwargs:
    """Test DataLoader kwargs generation."""
    
    def test_get_dataloader_kwargs_cpu(self):
        """Test DataLoader kwargs for CPU."""
        config = {"device": "cpu", "num_workers": 4}
        kwargs = get_dataloader_kwargs(config)
        
        assert kwargs["pin_memory"] is False
        assert kwargs["num_workers"] == 4
        assert kwargs["persistent_workers"] is True
    
    @patch('torch.cuda.is_available')
    def test_get_dataloader_kwargs_cuda(self, mock_cuda_available):
        """Test DataLoader kwargs for CUDA."""
        mock_cuda_available.return_value = True
        config = {"device": "cuda", "num_workers": 4}
        
        with patch('torch.device') as mock_device:
            mock_device.return_value = torch.device('cuda:0')
            kwargs = get_dataloader_kwargs(config)
            
            assert kwargs["pin_memory"] is True
            assert kwargs["num_workers"] == 4
            assert kwargs["persistent_workers"] is True


class TestOOMHandling:
    """Test out-of-memory error handling."""
    
    def test_handle_oom_error_non_oom(self):
        """Test handling non-OOM errors."""
        config = {"batch_size": 64}
        error = RuntimeError("Some other error")
        
        with pytest.raises(RuntimeError):
            handle_oom_error(error, config)
    
    def test_handle_oom_error_reduce_batch_size(self):
        """Test OOM handling reduces batch size."""
        config = {"batch_size": 64}
        error = RuntimeError("CUDA out of memory")
        
        updated_config = handle_oom_error(error, config)
        assert updated_config["batch_size"] == 32  # Should be halved
    
    def test_handle_oom_error_reduce_gradient_accumulation(self):
        """Test OOM handling reduces gradient accumulation."""
        config = {"batch_size": 64, "gradient_accumulation_steps": 8}
        error = RuntimeError("CUDA out of memory")
        
        updated_config = handle_oom_error(error, config)
        assert updated_config["gradient_accumulation_steps"] == 4  # Should be halved


class TestMemoryUtilities:
    """Test memory utility functions."""
    
    def test_get_memory_usage_cpu(self):
        """Test memory usage for CPU device."""
        device = torch.device("cpu")
        usage = get_memory_usage(device)
        
        assert usage["allocated"] == 0.0
        assert usage["cached"] == 0.0
        assert usage["total"] == 0.0
    
    @patch('torch.cuda.is_available')
    def test_get_memory_usage_cuda(self, mock_cuda_available):
        """Test memory usage for CUDA device."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.memory_allocated') as mock_allocated, \
             patch('torch.cuda.memory_reserved') as mock_reserved, \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_allocated.return_value = 1024**3  # 1GB
            mock_reserved.return_value = 2 * 1024**3  # 2GB
            mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
            
            device = torch.device("cuda:0")
            usage = get_memory_usage(device)
            
            assert usage["allocated"] == 1.0
            assert usage["cached"] == 2.0
            assert usage["total"] == 8.0


class TestWarmupModel:
    """Test model warmup functionality."""
    
    def test_warmup_model_cpu(self):
        """Test model warmup for CPU (should do nothing)."""
        model = torch.nn.Linear(10, 1)
        config = {"device": "cpu"}
        
        # Should not raise any errors
        warmup_model(model, config, (1, 10))
    
    @patch('torch.cuda.is_available')
    def test_warmup_model_cuda(self, mock_cuda_available):
        """Test model warmup for CUDA."""
        mock_cuda_available.return_value = True
        model = torch.nn.Linear(10, 1)
        config = {"device": "cuda", "amp": False}
        
        with patch('torch.device') as mock_device:
            mock_device.return_value = torch.device('cuda:0')
            # Should not raise any errors
            warmup_model(model, config, (1, 10))


class TestCUDAInfoString:
    """Test CUDA information string generation."""
    
    @patch('torch.cuda.is_available')
    def test_cuda_info_string_unavailable(self, mock_cuda_available):
        """Test CUDA info string when CUDA unavailable."""
        mock_cuda_available.return_value = False
        info = cuda_info_string()
        assert "CUDA not available" in info
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.current_device')
    def test_cuda_info_string_available(self, mock_current_device, mock_device_count, mock_cuda_available):
        """Test CUDA info string when CUDA available."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_current_device.return_value = 0
        
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value = MagicMock()
            mock_props.return_value.name = "Test GPU"
            mock_props.return_value.major = 8
            mock_props.return_value.minor = 6
            mock_props.return_value.total_memory = 8 * 1024**3
            
            info = cuda_info_string()
            assert "CUDA available: True" in info
            assert "Device count: 1" in info
            assert "Test GPU" in info


# Integration tests
class TestDeviceIntegration:
    """Integration tests for device functionality."""
    
    def test_device_config_override_priority(self):
        """Test that CLI overrides have priority over config."""
        config = {"device": "cpu", "amp": True, "compile": False}
        
        # Simulate CLI overrides
        config["device"] = "cuda"
        config["amp"] = False
        config["compile"] = True
        
        assert config["device"] == "cuda"
        assert config["amp"] is False
        assert config["compile"] is True
    
    @patch('torch.cuda.is_available')
    def test_end_to_end_device_selection(self, mock_cuda_available):
        """Test end-to-end device selection workflow."""
        mock_cuda_available.return_value = True
        
        # Test config
        config = {
            "device": "auto",
            "amp": True,
            "compile": False,
            "cudnn_benchmark": True,
            "deterministic": False
        }
        
        # Get device
        device = get_device_from_config(config)
        
        # Check AMP
        amp_on = amp_enabled(config)
        
        # Setup cuDNN
        setup_cudnn(config)
        
        # All should work without errors
        assert device.type in ["cpu", "cuda"]
        assert isinstance(amp_on, bool)
