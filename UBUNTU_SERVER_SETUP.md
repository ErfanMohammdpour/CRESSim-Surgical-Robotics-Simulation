# Ubuntu Server Deployment Guide

## 🚀 CRESSim Suction RL - Ubuntu GPU Server Setup

This guide will help you deploy the surgical robotics project on an Ubuntu server with GPU support.

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ (recommended: Ubuntu 22.04 LTS)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- **RAM**: 16GB+ (32GB recommended for training)
- **Storage**: 50GB+ free space
- **Python**: 3.10+ (3.11 recommended)

### GPU Requirements
- NVIDIA GPU with Compute Capability 6.0+
- CUDA 11.8+ or 12.0+
- cuDNN 8.0+

## 🔧 Step 1: System Setup

### Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential
```

### Install NVIDIA Drivers
```bash
# Check GPU
nvidia-smi

# If no output, install drivers
sudo apt install -y nvidia-driver-525  # or latest version
sudo reboot
```

### Install CUDA Toolkit
```bash
# Download CUDA 12.0 (adjust version as needed)
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run

# Install CUDA
sudo sh cuda_12.0.0_525.60.13_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Install cuDNN
```bash
# Download cuDNN (requires NVIDIA account)
# Go to: https://developer.nvidia.com/cudnn
# Download cuDNN v8.9.0 for CUDA 12.0

# Install cuDNN
tar -xvf cudnn-linux-x86_64-8.9.0.131_cuda12-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## 🐍 Step 2: Python Environment Setup

### Install Python 3.11
```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo apt install -y python3-pip
```

### Create Virtual Environment
```bash
# Clone your project
git clone <your-repo-url> surgical-robotics
cd surgical-robotics

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Install PyTorch with CUDA
```bash
# Install PyTorch with CUDA 12.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## 📦 Step 3: Install Project Dependencies

### Install Requirements
```bash
# Install all dependencies
pip install -r requirements.txt

# Install additional server dependencies
pip install screen tmux htop
```

### Verify Installation
```bash
# Test imports
python -c "import torch; import stable_baselines3; import gymnasium; print('All imports successful')"

# Test CUDA
python -c "import torch; x = torch.randn(1000, 1000).cuda(); print('CUDA tensor creation successful')"
```

## 🚀 Step 4: Project Setup

### Bootstrap Project
```bash
# Run bootstrap (this will download datasets and setup everything)
python manage.py bootstrap

# If Unity build is not available, use mock mode
python manage.py demos --mock
```

### Verify Setup
```bash
# Run tests
python -m pytest tests/ -v

# Check GPU utilization
nvidia-smi
```

## 🎯 Step 5: Running Training

### Option 1: Interactive Training
```bash
# Activate environment
source .venv/bin/activate

# Generate demos (mock mode)
python manage.py demos --mock --num-episodes 10

# Train IL model
python manage.py train-il

# Train RL model (mock mode)
python manage.py train-rl --mock --timesteps 100000
```

### Option 2: Background Training with Screen
```bash
# Start screen session
screen -S training

# Activate environment
source .venv/bin/activate

# Run training
python manage.py train-rl --mock --timesteps 1000000

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### Option 3: Background Training with tmux
```bash
# Start tmux session
tmux new-session -d -s training

# Run commands in tmux
tmux send-keys -t training "source .venv/bin/activate" Enter
tmux send-keys -t training "python manage.py train-rl --mock --timesteps 1000000" Enter

# Attach to session
tmux attach-session -t training
```

## 📊 Step 6: Monitoring Training

### GPU Monitoring
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use htop for system monitoring
htop
```

### Log Monitoring
```bash
# Monitor training logs
tail -f data/logs/training_*.log

# Monitor all logs
tail -f data/logs/*.log
```

### TensorBoard (if available)
```bash
# Start TensorBoard
tensorboard --logdir=data/logs/tensorboard --host=0.0.0.0 --port=6006

# Access via browser: http://your-server-ip:6006
```

## 🔧 Step 7: Server Optimization

### GPU Memory Optimization
```bash
# Set GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU

# Add to ~/.bashrc for persistence
echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
```

### System Optimization
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize swap (if needed)
sudo swapon --show
```

## 🐳 Step 8: Docker Deployment (Alternative)

### Create Dockerfile
```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project
COPY . .

# Create virtual environment
RUN python3.11 -m venv .venv
RUN .venv/bin/pip install --upgrade pip

# Install PyTorch with CUDA
RUN .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

# Install requirements
RUN .venv/bin/pip install -r requirements.txt

# Set environment
ENV PATH="/app/.venv/bin:$PATH"

# Run bootstrap
RUN python manage.py bootstrap

CMD ["python", "manage.py", "train-rl", "--mock", "--timesteps", "1000000"]
```

### Build and Run Docker
```bash
# Build image
docker build -t surgical-robotics .

# Run with GPU support
docker run --gpus all -it surgical-robotics
```

## 📈 Step 9: Performance Tuning

### GPU-Specific Optimizations
```bash
# Set optimal GPU settings
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -ac 1215,1410  # Set memory and graphics clocks

# Monitor GPU utilization
nvidia-smi -l 1  # Update every second
```

### Training Configuration
```yaml
# configs/train.yaml - GPU optimized settings
device: cuda
batch_size: 128  # Increase for GPU
n_steps: 4096    # Increase for GPU
learning_rate: 1e-4
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
```bash
# Reduce batch size
python manage.py train-rl --mock --timesteps 100000 --batch-size 32

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

2. **Permission Issues**
```bash
# Fix permissions
sudo chown -R $USER:$USER .
chmod +x manage.py
```

3. **Missing Dependencies**
```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt
```

4. **GPU Not Detected**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## 📋 Quick Start Commands

```bash
# Complete setup in one go
git clone <your-repo> && cd surgical-robotics
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
pip install -r requirements.txt
python manage.py bootstrap
python manage.py demos --mock
python manage.py train-il
python manage.py train-rl --mock --timesteps 100000
```

## 🎯 Production Deployment

### Systemd Service (for production)
```bash
# Create service file
sudo tee /etc/systemd/system/surgical-robotics.service > /dev/null <<EOF
[Unit]
Description=Surgical Robotics Training
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/surgical-robotics
Environment=PATH=/home/ubuntu/surgical-robotics/.venv/bin
ExecStart=/home/ubuntu/surgical-robotics/.venv/bin/python manage.py train-rl --mock --timesteps 1000000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable surgical-robotics
sudo systemctl start surgical-robotics
sudo systemctl status surgical-robotics
```

This guide provides a complete setup for running your surgical robotics project on an Ubuntu server with GPU support. The system is optimized for training and includes monitoring, troubleshooting, and production deployment options.


