# CRESSim Suction RL

A production-quality repository for training vision-based reinforcement learning policies for autonomous endoscopic suction inside CRESSim (Unity + ML-Agents). This project implements a complete IL→RL pipeline with visual safety shields and domain randomization.

## Project Overview

This repository trains a vision-based RL policy for autonomous endoscopic suction using CRESSim, a Unity-based surgical simulation environment. The system combines imitation learning from imperfect demonstrations with PPO reinforcement learning, enhanced by a visual safety shield that prevents dangerous actions. The pipeline includes domain randomization, curriculum learning, and comprehensive evaluation metrics.

Key features:
- **Visual Safety Shield**: Tiny segmentation network prevents dangerous actions
- **IL→RL Pipeline**: Behavior cloning warm-start followed by PPO fine-tuning
- **Domain Randomization**: Lighting, textures, camera noise, and fluid properties
- **Windows-First**: Complete CLI with no Makefiles required
- **Reproducible**: Single Python command bootstrap and training

## Requirements (Windows)

- **Python 3.10+** with pip
- **CUDA** (optional, for GPU acceleration with PyTorch)
- **Unity Hub/Editor 2022.3 LTS** (for building the environment)
- **Git** (for cloning CRESSim)

## Quick Start (from scratch)

### 1. Setup Environment

Create and activate a virtual environment (recommended):

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Bootstrap Everything

Download datasets, clone simulator, and run tests:

```powershell
python manage.py bootstrap
```

This will:
- Install Python dependencies
- Download Kvasir-SEG dataset to `data/kvasir_seg/`
- Clone CRESSim repository to `sim/CRESSim/`
- Run CPU-only tests

### 3. Build Unity Environment

**Unity Build Steps:**

1. Open Unity Hub
2. Open the project at `sim/CRESSim/`
3. Open the scene: `Assets/Scenes/SuctionEnv.unity`
4. Go to File → Build Settings
5. Select "PC, Mac & Linux Standalone" → "Windows"
6. Click "Build" and save as `sim/CRESSim/Build/SuctionEnv.exe`

Confirm the executable appears at the path in `configs/paths.yaml`.

### 4. Generate Demos

Create imperfect scripted demonstrations:

```powershell
python manage.py demos
```

**What gets created:**
- Raw demos saved to `data/demos/raw_*.npz`
- GMM-filtered weights saved to `data/demos/weights.npz`
- Demo statistics logged to `data/logs/demos.log`

### 5. Train - Imitation Learning

Run weighted behavior cloning:

```powershell
python manage.py train-il
```

**What to expect:**
- Training logs in `data/logs/il_training.log`
- Checkpoints saved to `data/checkpoints/il_*.pth`
- TensorBoard logs in `data/logs/tensorboard/`

### 6. Train - Reinforcement Learning

Run PPO with safety shield:

```powershell
python manage.py train-rl
```

**Notes:**
- Requires Unity build to be present
- Uses curriculum learning that increases difficulty
- Safety shield prevents dangerous actions
- Checkpoints selected using safety-first Pareto optimization

### 7. Evaluate

Run multiple rollouts and generate metrics:

```powershell
python manage.py eval
```

**Outputs:**
- `data/videos/eval_*.mp4` - Video recordings
- `data/videos/metrics.json` - Quantitative metrics
- `data/videos/plots/` - Performance plots

### 8. Ablations / Benchmark

Run comparison experiments:

```powershell
python manage.py bench
```

**Comparisons:**
- RL-only vs IL→RL vs IL→RL+Safety
- Summary table: `data/benchmarks/comparison.csv`
- Comparison plot: `data/benchmarks/comparison.png`

## Configuration

### `configs/env.yaml`
Environment settings including image size, action scaling, reward weights, and domain randomization ranges.

### `configs/train.yaml`
Training hyperparameters for PPO (learning rate, batch size, etc.) and model architecture settings.

### `configs/safety.yaml`
Safety shield configuration including distance thresholds, segmentation settings, and action projection parameters.

### `configs/demos.yaml`
Demo generation settings including noise levels, acceptance thresholds, and GMM components.

### `configs/paths.yaml`
File paths for datasets, simulator, Unity build, and output directories.

## Troubleshooting

### Missing Unity Build
If the Unity build is missing, you can still:
- Run `python manage.py demos` (creates mock demos)
- Run `python manage.py train-il` (uses mock demos)
- Run tests and dataset validation

### GPU vs CPU Performance
- GPU recommended for training (3-5x faster)
- CPU sufficient for evaluation and demos
- Set `CUDA_VISIBLE_DEVICES=""` to force CPU usage

### Dataset Issues
- Check `data/kvasir_seg/` contains >1000 images
- Verify `data/endoscapes/` structure if using optional dataset
- Run `python manage.py bootstrap` to re-download

## Licenses & Citations

### Kvasir-SEG Dataset
- **URL**: https://datasets.simula.no/downloads/kvasir-seg.zip
- **License**: CC BY 4.0
- **Citation**: Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MICCAI 2020

### CRESSim Simulator
- **Repository**: https://github.com/tbs-ualberta/CRESSim
- **License**: BSD-2-Clause
- **Citation**: Ou et al., "Learning autonomous surgical irrigation and suction with the Da Vinci Research Kit using reinforcement learning", arXiv 2024

## Project Structure

```
cressim-suction-rl/
├── manage.py                 # Main CLI
├── requirements.txt          # Python dependencies
├── configs/                  # Configuration files
├── src/                      # Source code
│   ├── envs/                # Environment wrappers
│   ├── vision/              # Vision components
│   ├── il/                  # Imitation learning
│   ├── rl/                  # Reinforcement learning
│   ├── eval/                # Evaluation tools
│   └── utils/               # Utilities
├── sim/CRESSim/             # Unity simulator (cloned)
├── data/                    # Datasets and outputs
└── tests/                   # Unit tests

```
