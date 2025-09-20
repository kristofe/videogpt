# VizDoom Dataset Creation for VideoGPT

This directory contains scripts to create a VizDoom dataset for VideoGPT training. The dataset consists of video sequences recorded from various VizDoom scenarios.

## Overview

VizDoom is a research platform based on the Doom engine, designed for developing AI agents using visual information. This dataset creation process records gameplay from different VizDoom scenarios and converts them into the format required by VideoGPT.

## Features

- **Multiple Scenarios**: Supports various VizDoom scenarios (basic, defend_the_center, defend_the_line, health_gathering)
- **Automatic Recording**: Records gameplay with random actions to create diverse video sequences
- **HDF5 Format**: Converts videos to the HDF5 format required by VideoGPT
- **Class Organization**: Organizes videos by scenario type for classification tasks

## Quick Start

To create the VizDoom dataset:

```bash
sh scripts/preprocess/vizdoom/create_vizdoom_dataset.sh datasets/vizdoom
```

This will create:
- `datasets/vizdoom/train/`: Directory containing training video files organized by class
- `datasets/vizdoom/test/`: Directory containing test video files organized by class
- `datasets/vizdoom/vizdoom.h5`: HDF5 file ready for VideoGPT training

## Prerequisites

### Install VizDoom

First, install VizDoom and its dependencies:

```bash
# Install VizDoom
pip install vizdoom

# On Ubuntu/Debian, you may also need:
sudo apt-get install build-essential libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libopenal-dev libvorbis-dev libflac-dev libxmp-dev libmad0-dev libz-dev cmake

# On macOS:
brew install sdl2 boost cmake
```

### Verify Installation

```python
import vizdoom as vzd
print("VizDoom version:", vzd.__version__)
```

## Dataset Structure

### Video Directory Structure
```
datasets/vizdoom/
├── train/
│   ├── class_0/          # basic scenario
│   │   ├── video_0000.mp4
│   │   ├── video_0001.mp4
│   │   └── ...
│   ├── class_1/          # defend_the_center scenario
│   │   └── ...
│   ├── class_2/          # defend_the_line scenario
│   │   └── ...
│   └── class_3/          # health_gathering scenario
│       └── ...
└── test/
    ├── class_0/
    └── ...
```

### HDF5 Format
The generated `vizdoom.h5` file contains:
- `train_data`: Training video frames (uint8, shape: [N_frames, H, W])
- `train_idx`: Start indices for each video
- `train_labels`: Class labels for each video
- `test_data`: Test video frames
- `test_idx`: Start indices for test videos
- `test_labels`: Test class labels
- `attributes`: Metadata (sequence_length, resolution, class_names, etc.)

## Usage

### Basic Usage

```bash
# Create default dataset (1000 train, 200 test videos)
sh scripts/preprocess/vizdoom/create_vizdoom_dataset.sh datasets/vizdoom
```

### Custom Configuration

```bash
# Generate custom dataset
python scripts/preprocess/vizdoom/generate_vizdoom_videos.py \
    --output_dir datasets/vizdoom_custom \
    --num_train_videos 2000 \
    --num_test_videos 400 \
    --sequence_length 32 \
    --resolution 128 \
    --scenarios basic defend_the_center health_gathering

# Convert to HDF5
python scripts/preprocess/vizdoom/vizdoom_to_hdf5.py \
    --data_dir datasets/vizdoom_custom \
    --output_file datasets/vizdoom_custom/vizdoom.h5 \
    --sequence_length 32 \
    --resolution 128
```

## Parameters

### Video Generation (`generate_vizdoom_videos.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_dir` | Required | Output directory for videos |
| `--num_train_videos` | 1000 | Number of training videos |
| `--num_test_videos` | 200 | Number of test videos |
| `--sequence_length` | 16 | Number of frames per video |
| `--resolution` | 64 | Video resolution (square) |
| `--scenarios` | basic, defend_the_center, defend_the_line, health_gathering | VizDoom scenarios to use |

### HDF5 Conversion (`vizdoom_to_hdf5.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Directory containing train/test video folders |
| `--output_file` | Required | Output HDF5 file path |
| `--sequence_length` | 16 | Number of frames per video |
| `--resolution` | 64 | Video resolution |

## Available Scenarios

The script supports several built-in VizDoom scenarios:

1. **basic**: Simple scenario with basic movement and shooting
2. **defend_the_center**: Defend a central position from enemies
3. **defend_the_line**: Defend a line of positions
4. **health_gathering**: Collect health items while avoiding enemies

You can add custom scenarios by:
1. Creating custom `.wad` files with Doom map editors
2. Modifying the `create_doom_game()` function to load custom scenarios

## Training with VizDoom Dataset

Once you have the HDF5 dataset, you can train VideoGPT:

```bash
# Train VQ-VAE
python scripts/train_vqvae.py \
    --data_path datasets/vizdoom/vizdoom.h5 \
    --resolution 64 \
    --sequence_length 16 \
    --batch_size 16 \
    --gpus 1

# Train VideoGPT
python scripts/train_videogpt.py \
    --data_path datasets/vizdoom/vizdoom.h5 \
    --vqvae path/to/vqvae_checkpoint.ckpt \
    --resolution 64 \
    --sequence_length 16 \
    --batch_size 8 \
    --gpus 1
```

## Troubleshooting

### Common Issues

1. **VizDoom Installation Issues**:
   - Make sure you have all system dependencies installed
   - On Windows, you may need Visual Studio Build Tools
   - On Linux, ensure you have the development packages

2. **Memory Issues**:
   - Reduce `num_train_videos` and `num_test_videos` for smaller datasets
   - Use lower resolution (e.g., 32x32 instead of 64x64)

3. **Video Generation Errors**:
   - Some scenarios may not be available on all systems
   - Try using only the "basic" scenario if others fail
   - Check that VizDoom scenarios are properly installed

4. **Performance Issues**:
   - The video generation process can be slow
   - Consider running on a machine with good CPU performance
   - You can run multiple instances in parallel for different scenarios

### Debug Mode

To debug video generation issues:

```python
import vizdoom as vzd

# Test basic scenario
game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/basic.cfg")
game.set_window_visible(True)  # Make window visible for debugging
game.init()
```

## Customization

### Adding New Scenarios

1. Create a custom scenario file (`.wad` and `.cfg`)
2. Place it in the VizDoom scenarios directory
3. Modify the `scenarios` list in `generate_vizdoom_videos.py`

### Custom Action Spaces

Modify the `generate_random_action()` function to create different action patterns:

```python
def generate_random_action():
    # Example: More structured action patterns
    actions = [
        [1, 0, 0, 0, 0, 0, 0],  # Move left
        [0, 1, 0, 0, 0, 0, 0],  # Move right
        [0, 0, 1, 0, 0, 0, 0],  # Attack
        [0, 0, 0, 1, 0, 0, 0],  # Move forward
    ]
    return random.choice(actions)
```

## File Structure

```
scripts/preprocess/vizdoom/
├── create_vizdoom_dataset.sh    # Main dataset creation script
├── generate_vizdoom_videos.py   # Video generation script
├── vizdoom_to_hdf5.py          # HDF5 conversion script
└── README.md                   # This documentation
```

## Requirements

- Python 3.6+
- VizDoom
- OpenCV (cv2)
- NumPy
- H5py
- TQDM
- Pathlib

Install with:
```bash
pip install vizdoom opencv-python numpy h5py tqdm
```

## Citation

If you use this VizDoom dataset creation code in your research, please cite:

```bibtex
@misc{yan2021videogpt,
    title={VideoGPT: Video Generation using VQ-VAE and Transformers}, 
    author={Wilson Yan and Yunzhi Zhang and Pieter Abbeel and Aravind Srinivas},
    year={2021},
    eprint={2104.10157},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

And the VizDoom platform:

```bibtex
@inproceedings{kempka2016vizdoom,
    title={Vizdoom: A doom-based ai research platform for visual reinforcement learning},
    author={Kempka, Michał and Wydmuch, Marek and Runc, Grzegorz and Toczek, Jakub and Jaśkowski, Wojciech},
    booktitle={2016 IEEE Conference on Computational Intelligence and Games (CIG)},
    pages={1--8},
    year={2016},
    organization={IEEE}
}
```
