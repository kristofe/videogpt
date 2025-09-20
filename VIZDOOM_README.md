# VizDoom Dataset for VideoGPT

This guide explains how to obtain and create VizDoom datasets for training VideoGPT models. VizDoom is a research platform based on the Doom engine, perfect for generating diverse video sequences for video generation tasks.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Creation](#dataset-creation)
- [Using Existing Datasets](#using-existing-datasets)
- [Training VideoGPT](#training-videogpt)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Citation](#citation)

## Overview

VizDoom provides an excellent source of video data for VideoGPT training because it offers:

- **Diverse Scenarios**: Multiple game scenarios with different visual styles and challenges
- **Controllable Environment**: Can generate consistent, reproducible video sequences
- **Rich Visual Content**: Complex 3D environments with textures, lighting, and movement
- **Action Sequences**: Natural video sequences with agent actions and environmental responses
- **Action Recording**: Captures the exact actions taken during gameplay for world model training
- **Scalable Generation**: Can generate unlimited amounts of training data

### Available Scenarios

The VizDoom dataset creation supports several built-in scenarios:

1. **basic**: Simple scenario with basic movement and shooting
2. **defend_the_center**: Defend a central position from enemies
3. **defend_the_line**: Defend a line of positions
4. **health_gathering**: Collect health items while avoiding enemies

## Quick Start

### Option 1: Generate Your Own Dataset (Recommended)

```bash
# 1. Install VizDoom
pip install vizdoom

# 2. Create the dataset
sh scripts/preprocess/vizdoom/create_vizdoom_dataset.sh datasets/vizdoom

# 3. Train VideoGPT
python scripts/train_vqvae.py --data_path datasets/vizdoom/vizdoom.h5 --resolution 64 --sequence_length 16 --batch_size 16 --gpus 1
python scripts/train_videogpt.py --data_path datasets/vizdoom/vizdoom.h5 --vqvae path/to/vqvae_checkpoint.ckpt --resolution 64 --sequence_length 16 --batch_size 8 --gpus 1
```

### Option 2: Use Existing Datasets

```bash
# Download Apple's VizDoom dataset
python scripts/download_vizdoom.py

# Or use Demo2Program datasets
# See: https://github.com/shaohua0116/demo2program
```

## Installation

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libopenal-dev libvorbis-dev libflac-dev libxmp-dev libmad0-dev libz-dev cmake
```

**macOS:**
```bash
brew install sdl2 boost cmake
```

**Windows:**
- Install Visual Studio Build Tools
- Install CMake
- Install SDL2 development libraries

### Python Dependencies

```bash
# Install VizDoom
pip install vizdoom

# Install additional dependencies
pip install opencv-python numpy h5py tqdm
```

### Verify Installation

```python
import vizdoom as vzd
print("VizDoom version:", vzd.__version__)

# Test basic functionality
game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/basic.cfg")
game.set_window_visible(False)
game.init()
print("VizDoom installation successful!")
game.close()
```

## Dataset Creation

### Automatic Dataset Generation

The easiest way to create a VizDoom dataset is using the provided script:

```bash
sh scripts/preprocess/vizdoom/create_vizdoom_dataset.sh datasets/vizdoom
```

This creates:
- **1,000 training videos** (250 per scenario class)
- **200 test videos** (50 per scenario class)
- **16-frame sequences** at 64x64 resolution
- **Action sequences** recorded for each video frame
- **HDF5 format** ready for VideoGPT training and world model development

### Manual Dataset Generation

For more control over the dataset creation:

```bash
# Step 1: Generate videos
python scripts/preprocess/vizdoom/generate_vizdoom_videos.py \
    --output_dir datasets/vizdoom \
    --num_train_videos 1000 \
    --num_test_videos 200 \
    --sequence_length 16 \
    --resolution 64 \
    --scenarios basic defend_the_center defend_the_line health_gathering

# Step 2: Convert to HDF5
python scripts/preprocess/vizdoom/vizdoom_to_hdf5.py \
    --data_dir datasets/vizdoom \
    --output_file datasets/vizdoom/vizdoom.h5 \
    --sequence_length 16 \
    --resolution 64
```

### Custom Dataset Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_train_videos` | 1000 | Number of training videos |
| `--num_test_videos` | 200 | Number of test videos |
| `--sequence_length` | 16 | Frames per video |
| `--resolution` | 64 | Video resolution (square) |
| `--scenarios` | basic, defend_the_center, defend_the_line, health_gathering | Scenarios to use |

## Using Existing Datasets

### Apple's VizDoom Dataset

Apple provides a pre-compiled VizDoom dataset with camera trajectories:

```bash
# Download Apple's dataset
python scripts/download_vizdoom.py
```

**Dataset Details:**
- Size: ~2.4 GB
- Format: RGB and depth frames with camera parameters
- Source: [Apple ML-GSN Repository](https://github.com/apple/ml-gsn)

### Demo2Program Datasets

Two VizDoom datasets for program synthesis research:

1. **vizdoom_shorter**: Short demonstration videos
2. **vizdoom_full**: Full demonstration videos

**Download:**
- Repository: [Demo2Program](https://github.com/shaohua0116/demo2program)
- Usage: Follow their documentation for dataset preparation

### Converting Existing Datasets

If you have existing VizDoom data, you can convert it to VideoGPT format:

```bash
# Convert existing video files to HDF5
python scripts/preprocess/vizdoom/vizdoom_to_hdf5.py \
    --data_dir path/to/your/videos \
    --output_file datasets/vizdoom_custom.h5 \
    --sequence_length 16 \
    --resolution 64
```

## Using Action Data for World Model Training

The VizDoom dataset includes synchronized action sequences that are perfect for training world models. Here's how to use them:

### Loading Action Data

```python
import h5py
import numpy as np

# Load the dataset
with h5py.File('datasets/vizdoom/vizdoom.h5', 'r') as f:
    # Get video frames and actions for first video
    start_idx = f['train_idx'][0]
    end_idx = f['train_idx'][1] if len(f['train_idx']) > 1 else start_idx + 16
    
    frames = f['train_data'][start_idx:end_idx]  # [16, 64, 64]
    actions = f['train_actions'][start_idx:end_idx]  # [16, 7]
    
    # Get metadata
    action_names = [name.decode('utf-8') for name in f.attrs['action_names']]
    print(f"Action names: {action_names}")
    print(f"Frame shape: {frames.shape}")
    print(f"Action shape: {actions.shape}")
```

### World Model Training Example

```python
# Example: Train a simple world model
def train_world_model(frames, actions):
    # frames: [batch, seq_len, height, width]
    # actions: [batch, seq_len, action_dim]
    
    # Encode current frame and action
    current_frame = frames[:, :-1]  # [batch, seq_len-1, h, w]
    next_frame = frames[:, 1:]      # [batch, seq_len-1, h, w]
    action = actions[:, :-1]        # [batch, seq_len-1, action_dim]
    
    # Your world model architecture here
    # This is just a placeholder
    predicted_next_frame = your_world_model(current_frame, action)
    
    # Loss between predicted and actual next frame
    loss = mse_loss(predicted_next_frame, next_frame)
    return loss
```

### Action-Controlled Video Generation

```python
# Generate videos by providing action sequences
def generate_video_with_actions(model, initial_frame, action_sequence):
    # initial_frame: [1, 1, height, width]
    # action_sequence: [seq_len, action_dim]
    
    frames = [initial_frame]
    current_frame = initial_frame
    
    for action in action_sequence:
        # Predict next frame given current frame and action
        next_frame = model.predict(current_frame, action.unsqueeze(0))
        frames.append(next_frame)
        current_frame = next_frame
    
    return torch.cat(frames, dim=1)  # [1, seq_len+1, h, w]
```

## Training VideoGPT

### Step 1: Train VQ-VAE

```bash
python scripts/train_vqvae.py \
    --data_path datasets/vizdoom/vizdoom.h5 \
    --resolution 64 \
    --sequence_length 16 \
    --embedding_dim 256 \
    --n_codes 512 \
    --n_hiddens 128 \
    --n_res_layers 2 \
    --downsample 2 4 4 \
    --batch_size 16 \
    --gpus 1
```

### Step 2: Train VideoGPT

```bash
python scripts/train_videogpt.py \
    --data_path datasets/vizdoom/vizdoom.h5 \
    --vqvae path/to/vqvae_checkpoint.ckpt \
    --resolution 64 \
    --sequence_length 16 \
    --hidden_dim 256 \
    --heads 4 \
    --layers 6 \
    --batch_size 8 \
    --gpus 1
```

### Step 3: Generate Videos

```bash
python scripts/sample_videogpt.py \
    --data_path datasets/vizdoom/vizdoom.h5 \
    --vqvae path/to/vqvae_checkpoint.ckpt \
    --videogpt path/to/videogpt_checkpoint.ckpt \
    --resolution 64 \
    --sequence_length 16
```

## Customization

### Adding New Scenarios

1. **Create Custom Scenario Files:**
   - Use Doom map editors (SLADE, Doom Builder) to create `.wad` files
   - Create corresponding `.cfg` configuration files
   - Place files in VizDoom scenarios directory

2. **Modify Generation Script:**
   ```python
   # In generate_vizdoom_videos.py
   scenarios = ["basic", "defend_the_center", "your_custom_scenario"]
   ```

### Custom Action Patterns

Modify the action generation for different gameplay styles:

```python
def generate_random_action():
    # Example: More structured action patterns
    action_patterns = [
        [1, 0, 0, 0, 0, 0, 0],  # Move left
        [0, 1, 0, 0, 0, 0, 0],  # Move right
        [0, 0, 1, 0, 0, 0, 0],  # Attack
        [0, 0, 0, 1, 0, 0, 0],  # Move forward
        [0, 0, 0, 0, 1, 0, 0],  # Move backward
        [0, 0, 0, 0, 0, 1, 0],  # Turn left
        [0, 0, 0, 0, 0, 0, 1],  # Turn right
    ]
    return random.choice(action_patterns)
```

### High-Quality Dataset Generation

For better quality videos:

```bash
python scripts/preprocess/vizdoom/generate_vizdoom_videos.py \
    --output_dir datasets/vizdoom_hq \
    --num_train_videos 5000 \
    --num_test_videos 1000 \
    --sequence_length 32 \
    --resolution 128 \
    --scenarios basic defend_the_center defend_the_line health_gathering
```

## Troubleshooting

### Common Issues

#### 1. VizDoom Installation Problems

**Error:** `ImportError: No module named 'vizdoom'`

**Solution:**
```bash
# Install system dependencies first
sudo apt-get install build-essential libboost-all-dev libsdl2-dev
pip install vizdoom
```

#### 2. Memory Issues

**Error:** Out of memory during video generation

**Solutions:**
- Reduce `num_train_videos` and `num_test_videos`
- Use lower resolution (32x32 instead of 64x64)
- Generate videos in smaller batches

#### 3. Video Generation Errors

**Error:** Scenario not found or crashes

**Solutions:**
- Try using only the "basic" scenario first
- Check VizDoom installation: `python -c "import vizdoom; print(vzd.scenarios_path)"`
- Ensure all system dependencies are installed

#### 4. Performance Issues

**Problem:** Slow video generation

**Solutions:**
- Run on a machine with good CPU performance
- Use fewer scenarios initially
- Generate videos in parallel for different scenarios

### Debug Mode

Enable debug mode to see what's happening:

```python
import vizdoom as vzd

# Test with visible window
game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/basic.cfg")
game.set_window_visible(True)  # Make window visible
game.init()

# Test episode
game.new_episode()
while not game.is_episode_finished():
    state = game.get_state()
    if state is not None:
        print(f"Frame shape: {state.screen_buffer.shape}")
    game.make_action([0, 0, 0, 0, 0, 0, 0])  # No action

game.close()
```

## File Structure

```
videogpt/
├── scripts/preprocess/vizdoom/
│   ├── create_vizdoom_dataset.sh    # Main dataset creation script
│   ├── generate_vizdoom_videos.py   # Video generation script
│   ├── vizdoom_to_hdf5.py          # HDF5 conversion script
│   └── README.md                   # Detailed documentation
├── datasets/vizdoom/               # Generated dataset directory
│   ├── train/                      # Training videos by class
│   │   ├── class_0/               # basic scenario
│   │   ├── class_1/               # defend_the_center scenario
│   │   ├── class_2/               # defend_the_line scenario
│   │   ├── class_3/               # health_gathering scenario
│   │   └── actions/               # Action sequences (NPY files)
│   ├── test/                      # Test videos by class
│   │   ├── class_0/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   ├── class_3/
│   │   └── actions/               # Action sequences (NPY files)
│   └── vizdoom.h5                 # HDF5 dataset file with actions
└── VIZDOOM_README.md              # This file
```

## Requirements

### System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.6 or higher
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2-10GB depending on dataset size
- **CPU**: Multi-core processor recommended

### Python Dependencies

```bash
# Core dependencies
pip install vizdoom opencv-python numpy h5py tqdm

# Optional for better performance
pip install pillow matplotlib scipy
```

### VizDoom Dependencies

- **SDL2**: Graphics library
- **Boost**: C++ libraries
- **CMake**: Build system
- **OpenGL**: Graphics rendering

## Dataset Format

### HDF5 Structure

The generated `vizdoom.h5` file contains:

```
vizdoom.h5
├── train_data      # Training video frames [N_frames, 64, 64] uint8
├── train_actions   # Training action sequences [N_frames, 7] uint8
├── train_idx       # Start indices for each training video
├── train_labels    # Class labels for training videos (0-3)
├── test_data       # Test video frames [N_frames, 64, 64] uint8
├── test_actions    # Test action sequences [N_frames, 7] uint8
├── test_idx        # Start indices for each test video
├── test_labels     # Class labels for test videos (0-3)
└── attributes      # Metadata
    ├── sequence_length: 16
    ├── resolution: 64
    ├── n_classes: 4
    ├── action_dim: 7
    ├── class_names: [b'basic', b'defend_the_center', b'defend_the_line', b'health_gathering']
    └── action_names: [b'move_left', b'move_right', b'attack', b'move_forward', b'move_backward', b'turn_left', b'turn_right']
```

### Video Characteristics

- **Resolution**: 64x64 pixels (configurable)
- **Sequence Length**: 16 frames per video (configurable)
- **Format**: Grayscale (single channel)
- **Value Range**: 0-255 (uint8)
- **Classes**: 4 scenario types
- **Frame Rate**: ~10 FPS (configurable)

### Action Characteristics

- **Action Dimension**: 7 binary actions per frame
- **Action Types**: 
  - `move_left` (0): Move left
  - `move_right` (1): Move right  
  - `attack` (2): Fire weapon
  - `move_forward` (3): Move forward
  - `move_backward` (4): Move backward
  - `turn_left` (5): Turn left
  - `turn_right` (6): Turn right
- **Format**: Binary array [0, 1, 0, 0, 0, 0, 0] indicating which actions are active
- **Synchronization**: Each action corresponds to the frame that follows it

## Examples

### Generate Small Test Dataset

```bash
python scripts/preprocess/vizdoom/generate_vizdoom_videos.py \
    --output_dir test_vizdoom \
    --num_train_videos 50 \
    --num_test_videos 10 \
    --sequence_length 8 \
    --resolution 32
```

### Generate High-Quality Dataset

```bash
python scripts/preprocess/vizdoom/generate_vizdoom_videos.py \
    --output_dir vizdoom_hq \
    --num_train_videos 5000 \
    --num_test_videos 1000 \
    --sequence_length 32 \
    --resolution 128 \
    --scenarios basic defend_the_center health_gathering
```

### Custom Scenario Dataset

```bash
# First, add your custom scenario to the scenarios list
python scripts/preprocess/vizdoom/generate_vizdoom_videos.py \
    --output_dir custom_vizdoom \
    --scenarios basic your_custom_scenario
```

## Citation

If you use this VizDoom dataset creation code in your research, please cite:

### VideoGPT Paper
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

### VizDoom Platform
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

## License

This VizDoom dataset creation code follows the same license as the VideoGPT project. See the main project LICENSE file for details.

## Support

For issues related to:
- **VizDoom installation**: Check the [official VizDoom documentation](https://vizdoom.farama.org/)
- **VideoGPT training**: Check the main VideoGPT repository
- **Dataset creation**: Open an issue in this repository

---

**Happy video generation with VizDoom and VideoGPT! 🎮🎬**
