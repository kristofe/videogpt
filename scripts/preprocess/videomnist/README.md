# VideoMNIST Dataset Creation

This directory contains scripts to create a VideoMNIST dataset for VideoGPT training. The dataset consists of moving MNIST digits with various motion patterns.

## Overview

VideoMNIST is a synthetic video dataset created by animating MNIST digits with different motion patterns:
- **Bouncing motion**: Digits bounce around the frame
- **Circular motion**: Digits move in circular patterns
- **Linear motion**: Digits move in straight lines with some randomness

## Files

- `generate_videomnist.py`: Generates Moving MNIST videos from the MNIST dataset
- `videomnist_to_hdf5.py`: Converts video files to HDF5 format for VideoGPT
- `create_videomnist_dataset.sh`: Orchestrates the complete dataset creation process
- `README.md`: This documentation file

## Quick Start

To create the VideoMNIST dataset:

```bash
sh scripts/preprocess/videomnist/create_videomnist_dataset.sh datasets/videomnist
```

This will create:
- `datasets/videomnist/videos/`: Directory containing MP4 video files organized by class
- `datasets/videomnist/hdf5/`: Directory containing HDF5 files for VideoGPT training

## Dataset Structure

### Video Directory Structure
```
videos/
├── train/
│   ├── class_0/
│   │   ├── video_0000.mp4
│   │   ├── video_0001.mp4
│   │   └── ...
│   ├── class_1/
│   │   └── ...
│   └── ...
└── test/
    ├── class_0/
    └── ...
```

### HDF5 Format
The HDF5 files contain:
- `train_data`: Training video frames (uint8, shape: [N_frames, H, W])
- `train_idx`: Start indices for each video
- `train_labels`: Class labels for each video
- `test_data`: Test video frames
- `test_idx`: Start indices for test videos
- `test_labels`: Test class labels

## Parameters

### Video Generation (`generate_videomnist.py`)
- `--output_dir`: Output directory for videos
- `--sequence_length`: Number of frames per video (default: 16)
- `--videos_per_digit`: Number of videos per digit class (default: 1000)
- `--motion_types`: Types of motion to apply (default: ['bounce', 'circular', 'linear'])
- `--image_size`: Size of video frames (default: 64)

### HDF5 Conversion (`videomnist_to_hdf5.py`)
- `--data_dir`: Directory containing video files
- `--output_file`: Output HDF5 file path
- `--sequence_length`: Number of frames per video sequence (default: 16)
- `--resolution`: Resolution of video frames (default: 64)

## Usage with VideoGPT

### Training VQ-VAE
```bash
python scripts/train_vqvae.py \
    --data_path datasets/videomnist/hdf5/videomnist.h5 \
    --resolution 64 \
    --sequence_length 16 \
    --batch_size 16 \
    --gpus 1
```

### Training VideoGPT
```bash
python scripts/train_videogpt.py \
    --data_path datasets/videomnist/hdf5/videomnist.h5 \
    --resolution 64 \
    --sequence_length 16 \
    --batch_size 8 \
    --gpus 1
```

### Sampling VideoGPT
```bash
python scripts/sample_videogpt.py \
    --data_path datasets/videomnist/hdf5/videomnist.h5 \
    --resolution 64 \
    --sequence_length 16
```

## Customization

### Adding New Motion Types
To add new motion patterns, modify the `generate_motion_trajectory()` function in `generate_videomnist.py`:

```python
def generate_motion_trajectory(sequence_length, image_size, motion_type='bounce'):
    if motion_type == 'your_new_motion':
        # Implement your motion pattern
        x = your_x_trajectory
        y = your_y_trajectory
        return x, y
    # ... existing motion types
```

### Modifying Video Parameters
You can adjust various parameters in the generation script:
- Frame rate (currently 10 FPS)
- Video duration (controlled by sequence_length)
- Image transformations (rotation, scaling)
- Background color and effects

## Requirements

- Python 3.6+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- PyTorch (`pip install torch`)
- Torchvision (`pip install torchvision`)
- H5py (`pip install h5py`)
- TQDM (`pip install tqdm`)

## Notes

- The dataset is generated deterministically with fixed random seeds for reproducibility
- Videos are generated in grayscale format
- The default resolution is 64x64 pixels
- Each video contains 16 frames by default
- Training set contains 1000 videos per digit class (10,000 total)
- Test set contains 200 videos per digit class (2,000 total)
