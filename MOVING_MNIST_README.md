# Moving MNIST Integration for VideoGPT

This document describes how to use the Moving MNIST dataset with VideoGPT, including automatic scaling to the correct resolution.

## Overview

Moving MNIST is a standard benchmark dataset for video generation that consists of MNIST digits moving around in a 64x64 canvas with bouncing motion. This integration provides:

- **Simple Integration**: Easy-to-use Moving MNIST dataset class that works with VideoGPT's existing framework
- **Automatic Scaling**: Built-in scaling from the default 64x64 resolution to any target resolution
- **On-the-fly Generation**: Videos are generated dynamically during training (no need to pre-generate large datasets)
- **Configurable Parameters**: Control number of digits, motion patterns, and other parameters

## Quick Start

### 1. Create a Moving MNIST Dataset

```bash
# Create a small dataset for testing
python scripts/create_moving_mnist.py \
    --output_file datasets/moving_mnist/moving_mnist_64x64.h5 \
    --sequence_length 16 \
    --resolution 64 \
    --num_digits 2 \
    --videos_per_digit 1000
```

### 2. Train VideoGPT on Moving MNIST

```bash
# Train with Moving MNIST dataset
python scripts/train_videogpt.py \
    --data_path moving_mnist \
    --dataset_type moving_mnist \
    --resolution 64 \
    --sequence_length 16 \
    --batch_size 8 \
    --num_digits 2 \
    --videos_per_digit 1000 \
    --max_steps 10000
```

### 3. Use the Example Script

```bash
# Run the complete example (create dataset, train, generate)
python scripts/example_moving_mnist.py --create_dataset --visualize --train --generate
```

## Features

### Automatic Resolution Scaling

The Moving MNIST dataset automatically scales from its native 64x64 resolution to any target resolution:

```python
# Scale to 128x128
dataset = MovingMNISTDataset(
    sequence_length=16,
    train=True,
    resolution=128,  # Will be scaled from 64x64
    num_digits=2
)

# Scale to 256x256
dataset = MovingMNISTDataset(
    sequence_length=16,
    train=True,
    resolution=256,  # Will be scaled from 64x64
    num_digits=2
)
```

### Configurable Parameters

- **`num_digits`**: Number of digits per video (1 or 2)
- **`videos_per_digit`**: Number of videos to generate per digit class
- **`sequence_length`**: Number of frames per video
- **`resolution`**: Target resolution (automatically scaled)
- **`max_velocity`**: Maximum velocity for digit movement
- **`canvas_size`**: Size of the canvas where digits move (default: 64)

### Motion Patterns

The current implementation uses bouncing motion where digits:
- Move with random initial velocities
- Bounce off the canvas boundaries
- Can have 1 or 2 digits per video
- Mix different digit classes when using 2 digits

## Integration with VideoGPT

The Moving MNIST dataset integrates seamlessly with VideoGPT's existing data loading framework:

```python
from videogpt import VideoData, VideoGPT

# Create data module
args = argparse.Namespace(
    data_path='moving_mnist',
    dataset_type='moving_mnist',
    resolution=64,
    sequence_length=16,
    num_digits=2,
    videos_per_digit=1000
)

data = VideoData(args)
model = VideoGPT(args)

# Use with PyTorch Lightning trainer
trainer = pl.Trainer(max_steps=10000)
trainer.fit(model, data)
```

## File Structure

```
videogpt/
├── moving_mnist.py              # Main Moving MNIST implementation
├── data.py                      # Updated with Moving MNIST integration
scripts/
├── create_moving_mnist.py       # Script to create HDF5 datasets
├── example_moving_mnist.py      # Complete example script
└── train_videogpt.py            # Updated training script
```

## API Reference

### MovingMNISTDataset

```python
class MovingMNISTDataset(data.Dataset):
    def __init__(
        self,
        sequence_length: int = 16,
        train: bool = True,
        resolution: int = 64,
        num_digits: int = 2,
        digit_size: int = 28,
        canvas_size: int = 64,
        max_velocity: float = 1.0,
        videos_per_digit: int = 1000,
        seed: int = 42
    ):
        """
        Moving MNIST Dataset for VideoGPT.
        
        Args:
            sequence_length: Number of frames per video
            train: Whether to use training or test split
            resolution: Target resolution (will be scaled from canvas_size)
            num_digits: Number of digits per video (1 or 2)
            digit_size: Size of individual MNIST digits
            canvas_size: Size of the canvas where digits move
            max_velocity: Maximum velocity for digit movement
            videos_per_digit: Number of videos per digit class
            seed: Random seed for reproducibility
        """
```

### create_moving_mnist_dataset

```python
def create_moving_mnist_dataset(
    output_file: str,
    sequence_length: int = 16,
    resolution: int = 64,
    num_digits: int = 2,
    videos_per_digit: int = 1000,
    train_split: float = 0.8
):
    """
    Create and save a Moving MNIST dataset to HDF5 format.
    
    Args:
        output_file: Path to save the HDF5 file
        sequence_length: Number of frames per video
        resolution: Target resolution for videos
        num_digits: Number of digits per video
        videos_per_digit: Number of videos per digit class
        train_split: Fraction of data to use for training
    """
```

## Examples

### Basic Usage

```python
from videogpt.moving_mnist import MovingMNISTDataset

# Create dataset
dataset = MovingMNISTDataset(
    sequence_length=16,
    train=True,
    resolution=128,  # Scaled from 64x64
    num_digits=2
)

# Get a sample
sample = dataset[0]
video = sample['video']  # Shape: [1, 16, 128, 128]
```

### Training VideoGPT

```python
import argparse
from videogpt import VideoData, VideoGPT
import pytorch_lightning as pl

# Create args
args = argparse.Namespace(
    data_path='moving_mnist',
    dataset_type='moving_mnist',
    resolution=64,
    sequence_length=16,
    batch_size=8,
    num_digits=2,
    videos_per_digit=1000,
    # ... other VideoGPT args
)

# Create data and model
data = VideoData(args)
model = VideoGPT(args)

# Train
trainer = pl.Trainer(max_steps=10000)
trainer.fit(model, data)
```

## Performance Notes

- **Memory Efficient**: Videos are generated on-the-fly, so memory usage is minimal
- **Fast Generation**: Each video is generated in real-time during training
- **Reproducible**: Uses fixed random seeds for consistent results
- **Scalable**: Can easily generate millions of videos without storage issues

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Memory Issues**: Reduce `batch_size` or `videos_per_digit` if you run out of memory
3. **Slow Training**: The dataset generates videos on-the-fly, which may be slower than pre-generated datasets

### Performance Tips

1. **Use HDF5 Datasets**: For faster training, pre-generate datasets using `create_moving_mnist.py`
2. **Adjust Workers**: Increase `num_workers` for faster data loading
3. **Batch Size**: Use larger batch sizes if you have enough GPU memory

## Future Enhancements

- [ ] More motion patterns (circular, linear, etc.)
- [ ] Variable number of digits per video
- [ ] Custom digit sizes and fonts
- [ ] Background variations
- [ ] More realistic physics simulation
