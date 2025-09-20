# VideoMNIST Dataset Generation for VideoGPT

This guide explains how to create and use the VideoMNIST dataset for training VideoGPT models. VideoMNIST is a synthetic video dataset created by animating MNIST digits with various motion patterns.

## Overview

VideoMNIST extends the classic MNIST dataset by creating video sequences where digits move with different patterns:
- **Bouncing Motion**: Digits bounce around the frame like a ball
- **Circular Motion**: Digits move in circular patterns
- **Linear Motion**: Digits move in straight lines with some randomness

Each video contains 16 frames at 64x64 resolution, showing a single digit moving through various transformations.

### Recent Improvements (v2.0)

- ✅ **Fixed Black Image Issue**: Proper pixel value scaling (0-255 range)
- ✅ **Improved Motion Bounds**: Digits stay within visible frame area
- ✅ **Enhanced Visualization**: Added GIF and MP4 output capabilities
- ✅ **Better Motion Patterns**: More realistic and constrained movements
- ✅ **Quality Assurance**: 100-200+ visible pixels per frame

## Quick Start

### 1. Generate the Dataset

```bash
# Create VideoMNIST dataset
sh scripts/preprocess/videomnist/create_videomnist_dataset.sh datasets/videomnist
```

This will create:
- `datasets/videomnist/videomnist.h5` - HDF5 file ready for VideoGPT training
- 10,000 training videos (1,000 per digit class 0-9)
- 2,000 test videos (200 per digit class 0-9)

### 2. Train VideoGPT

```bash
# Train VQ-VAE first
python scripts/train_vqvae.py \
    --data_path datasets/videomnist/videomnist.h5 \
    --resolution 64 \
    --sequence_length 16 \
    --batch_size 16 \
    --gpus 1

# Train VideoGPT
python scripts/train_videogpt.py \
    --data_path datasets/videomnist/videomnist.h5 \
    --resolution 64 \
    --sequence_length 16 \
    --batch_size 8 \
    --gpus 1
```

### 3. Visualize Dataset

```bash
# Create sample visualizations (GIFs and MP4s)
python scripts/preprocess/videomnist/visualize_videomnist.py \
    --hdf5_path datasets/videomnist/videomnist.h5 \
    --output_dir samples \
    --num_samples 20 \
    --formats gif mp4 \
    --create_comparison
```

### 4. Generate Videos

```bash
# Sample new videos
python scripts/sample_videogpt.py \
    --data_path datasets/videomnist/videomnist.h5 \
    --resolution 64 \
    --sequence_length 16
```

## Dataset Structure

### HDF5 Format

The generated `videomnist.h5` file contains:

```
videomnist.h5
├── train_data      # Training video frames [N_frames, 64, 64] uint8
├── train_idx       # Start indices for each training video
├── train_labels    # Class labels for training videos (0-9)
├── test_data       # Test video frames [N_frames, 64, 64] uint8
├── test_idx        # Start indices for each test video
├── test_labels     # Class labels for test videos (0-9)
└── attributes      # Metadata (sequence_length, resolution, etc.)
```

### Video Characteristics

- **Resolution**: 64x64 pixels
- **Sequence Length**: 16 frames per video
- **Frame Rate**: 10 FPS (configurable)
- **Format**: Grayscale (single channel)
- **Value Range**: 0-255 (uint8)
- **Classes**: 10 digit classes (0-9)
- **Motion Bounds**: Digits constrained to stay within visible area (5px margin)
- **Pixel Density**: 100-200+ non-zero pixels per frame for clear visibility

## Customization

### Motion Patterns

You can customize the motion types by modifying the script:

```bash
python scripts/preprocess/videomnist/generate_videomnist_numpy.py \
    --output_file datasets/videomnist_custom.h5 \
    --motion_types bounce circular linear spiral \
    --videos_per_digit 500 \
    --sequence_length 20 \
    --image_size 128
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_file` | Required | Output HDF5 file path |
| `--sequence_length` | 16 | Number of frames per video |
| `--videos_per_digit` | 1000 | Videos per digit class |
| `--motion_types` | bounce, circular, linear | Motion patterns to use |
| `--image_size` | 64 | Video frame resolution |

### Adding New Motion Types

To add custom motion patterns, edit `generate_videomnist_numpy.py`:

```python
def generate_motion_trajectory(sequence_length, image_size, motion_type='bounce'):
    if motion_type == 'spiral':
        # Spiral motion
        t = np.linspace(0, 4 * np.pi, sequence_length)
        radius = np.linspace(10, 30, sequence_length)
        center_x, center_y = image_size // 2, image_size // 2
        x = center_x + radius * np.cos(t) - 14
        y = center_y + radius * np.sin(t) - 14
        return x, y
    # ... existing motion types
```

## Visualization

### Sample Video Generation

The VideoMNIST toolkit includes powerful visualization tools to create sample videos in multiple formats:

#### Basic Visualization

```bash
# Create sample GIFs and MP4s
python scripts/preprocess/videomnist/visualize_videomnist.py \
    --hdf5_path datasets/videomnist/videomnist.h5 \
    --output_dir samples \
    --num_samples 20 \
    --formats gif mp4
```

#### Advanced Visualization

```bash
# Create comprehensive visualization with class comparisons
python scripts/preprocess/videomnist/visualize_videomnist.py \
    --hdf5_path datasets/videomnist/videomnist.h5 \
    --output_dir samples \
    --num_samples 50 \
    --formats gif mp4 \
    --create_comparison \
    --fps 15
```

#### Quick Sample Creation

```bash
# Use the convenience script
sh scripts/preprocess/videomnist/create_samples.sh \
    datasets/videomnist/videomnist.h5 \
    output_samples \
    --num_samples 30 \
    --formats gif mp4 \
    --comparison
```

### Output Files

The visualization creates several types of files:

- **Individual Videos**: `digit_X_video_YYY.gif/mp4` - Sample videos for each digit class
- **Montage Image**: `videomnist_montage.png` - Static image showing start/end frames
- **Class Comparisons**: `class_comparisons/digit_X_comparison.gif` - Side-by-side motion patterns
- **Statistics**: `dataset_stats.txt` - Dataset information and statistics

### Visualization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_samples` | Number of sample videos to create | 20 |
| `--formats` | Output formats (gif, mp4) | gif mp4 |
| `--fps` | Frames per second for MP4 | 10 |
| `--create_comparison` | Create class comparison GIFs | False |
| `--num_per_class` | Videos per class for comparison | 3 |

### Integration with Dataset Generation

You can automatically create samples during dataset generation:

```bash
python scripts/preprocess/videomnist/generate_videomnist_numpy.py \
    --output_file datasets/videomnist/videomnist.h5 \
    --create_samples \
    --sample_formats gif mp4 \
    --num_samples 20
```

## Advanced Usage

### Memory-Efficient Generation

For large datasets, you can generate in batches:

```python
# Generate dataset in chunks
python scripts/preprocess/videomnist/generate_videomnist_numpy.py \
    --output_file datasets/videomnist_large.h5 \
    --videos_per_digit 5000 \
    --sequence_length 32
```

### Quality Control

Inspect generated videos:

```python
import h5py
import matplotlib.pyplot as plt

# Load dataset
with h5py.File('datasets/videomnist/videomnist.h5', 'r') as f:
    frames = f['train_data'][:16]  # First video
    labels = f['train_labels'][0]

# Display video frames
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, frame in enumerate(frames):
    axes[i//8, i%8].imshow(frame, cmap='gray')
    axes[i//8, i%8].set_title(f'Frame {i}')
    axes[i//8, i%8].axis('off')
plt.suptitle(f'Digit {labels} - Bouncing Motion')
plt.tight_layout()
plt.show()
```

## Training Tips

### VQ-VAE Training

For VideoMNIST, recommended VQ-VAE settings:

```bash
python scripts/train_vqvae.py \
    --data_path datasets/videomnist/videomnist.h5 \
    --resolution 64 \
    --sequence_length 16 \
    --embedding_dim 256 \
    --n_codes 512 \
    --n_hiddens 128 \
    --n_res_layers 2 \
    --downsample 2 4 4 \
    --batch_size 32 \
    --gpus 1
```

### VideoGPT Training

Recommended VideoGPT settings:

```bash
python scripts/train_videogpt.py \
    --data_path datasets/videomnist/videomnist.h5 \
    --vqvae path/to/vqvae_checkpoint.ckpt \
    --resolution 64 \
    --sequence_length 16 \
    --hidden_dim 256 \
    --heads 4 \
    --layers 6 \
    --batch_size 16 \
    --gpus 1
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `videos_per_digit` or `sequence_length`
2. **Slow Generation**: Use fewer motion types or smaller image size
3. **Poor Quality**: Increase `sequence_length` or add more motion variety
4. **Black Images**: This was a common issue that has been fixed in the current version
   - Ensure you're using the latest version of the generation scripts
   - The issue was related to pixel value scaling and motion bounds

### Performance Optimization

- Use SSD storage for faster I/O
- Increase `num_workers` in training scripts
- Use mixed precision training with `--precision 16`

### Quality Assurance

The VideoMNIST generation has been tested to ensure:
- **Pixel Values**: Proper 0-255 range for visible digits
- **Motion Bounds**: Digits stay within the visible frame area
- **Motion Patterns**: Clear bouncing, circular, and linear movements
- **Frame Quality**: 100-200+ non-zero pixels per frame for clear visibility

## File Structure

```
scripts/preprocess/videomnist/
├── generate_videomnist_numpy.py    # Main generation script
├── generate_videomnist.py          # Alternative (video files)
├── videomnist_to_hdf5.py           # Video to HDF5 converter
├── visualize_videomnist.py         # Visualization script
├── create_videomnist_dataset.sh    # Complete pipeline script
├── create_samples.sh               # Sample creation script
└── README.md                       # Detailed documentation
```

## Requirements

- Python 3.6+
- PyTorch
- Torchvision
- NumPy
- H5py
- TQDM
- SciPy (for transformations)
- OpenCV (for MP4 video creation)
- Pillow (for GIF creation)
- Matplotlib (for montage images)

Install with:
```bash
pip install torch torchvision numpy h5py tqdm scipy opencv-python pillow matplotlib
```

## Technical Implementation

### Motion Generation

The VideoMNIST system uses several key techniques:

1. **Pixel Value Scaling**: MNIST data (0-1 range) is properly scaled to 0-255 for visibility
2. **Bounded Motion**: All motion patterns are constrained to keep digits within the visible frame
3. **Affine Transformations**: Uses scipy's `affine_transform` for smooth motion and rotation
4. **Trajectory Generation**: Mathematical functions create realistic motion paths

### Motion Pattern Details

- **Bouncing**: Sine wave with linear horizontal movement
- **Circular**: Parametric circle with constrained radius
- **Linear**: Straight-line movement with random perturbations

### Quality Metrics

- **Visibility**: 100-200+ non-zero pixels per frame
- **Motion Range**: 5px margin from frame edges
- **Transformation**: Smooth interpolation with order=1
- **Bounds Checking**: All coordinates clipped to valid ranges

## Examples

### Generate Small Test Dataset

```bash
python scripts/preprocess/videomnist/generate_videomnist_numpy.py \
    --output_file test_videomnist.h5 \
    --videos_per_digit 10 \
    --sequence_length 8
```

### Generate High-Quality Dataset

```bash
python scripts/preprocess/videomnist/generate_videomnist_numpy.py \
    --output_file videomnist_hq.h5 \
    --videos_per_digit 2000 \
    --sequence_length 32 \
    --image_size 128 \
    --motion_types bounce circular linear
```

## Citation

If you use VideoMNIST in your research, please cite:

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

## License

This VideoMNIST generation code follows the same license as the VideoGPT project. See the main project LICENSE file for details.
