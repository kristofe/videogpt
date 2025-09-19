# Device Support for VideoGPT

This document describes the device support modifications made to VideoGPT to enable running on CPU, MPS (Metal Performance Shaders), and CUDA devices.

## Supported Devices

- **CUDA**: NVIDIA GPUs (original support)
- **MPS**: Apple Silicon Macs with Metal Performance Shaders
- **CPU**: Any system with PyTorch CPU support

## Changes Made

### 1. Device Detection Utilities (`videogpt/utils.py`)

Added new utility functions for device detection and management:

- `get_device(device_type=None)`: Auto-detects the best available device or uses specified device type
- `get_device_count()`: Returns the number of available devices for the current backend
- `is_distributed_available()`: Checks if distributed training is available for the current device type

### 2. Model Loading (`videogpt/download.py`)

Updated all model loading functions to support device-agnostic loading:

- `load_vqvae()`: Now accepts device parameter (string or torch.device)
- `load_videogpt()`: Now accepts device parameter (string or torch.device)  
- `load_i3d_pretrained()`: Now accepts device parameter (string or torch.device)

### 3. Training Scripts

Updated both training scripts to support different devices:

- `scripts/train_videogpt.py`: Added `--device` argument and device-specific PyTorch Lightning configuration
- `scripts/train_vqvae.py`: Added `--device` argument and device-specific PyTorch Lightning configuration

### 4. Sampling Scripts

Updated sampling script to support different devices:

- `scripts/sample_videogpt.py`: Added `--device` argument for device selection

### 5. FVD Computation

Updated FVD computation script to support different devices:

- `scripts/compute_fvd.py`: Added `--device` argument and fallback to single-process mode for MPS/CPU

### 6. Dependencies

Updated `requirements.txt` to use more recent PyTorch versions that support MPS:

- PyTorch >= 2.0.0 (supports MPS on Apple Silicon)
- PyTorch Lightning >= 2.0.0 (updated API compatibility)

## Usage

### Command Line Interface

All scripts now support a `--device` argument:

```bash
# Auto-detect best device
python scripts/sample_videogpt.py --ckpt ucf101_uncond_gpt --n 8

# Force specific device
python scripts/sample_videogpt.py --ckpt ucf101_uncond_gpt --n 8 --device mps
python scripts/sample_videogpt.py --ckpt ucf101_uncond_gpt --n 8 --device cpu
python scripts/sample_videogpt.py --ckpt ucf101_uncond_gpt --n 8 --device cuda

# Training with specific device
python scripts/train_videogpt.py --data_path /path/to/data --device mps
```

### Programmatic Usage

```python
from videogpt.utils import get_device
from videogpt.download import load_videogpt

# Auto-detect device
device = get_device()
print(f"Using device: {device}")

# Load model on specific device
gpt = load_videogpt('ucf101_uncond_gpt', device='mps')
# or
gpt = load_videogpt('ucf101_uncond_gpt', device=device)
```

## Device Priority

The auto-detection follows this priority order:

1. **CUDA** (if available and requested)
2. **MPS** (if available and on Apple Silicon)
3. **CPU** (fallback)

## Performance Considerations

- **CUDA**: Best performance for training and inference on NVIDIA GPUs
- **MPS**: Good performance on Apple Silicon Macs, especially for inference
- **CPU**: Slower but most compatible, useful for development and testing

## Testing

A test script is provided to verify device support:

```bash
python test_device_support.py
```

This script will:
- Test device detection
- Test model loading (if network allows)
- Test simple model creation and forward pass
- Report the recommended device for your system

## Notes

- MPS support requires PyTorch 2.0+ and Apple Silicon Mac
- Distributed training is only supported on CUDA devices
- Some operations may be slower on MPS compared to CUDA
- CPU mode is always available as a fallback

## Troubleshooting

### MPS Issues
If you encounter MPS-related errors, try:
1. Update to the latest PyTorch version
2. Use CPU mode as fallback: `--device cpu`
3. Check that you're running on Apple Silicon

### CUDA Issues
If CUDA is not detected:
1. Ensure NVIDIA drivers are installed
2. Install PyTorch with CUDA support
3. Use CPU or MPS mode as fallback

### Memory Issues
If you run out of memory:
1. Reduce batch size
2. Use CPU mode (slower but less memory)
3. Use gradient checkpointing in training
