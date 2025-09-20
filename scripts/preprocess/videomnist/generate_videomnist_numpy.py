#!/usr/bin/env python3
"""
Generate Moving MNIST videos as numpy arrays for VideoGPT training.

This script creates a VideoMNIST dataset by generating video sequences
from MNIST digits with various motion patterns and saves them directly
as numpy arrays instead of video files.
"""

import os
import argparse
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import math
import random
import h5py
from tqdm import tqdm


def generate_motion_trajectory(sequence_length, image_size, motion_type='bounce'):
    """Generate motion trajectory for a digit."""
    if motion_type == 'bounce':
        # Bouncing ball motion - keep within bounds
        x = np.linspace(5, image_size - 33, sequence_length)
        y = np.abs(np.sin(np.linspace(0, 4 * np.pi, sequence_length))) * (image_size - 33) + 5
        return x, y
    elif motion_type == 'circular':
        # Circular motion - smaller radius to keep within bounds
        t = np.linspace(0, 2 * np.pi, sequence_length)
        center_x, center_y = image_size // 2, image_size // 2
        radius = min(image_size // 2 - 20, 20)  # Smaller radius
        x = center_x + radius * np.cos(t) - 14
        y = center_y + radius * np.sin(t) - 14
        return x, y
    elif motion_type == 'linear':
        # Linear motion - keep within bounds
        x = np.linspace(5, image_size - 33, sequence_length)
        y = np.linspace(5, image_size - 33, sequence_length)
        # Add some randomness but keep within bounds
        x += np.random.normal(0, 1, sequence_length)
        y += np.random.normal(0, 1, sequence_length)
        # Clamp to bounds
        x = np.clip(x, 5, image_size - 33)
        y = np.clip(y, 5, image_size - 33)
        return x, y
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")


def apply_transformations(image, frame_idx, sequence_length, motion_type='bounce'):
    """Apply transformations to create motion."""
    # Get motion trajectory
    x, y = generate_motion_trajectory(sequence_length, 64, motion_type)
    
    # Create transformation parameters
    angle = random.uniform(-10, 10) if frame_idx % 10 == 0 else 0  # Occasional rotation
    scale = 1.0 + random.uniform(-0.1, 0.1) if frame_idx % 15 == 0 else 1.0  # Occasional scaling
    
    # Calculate translation
    dx = x[frame_idx] - 32
    dy = y[frame_idx] - 32
    
    # Apply transformation using scipy
    from scipy.ndimage import affine_transform
    
    # Create rotation matrix
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    
    # Apply scaling and rotation
    matrix = np.array([[cos_a * scale, -sin_a * scale],
                       [sin_a * scale, cos_a * scale]])
    
    # Apply transformation
    transformed = affine_transform(image, matrix, offset=[dy, dx], order=1, mode='constant', cval=0)
    
    return transformed


def create_video_sequence(digit_image, sequence_length=16, motion_type='bounce'):
    """Create a video sequence from a single MNIST digit image."""
    # Convert to numpy and scale to 0-255 range
    digit_np = digit_image.squeeze().numpy()
    digit_resized = (digit_np * 255).astype(np.uint8)  # Scale to 0-255
    
    # Create 64x64 canvas
    canvas = np.zeros((64, 64), dtype=np.uint8)
    canvas[18:46, 18:46] = digit_resized
    
    # Generate video sequence
    video_frames = []
    for frame_idx in range(sequence_length):
        frame = apply_transformations(canvas, frame_idx, sequence_length, motion_type)
        video_frames.append(frame)
    
    return np.array(video_frames)


def generate_videomnist_dataset(output_file, sequence_length=16, videos_per_digit=1000, 
                               motion_types=['bounce', 'circular', 'linear'], 
                               image_size=64):
    """Generate the complete VideoMNIST dataset and save as HDF5."""
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Prepare data storage
    train_frames = []
    train_labels = []
    train_video_starts = []
    test_frames = []
    test_labels = []
    test_video_starts = []
    
    current_train_idx = 0
    current_test_idx = 0
    
    # Generate training videos
    print("Generating training videos...")
    video_count = {digit: 0 for digit in range(10)}
    
    for idx in tqdm(range(len(train_dataset)), desc="Training videos"):
        image, label = train_dataset[idx]
        
        if video_count[label] >= videos_per_digit:
            continue
            
        # Select random motion type
        motion_type = random.choice(motion_types)
        
        # Create video sequence
        video_frames = create_video_sequence(image, sequence_length, motion_type)
        
        # Store frames
        train_frames.append(video_frames)
        train_labels.append(label)
        train_video_starts.append(current_train_idx)
        current_train_idx += len(video_frames)
        
        video_count[label] += 1
    
    # Generate test videos
    print("Generating test videos...")
    video_count = {digit: 0 for digit in range(10)}
    test_videos_per_digit = max(1, videos_per_digit // 5)  # At least 1 video per digit
    
    for idx in tqdm(range(len(test_dataset)), desc="Test videos"):
        image, label = test_dataset[idx]
        
        if video_count[label] >= test_videos_per_digit:
            continue
            
        # Select random motion type
        motion_type = random.choice(motion_types)
        
        # Create video sequence
        video_frames = create_video_sequence(image, sequence_length, motion_type)
        
        # Store frames
        test_frames.append(video_frames)
        test_labels.append(label)
        test_video_starts.append(current_test_idx)
        current_test_idx += len(video_frames)
        
        video_count[label] += 1
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    train_frames = np.concatenate(train_frames, axis=0).astype(np.uint8)
    train_labels = np.array(train_labels, dtype=np.int64)
    train_video_starts = np.array(train_video_starts, dtype=np.int64)
    
    test_frames = np.concatenate(test_frames, axis=0).astype(np.uint8)
    test_labels = np.array(test_labels, dtype=np.int64)
    test_video_starts = np.array(test_video_starts, dtype=np.int64)
    
    # Save to HDF5
    print(f"Saving dataset to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        # Training data
        f.create_dataset('train_data', data=train_frames, compression='gzip')
        f.create_dataset('train_idx', data=train_video_starts, compression='gzip')
        f.create_dataset('train_labels', data=train_labels, compression='gzip')
        
        # Test data
        f.create_dataset('test_data', data=test_frames, compression='gzip')
        f.create_dataset('test_idx', data=test_video_starts, compression='gzip')
        f.create_dataset('test_labels', data=test_labels, compression='gzip')
        
        # Metadata
        f.attrs['sequence_length'] = sequence_length
        f.attrs['resolution'] = image_size
        f.attrs['n_classes'] = 10
        f.attrs['train_videos'] = len(train_video_starts)
        f.attrs['test_videos'] = len(test_video_starts)
        f.attrs['motion_types'] = motion_types
    
    print(f"VideoMNIST dataset generated successfully!")
    print(f"Training videos: {len(train_video_starts)}")
    print(f"Test videos: {len(test_video_starts)}")
    print(f"Total frames: {len(train_frames) + len(test_frames)}")
    print(f"Motion types used: {motion_types}")


def main():
    parser = argparse.ArgumentParser(description='Generate VideoMNIST dataset')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output HDF5 file path')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Number of frames per video')
    parser.add_argument('--videos_per_digit', type=int, default=1000,
                       help='Number of videos to generate per digit class')
    parser.add_argument('--motion_types', nargs='+', 
                       default=['bounce', 'circular', 'linear'],
                       help='Types of motion to apply')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Size of the video frames')
    parser.add_argument('--create_samples', action='store_true',
                       help='Create sample video visualizations')
    parser.add_argument('--sample_formats', nargs='+', default=['gif', 'mp4'],
                       choices=['gif', 'mp4'],
                       help='Formats for sample videos')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sample videos to create')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    generate_videomnist_dataset(
        output_file=args.output_file,
        sequence_length=args.sequence_length,
        videos_per_digit=args.videos_per_digit,
        motion_types=args.motion_types,
        image_size=args.image_size
    )
    
    # Create sample visualizations if requested
    if args.create_samples:
        print("\nCreating sample visualizations...")
        output_dir = os.path.dirname(args.output_file)
        sample_dir = os.path.join(output_dir, "samples")
        
        # Import and run visualization
        import sys
        sys.path.append(os.path.dirname(__file__))
        from visualize_videomnist import visualize_dataset
        
        visualize_dataset(
            hdf5_path=args.output_file,
            output_dir=sample_dir,
            num_samples=args.num_samples,
            formats=args.sample_formats,
            create_montage=True,
            fps=10,
            gif_duration=200
        )


if __name__ == '__main__':
    main()
