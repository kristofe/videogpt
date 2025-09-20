#!/usr/bin/env python3
"""
Generate Moving MNIST videos for VideoGPT training.

This script creates a VideoMNIST dataset by generating video sequences
from MNIST digits with various motion patterns (translation, rotation, scaling).
"""

import os
import argparse
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import cv2
from tqdm import tqdm
import math
import random


def generate_motion_trajectory(sequence_length, image_size, motion_type='bounce'):
    """Generate motion trajectory for a digit."""
    if motion_type == 'bounce':
        # Bouncing ball motion
        x = np.linspace(0, image_size - 28, sequence_length)
        y = np.abs(np.sin(np.linspace(0, 4 * np.pi, sequence_length))) * (image_size - 28)
        return x, y
    elif motion_type == 'circular':
        # Circular motion
        t = np.linspace(0, 2 * np.pi, sequence_length)
        center_x, center_y = image_size // 2, image_size // 2
        radius = min(image_size // 2 - 14, 50)
        x = center_x + radius * np.cos(t) - 14
        y = center_y + radius * np.sin(t) - 14
        return x, y
    elif motion_type == 'linear':
        # Linear motion with random direction
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(1, 3)
        x = np.linspace(0, image_size - 28, sequence_length)
        y = np.linspace(0, image_size - 28, sequence_length)
        # Add some randomness
        x += np.random.normal(0, 2, sequence_length)
        y += np.random.normal(0, 2, sequence_length)
        return x, y
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")


def apply_transformations(image, frame_idx, sequence_length, motion_type='bounce'):
    """Apply transformations to create motion."""
    # Get motion trajectory
    x, y = generate_motion_trajectory(sequence_length, 64, motion_type)
    
    # Create transformation matrix
    angle = random.uniform(-10, 10) if frame_idx % 10 == 0 else 0  # Occasional rotation
    scale = 1.0 + random.uniform(-0.1, 0.1) if frame_idx % 15 == 0 else 1.0  # Occasional scaling
    
    # Create 3x3 transformation matrix
    center = (32, 32)  # Center of 28x28 image in 64x64 canvas
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply translation
    M[0, 2] += x[frame_idx] - 32
    M[1, 2] += y[frame_idx] - 32
    
    # Apply transformation
    transformed = cv2.warpAffine(image, M, (64, 64), borderValue=0)
    
    return transformed


def create_video_sequence(digit_image, sequence_length=16, motion_type='bounce'):
    """Create a video sequence from a single MNIST digit image."""
    # Convert to numpy and resize to 64x64 for better motion
    digit_np = digit_image.squeeze().numpy()
    digit_resized = cv2.resize(digit_np, (28, 28))
    
    # Create 64x64 canvas
    canvas = np.zeros((64, 64), dtype=np.uint8)
    canvas[18:46, 18:46] = digit_resized
    
    # Generate video sequence
    video_frames = []
    for frame_idx in range(sequence_length):
        frame = apply_transformations(canvas, frame_idx, sequence_length, motion_type)
        video_frames.append(frame)
    
    return np.array(video_frames)


def generate_videomnist_dataset(output_dir, sequence_length=16, videos_per_digit=1000, 
                               motion_types=['bounce', 'circular', 'linear'], 
                               image_size=64):
    """Generate the complete VideoMNIST dataset."""
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create class directories
    for digit in range(10):
        os.makedirs(os.path.join(train_dir, f'class_{digit}'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, f'class_{digit}'), exist_ok=True)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
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
        
        # Save video
        video_filename = f'video_{video_count[label]:04d}.avi'
        video_path = os.path.join(train_dir, f'class_{label}', video_filename)
        
        # Write video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (image_size, image_size), isColor=False)
        
        if out.isOpened():
            for frame in video_frames:
                # Ensure frame is in the correct format
                frame_uint8 = frame.astype(np.uint8)
                out.write(frame_uint8)
            out.release()
        else:
            print(f"Error: Could not open video writer for {video_path}")
            continue
        
        video_count[label] += 1
    
    # Generate test videos
    print("Generating test videos...")
    video_count = {digit: 0 for digit in range(10)}
    test_videos_per_digit = videos_per_digit // 5  # 20% for test
    
    for idx in tqdm(range(len(test_dataset)), desc="Test videos"):
        image, label = test_dataset[idx]
        
        if video_count[label] >= test_videos_per_digit:
            continue
            
        # Select random motion type
        motion_type = random.choice(motion_types)
        
        # Create video sequence
        video_frames = create_video_sequence(image, sequence_length, motion_type)
        
        # Save video
        video_filename = f'video_{video_count[label]:04d}.avi'
        video_path = os.path.join(test_dir, f'class_{label}', video_filename)
        
        # Write video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (image_size, image_size), isColor=False)
        
        if out.isOpened():
            for frame in video_frames:
                # Ensure frame is in the correct format
                frame_uint8 = frame.astype(np.uint8)
                out.write(frame_uint8)
            out.release()
        else:
            print(f"Error: Could not open video writer for {video_path}")
            continue
        
        video_count[label] += 1
    
    print(f"VideoMNIST dataset generated successfully in {output_dir}")
    print(f"Training videos per digit: {videos_per_digit}")
    print(f"Test videos per digit: {test_videos_per_digit}")
    print(f"Motion types used: {motion_types}")


def main():
    parser = argparse.ArgumentParser(description='Generate VideoMNIST dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for the dataset')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Number of frames per video')
    parser.add_argument('--videos_per_digit', type=int, default=1000,
                       help='Number of videos to generate per digit class')
    parser.add_argument('--motion_types', nargs='+', 
                       default=['bounce', 'circular', 'linear'],
                       help='Types of motion to apply')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Size of the video frames')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    generate_videomnist_dataset(
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        videos_per_digit=args.videos_per_digit,
        motion_types=args.motion_types,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
