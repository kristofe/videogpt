"""
Moving MNIST Dataset for VideoGPT

This module provides a simple implementation of the Moving MNIST dataset
that integrates with VideoGPT's data loading framework. It includes
automatic scaling to the target resolution.
"""

import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import math
import random
from typing import Optional, Tuple, List


class MovingMNISTDataset(data.Dataset):
    """
    Moving MNIST Dataset that generates video sequences on-the-fly.
    
    This dataset creates moving MNIST digits with bouncing motion patterns
    and automatically scales them to the target resolution.
    """
    
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
        Initialize Moving MNIST dataset.
        
        Args:
            sequence_length: Number of frames in each video sequence
            train: Whether to use training or test split
            resolution: Target resolution for output videos (will be scaled)
            num_digits: Number of digits per video (1 or 2)
            digit_size: Size of individual MNIST digits
            canvas_size: Size of the canvas where digits move
            max_velocity: Maximum velocity for digit movement
            videos_per_digit: Number of videos to generate per digit class
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.train = train
        self.resolution = resolution
        self.num_digits = num_digits
        self.digit_size = digit_size
        self.canvas_size = canvas_size
        self.max_velocity = max_velocity
        self.videos_per_digit = videos_per_digit
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load MNIST dataset
        self._load_mnist()
        
        # Generate video metadata
        self._generate_video_metadata()
        
    def _load_mnist(self):
        """Load MNIST dataset with fallback to synthetic data."""
        transform = transforms.Compose([transforms.ToTensor()])
        
        try:
            self.mnist_dataset = datasets.MNIST(
                root='./data', 
                train=self.train, 
                download=True, 
                transform=transform
            )
        except Exception as e:
            print(f"Failed to download MNIST dataset: {e}")
            print("Creating synthetic MNIST-like data as fallback...")
            self.mnist_dataset = self._create_synthetic_mnist()
        
        # Group images by digit class
        self.digit_images = {i: [] for i in range(10)}
        for image, label in self.mnist_dataset:
            self.digit_images[label].append(image)
            
        # Ensure we have enough images per class
        min_images = min(len(images) for images in self.digit_images.values())
        if min_images < self.videos_per_digit:
            print(f"Warning: Only {min_images} images per class available, "
                  f"but {self.videos_per_digit} videos requested. Using {min_images}.")
            self.videos_per_digit = min_images
    
    def _create_synthetic_mnist(self):
        """Create synthetic MNIST-like data when real MNIST download fails."""
        print("Creating synthetic MNIST-like dataset...")
        
        class SyntheticMNIST:
            def __init__(self, train=True):
                self.train = train
                self.data = []
                self.targets = []
                
                # Generate synthetic digit images
                for digit in range(10):
                    for _ in range(1000 if train else 200):  # 1000 train, 200 test per digit
                        # Create a simple 28x28 image with the digit
                        image = torch.zeros(1, 28, 28)
                        
                        # Draw a simple representation of the digit
                        if digit == 0:
                            # Draw a circle
                            for i in range(28):
                                for j in range(28):
                                    dist = ((i - 14) ** 2 + (j - 14) ** 2) ** 0.5
                                    if 8 <= dist <= 12:
                                        image[0, i, j] = 1.0
                        elif digit == 1:
                            # Draw a vertical line
                            for i in range(28):
                                image[0, i, 14] = 1.0
                                if i > 5:
                                    image[0, i, 13] = 1.0
                        elif digit == 2:
                            # Draw a 2 shape
                            for i in range(28):
                                for j in range(28):
                                    if (i < 8 and j > 8) or (8 <= i < 16 and j > 20) or (16 <= i < 24 and j < 8) or (i >= 24 and j < 20):
                                        if abs(i - j) < 2 or abs(i + j - 28) < 2:
                                            image[0, i, j] = 1.0
                        else:
                            # For other digits, create simple patterns
                            for i in range(28):
                                for j in range(28):
                                    if (i + j) % 3 == digit % 3 and 5 < i < 23 and 5 < j < 23:
                                        image[0, i, j] = 1.0
                        
                        self.data.append(image)
                        self.targets.append(digit)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        return SyntheticMNIST(train=self.train)
    
    def _generate_video_metadata(self):
        """Generate metadata for all videos."""
        self.video_metadata = []
        
        for digit_class in range(10):
            for video_idx in range(self.videos_per_digit):
                # Select random images for this video
                if self.num_digits == 1:
                    digit_classes = [digit_class]
                else:  # num_digits == 2
                    # Mix with other classes for more variety
                    other_classes = [i for i in range(10) if i != digit_class]
                    digit_classes = [digit_class, random.choice(other_classes)]
                
                # Select random images
                selected_images = []
                for cls in digit_classes:
                    img_idx = random.randint(0, len(self.digit_images[cls]) - 1)
                    selected_images.append(self.digit_images[cls][img_idx])
                
                # Generate random initial positions and velocities
                positions = []
                velocities = []
                
                for _ in range(len(digit_classes)):
                    # Random initial position (leave margin for digit size)
                    margin = self.digit_size // 2
                    x = random.uniform(margin, self.canvas_size - margin)
                    y = random.uniform(margin, self.canvas_size - margin)
                    
                    # Random velocity
                    vx = random.uniform(-self.max_velocity, self.max_velocity)
                    vy = random.uniform(-self.max_velocity, self.max_velocity)
                    
                    positions.append([x, y])
                    velocities.append([vx, vy])
                
                self.video_metadata.append({
                    'digit_classes': digit_classes,
                    'images': selected_images,
                    'positions': positions,
                    'velocities': velocities
                })
    
    def __len__(self):
        return len(self.video_metadata)
    
    def __getitem__(self, idx):
        """Generate a video sequence."""
        metadata = self.video_metadata[idx]
        
        # Create video frames
        video_frames = self._generate_video_sequence(metadata)
        
        # Convert to tensor and scale to target resolution
        video_tensor = torch.from_numpy(video_frames).float() / 255.0  # [T, H, W]
        video_tensor = video_tensor.unsqueeze(1)  # [T, 1, H, W] for grayscale
        
        # Scale to target resolution
        if self.resolution != self.canvas_size:
            video_tensor = self._scale_video(video_tensor, self.resolution)
        else:
            # Convert to [C, T, H, W] format expected by VideoGPT
            video_tensor = video_tensor.permute(1, 0, 2, 3)  # [1, T, H, W]
        
        # Note: _scale_video already returns [C, T, H, W] format, so no additional permute needed
        
        # Convert from 1-channel to 3-channel by replicating the grayscale channel
        if video_tensor.shape[0] == 1:  # [1, T, H, W]
            video_tensor = video_tensor.repeat(3, 1, 1, 1)  # [3, T, H, W]
        
        # Normalize to [-0.5, 0.5] range as expected by VideoGPT
        video_tensor = video_tensor - 0.5
        
        return dict(video=video_tensor)
    
    def _generate_video_sequence(self, metadata):
        """Generate a single video sequence."""
        frames = []
        positions = [pos.copy() for pos in metadata['positions']]
        velocities = [vel.copy() for vel in metadata['velocities']]
        
        for frame_idx in range(self.sequence_length):
            # Create canvas
            canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
            
            # Draw each digit
            for i, (image, pos, vel) in enumerate(zip(metadata['images'], positions, velocities)):
                # Update position
                pos[0] += vel[0]
                pos[1] += vel[1]
                
                # Bounce off walls
                if pos[0] <= self.digit_size // 2 or pos[0] >= self.canvas_size - self.digit_size // 2:
                    vel[0] = -vel[0]
                if pos[1] <= self.digit_size // 2 or pos[1] >= self.canvas_size - self.digit_size // 2:
                    vel[1] = -vel[1]
                
                # Clamp position
                pos[0] = np.clip(pos[0], self.digit_size // 2, self.canvas_size - self.digit_size // 2)
                pos[1] = np.clip(pos[1], self.digit_size // 2, self.canvas_size - self.digit_size // 2)
                
                # Draw digit on canvas
                self._draw_digit(canvas, image, pos)
            
            frames.append(canvas)
        
        return np.array(frames)
    
    def _draw_digit(self, canvas, image, position):
        """Draw a digit on the canvas at the given position."""
        x, y = position
        x = int(x - self.digit_size // 2)
        y = int(y - self.digit_size // 2)
        
        # Convert image to numpy and scale to 0-255
        digit = image.squeeze().numpy()
        digit = (digit * 255).astype(np.uint8)
        
        # Draw digit on canvas - only draw non-zero pixels to avoid overwriting
        h, w = digit.shape
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(canvas.shape[1], x + w)
        y_end = min(canvas.shape[0], y + h)
        
        if x_start < x_end and y_start < y_end:
            # Extract the region of the digit that fits in the canvas
            digit_region = digit[y_start - y:y_end - y, x_start - x:x_end - x]
            
            # Only update pixels where the digit has non-zero values (white pixels)
            mask = digit_region > 0
            canvas[y_start:y_end, x_start:x_end][mask] = digit_region[mask]
    
    def _scale_video(self, video, target_resolution):
        """Scale video to target resolution."""
        from torch.nn.functional import interpolate
        
        # video shape: [T, C, H, W]
        T, C, H, W = video.shape
        
        # Scale shorter side to target resolution
        scale = target_resolution / min(H, W)
        if H < W:
            target_size = (target_resolution, int(W * scale))
        else:
            target_size = (int(H * scale), target_resolution)
        
        # Reshape for interpolation: [T*C, 1, H, W] -> [T*C, 1, H', W']
        video_reshaped = video.view(T * C, 1, H, W)
        video_scaled = interpolate(
            video_reshaped, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Center crop to target resolution
        _, _, h_scaled, w_scaled = video_scaled.shape
        h_start = (h_scaled - target_resolution) // 2
        w_start = (w_scaled - target_resolution) // 2
        
        video_cropped = video_scaled[:, :, h_start:h_start + target_resolution, 
                                   w_start:w_start + target_resolution]
        
        # Reshape back to [T, C, H, W] then permute to [C, T, H, W]
        video_final = video_cropped.view(T, C, target_resolution, target_resolution)
        video_final = video_final.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        return video_final
    
    @property
    def n_classes(self):
        """Number of classes (not applicable for Moving MNIST)."""
        return 0  # Moving MNIST doesn't use class conditioning


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
    import h5py
    
    # Create datasets
    train_dataset = MovingMNISTDataset(
        sequence_length=sequence_length,
        train=True,
        resolution=resolution,
        num_digits=num_digits,
        videos_per_digit=videos_per_digit
    )
    
    test_dataset = MovingMNISTDataset(
        sequence_length=sequence_length,
        train=False,
        resolution=resolution,
        num_digits=num_digits,
        videos_per_digit=max(1, videos_per_digit // 5)  # Smaller test set
    )
    
    # Collect all data
    print("Generating training data...")
    train_data = []
    train_idx = []
    current_idx = 0
    
    for i in range(len(train_dataset)):
        video = train_dataset[i]['video']  # [C, T, H, W]
        video_np = video.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
        video_np = (video_np + 0.5) * 255  # Convert back to 0-255 range
        video_np = video_np.astype(np.uint8)
        
        train_data.append(video_np)
        train_idx.append(current_idx)
        current_idx += len(video_np)
    
    print("Generating test data...")
    test_data = []
    test_idx = []
    current_idx = 0
    
    for i in range(len(test_dataset)):
        video = test_dataset[i]['video']  # [C, T, H, W]
        video_np = video.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
        video_np = (video_np + 0.5) * 255  # Convert back to 0-255 range
        video_np = video_np.astype(np.uint8)
        
        test_data.append(video_np)
        test_idx.append(current_idx)
        current_idx += len(video_np)
    
    # Convert to numpy arrays
    train_data = np.concatenate(train_data, axis=0)
    train_idx = np.array(train_idx, dtype=np.int64)
    test_data = np.concatenate(test_data, axis=0)
    test_idx = np.array(test_idx, dtype=np.int64)
    
    # Save to HDF5
    print(f"Saving dataset to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train_data', data=train_data, compression='gzip')
        f.create_dataset('train_idx', data=train_idx, compression='gzip')
        f.create_dataset('test_data', data=test_data, compression='gzip')
        f.create_dataset('test_idx', data=test_idx, compression='gzip')
        
        # Metadata
        f.attrs['sequence_length'] = sequence_length
        f.attrs['resolution'] = resolution
        f.attrs['num_digits'] = num_digits
        f.attrs['train_videos'] = len(train_idx)
        f.attrs['test_videos'] = len(test_idx)
    
    print(f"Moving MNIST dataset created successfully!")
    print(f"Training videos: {len(train_idx)}")
    print(f"Test videos: {len(test_idx)}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Sequence length: {sequence_length}")
    print(f"Digits per video: {num_digits}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Moving MNIST dataset')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output HDF5 file path')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Number of frames per video')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Target resolution for videos')
    parser.add_argument('--num_digits', type=int, default=2,
                       help='Number of digits per video')
    parser.add_argument('--videos_per_digit', type=int, default=1000,
                       help='Number of videos per digit class')
    
    args = parser.parse_args()
    
    create_moving_mnist_dataset(
        output_file=args.output_file,
        sequence_length=args.sequence_length,
        resolution=args.resolution,
        num_digits=args.num_digits,
        videos_per_digit=args.videos_per_digit
    )
