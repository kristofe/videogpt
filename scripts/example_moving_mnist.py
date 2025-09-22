#!/usr/bin/env python3
"""
Example script showing how to use Moving MNIST with VideoGPT.

This script demonstrates:
1. Creating a Moving MNIST dataset
2. Training VideoGPT on Moving MNIST
3. Generating samples from the trained model
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import videogpt modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogpt.moving_mnist import create_moving_mnist_dataset
from videogpt import VideoGPT, VideoData


def create_dataset_example():
    """Create a small Moving MNIST dataset for testing."""
    print("Creating Moving MNIST dataset...")
    
    output_file = "datasets/moving_mnist/moving_mnist_64x64.h5"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    create_moving_mnist_dataset(
        output_file=output_file,
        sequence_length=16,
        resolution=64,
        num_digits=2,
        videos_per_digit=100  # Small dataset for quick testing
    )
    
    return output_file


def visualize_dataset(dataset_path, num_samples=4):
    """Visualize some samples from the Moving MNIST dataset as animated GIFs."""
    print(f"Visualizing samples from {dataset_path}...")
    
    # Create dataset
    from videogpt.moving_mnist import MovingMNISTDataset as FullMovingMNISTDataset
    dataset = FullMovingMNISTDataset(
        sequence_length=16,
        train=True,
        resolution=64,
        num_digits=2,
        videos_per_digit=100
    )
    
    # Create animated GIFs for each sample
    for i in range(num_samples):
        sample = dataset[i]
        video = sample['video']  # [C, T, H, W]
        
        # Debug: print the shape to understand the issue
        print(f"Video {i} shape: {video.shape}")
        
        # Convert to [T, H, W] format for visualization
        if video.dim() == 4:  # [C, T, H, W]
            video = video.squeeze(0)  # [T, H, W]
        elif video.dim() == 3:  # [T, H, W] or [T, C, H, W] -> [T, H, W]
            if video.shape[1] == 1:  # [T, 1, H, W]
                video = video.squeeze(1)  # [T, H, W]
        
        print(f"Video {i} shape after processing: {video.shape}")
        
        # Prepare frames for GIF
        frames = []
        for t in range(16):
            frame = video[t].numpy()
            # Ensure frame is 2D for imshow
            if frame.ndim == 3 and frame.shape[0] == 1:
                frame = frame.squeeze(0)  # Remove channel dimension
            elif frame.ndim == 3:
                frame = frame.squeeze()  # Remove any singleton dimensions
            
            # Convert from [-0.5, 0.5] range to [0, 255] range
            frame = np.clip((frame + 0.5) * 255, 0, 255).astype(np.uint8)
            frames.append(frame)
        
        # Create animated GIF
        import imageio
        gif_filename = f'moving_mnist_sample_{i}.gif'
        imageio.mimsave(gif_filename, frames, duration=0.2)  # 0.2 seconds per frame
        print(f"Animated GIF saved as {gif_filename}")
    
    # Also create a combined visualization with still frames for comparison
    fig, axes = plt.subplots(num_samples, 16, figsize=(20, 5))
    
    for i in range(num_samples):
        sample = dataset[i]
        video = sample['video']  # [C, T, H, W]
        
        # Convert to [T, H, W] format for visualization
        if video.dim() == 4:  # [C, T, H, W]
            video = video.squeeze(0)  # [T, H, W]
        elif video.dim() == 3:  # [T, H, W] or [T, C, H, W] -> [T, H, W]
            if video.shape[1] == 1:  # [T, 1, H, W]
                video = video.squeeze(1)  # [T, H, W]
        
        for t in range(16):
            frame = video[t].numpy()
            # Ensure frame is 2D for imshow
            if frame.ndim == 3 and frame.shape[0] == 1:
                frame = frame.squeeze(0)  # Remove channel dimension
            elif frame.ndim == 3:
                frame = frame.squeeze()  # Remove any singleton dimensions
            axes[i, t].imshow(frame, cmap='gray')
            axes[i, t].set_title(f'Frame {t}')
            axes[i, t].axis('off')
    
    plt.tight_layout()
    plt.savefig('moving_mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Still frames saved to moving_mnist_samples.png")
    print(f"Animated GIFs saved as moving_mnist_sample_0.gif through moving_mnist_sample_{num_samples-1}.gif")


def train_example(dataset_path, max_steps=100):
    """Train VideoGPT on Moving MNIST (short training for demo)."""
    print(f"Training VideoGPT on Moving MNIST (max_steps={max_steps})...")
    
    # Create a simple args object
    class Args:
        def __init__(self):
            self.data_path = dataset_path
            self.dataset_type = 'moving_mnist'
            self.resolution = 64
            self.sequence_length = 16
            self.batch_size = 4
            self.num_workers = 2
            self.num_digits = 2
            self.videos_per_digit = 100
            self.class_cond = False
            self.class_cond_dim = None
            
            # VideoGPT model args
            self.embed_dim = 256
            self.n_heads = 8
            self.n_layers = 12
            self.attn_dropout = 0.1
            self.resid_dropout = 0.1
            self.embd_pdrop = 0.1
            
            # Training args
            self.learning_rate = 1e-4
            self.weight_decay = 0.01
            self.max_steps = max_steps
            self.val_check_interval = 50
            self.log_every_n_steps = 10
            
            # VQ-VAE args (simplified)
            self.vqvae_model = 'bair_stride4x2x2'
            self.vqvae_ckpt = None
            
            # Other args
            self.gpus = 1 if torch.cuda.is_available() else 0
            self.precision = 16 if torch.cuda.is_available() else 32
    
    args = Args()
    
    # Create data module
    data = VideoData(args)
    
    # Create model
    model = VideoGPT(args)
    
    # Create trainer
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    callbacks = [
        ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            filename='moving_mnist-{epoch:02d}-{val/loss:.2f}'
        )
    ]
    
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        gpus=args.gpus,
        precision=args.precision,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, data)
    
    return model


def generate_samples(model, num_samples=4):
    """Generate samples from the trained model."""
    print("Generating samples...")
    
    model.eval()
    with torch.no_grad():
        # Generate samples
        samples = model.sample(num_samples)
        
        # Create animated GIFs for each generated sample
        for i in range(num_samples):
            video = samples[i]  # [C, T, H, W]
            video = video.squeeze(0)  # [T, H, W]
            
            # Prepare frames for GIF
            frames = []
            for t in range(16):
                frame = video[t].numpy()
                # Ensure frame is 2D for imshow
                if frame.ndim == 3 and frame.shape[0] == 1:
                    frame = frame.squeeze(0)  # Remove channel dimension
                elif frame.ndim == 3:
                    frame = frame.squeeze()  # Remove any singleton dimensions
                
                # Convert from [-0.5, 0.5] range to [0, 255] range
                frame = np.clip((frame + 0.5) * 255, 0, 255).astype(np.uint8)
                frames.append(frame)
            
            # Create animated GIF
            import imageio
            gif_filename = f'moving_mnist_generated_{i}.gif'
            imageio.mimsave(gif_filename, frames, duration=0.2)  # 0.2 seconds per frame
            print(f"Generated animated GIF saved as {gif_filename}")
        
        # Also create a combined visualization with still frames for comparison
        fig, axes = plt.subplots(num_samples, 16, figsize=(20, 5))
        
        for i in range(num_samples):
            video = samples[i]  # [C, T, H, W]
            video = video.squeeze(0)  # [T, H, W]
            
            for t in range(16):
                frame = video[t].numpy()
                # Ensure frame is 2D for imshow
                if frame.ndim == 3 and frame.shape[0] == 1:
                    frame = frame.squeeze(0)  # Remove channel dimension
                elif frame.ndim == 3:
                    frame = frame.squeeze()  # Remove any singleton dimensions
                axes[i, t].imshow(frame, cmap='gray')
                axes[i, t].set_title(f'Frame {t}')
                axes[i, t].axis('off')
        
        plt.tight_layout()
        plt.savefig('moving_mnist_generated.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Generated still frames saved to moving_mnist_generated.png")
        print(f"Generated animated GIFs saved as moving_mnist_generated_0.gif through moving_mnist_generated_{num_samples-1}.gif")


def main():
    parser = argparse.ArgumentParser(description='Moving MNIST VideoGPT Example')
    parser.add_argument('--create_dataset', action='store_true',
                       help='Create Moving MNIST dataset')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize dataset samples')
    parser.add_argument('--train', action='store_true',
                       help='Train VideoGPT model')
    parser.add_argument('--generate', action='store_true',
                       help='Generate samples from trained model')
    parser.add_argument('--dataset_path', type=str, default='datasets/moving_mnist/moving_mnist_64x64.h5',
                       help='Path to Moving MNIST dataset')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum training steps')
    parser.add_argument('--model_path', type=str, default='lightning_logs/version_0/checkpoints/',
                       help='Path to trained model checkpoints')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.create_dataset:
        dataset_path = create_dataset_example()
        print(f"Dataset created at: {dataset_path}")
    
    if args.visualize:
        if not os.path.exists(args.dataset_path):
            print(f"Dataset not found at {args.dataset_path}. Creating it first...")
            create_dataset_example()
        visualize_dataset(args.dataset_path)
    
    if args.train:
        if not os.path.exists(args.dataset_path):
            print(f"Dataset not found at {args.dataset_path}. Creating it first...")
            create_dataset_example()
        model = train_example(args.dataset_path, args.max_steps)
        print("Training completed!")
    
    if args.generate:
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Please train first.")
            return
        
        # Load the trained model
        import pytorch_lightning as pl
        model = VideoGPT.load_from_checkpoint(args.model_path)
        generate_samples(model)


if __name__ == '__main__':
    main()
