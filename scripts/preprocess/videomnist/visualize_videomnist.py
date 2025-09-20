#!/usr/bin/env python3
"""
Visualize VideoMNIST dataset by creating sample video clips as GIFs and MP4s.

This script loads a VideoMNIST HDF5 dataset and creates sample video clips
showing the different motion patterns and digit classes.
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import cv2
from tqdm import tqdm


def frames_to_gif(frames, output_path, duration=200, loop=0):
    """Convert video frames to animated GIF."""
    # Normalize frames to 0-255 range
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)
    
    # Convert to PIL Images
    images = []
    for frame in frames:
        img = Image.fromarray(frame, mode='L')
        images.append(img)
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )


def frames_to_mp4(frames, output_path, fps=10):
    """Convert video frames to MP4 video."""
    # Normalize frames to 0-255 range
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)
    
    height, width = frames[0].shape
    
    # Try different codecs
    codecs = ['mp4v', 'XVID', 'MJPG']
    success = False
    
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
        
        if out.isOpened():
            for frame in frames:
                out.write(frame)
            out.release()
            success = True
            break
        else:
            out.release()
    
    if not success:
        print(f"Warning: Could not create MP4 video for {output_path}, trying GIF instead")
        # Fallback to GIF
        gif_path = output_path.replace('.mp4', '.gif')
        frames_to_gif(frames, gif_path, duration=int(1000/fps))
        return False
    
    return True


def create_montage_image(frames_list, labels, output_path, title="VideoMNIST Samples"):
    """Create a montage of multiple video sequences."""
    n_videos = len(frames_list)
    n_frames = len(frames_list[0])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_videos, figsize=(n_videos * 2, 4))
    if n_videos == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(title, fontsize=16)
    
    # Show first and last frame of each video
    for i, (frames, label) in enumerate(zip(frames_list, labels)):
        # First frame
        axes[0, i].imshow(frames[0], cmap='gray')
        axes[0, i].set_title(f'Digit {label} - Start')
        axes[0, i].axis('off')
        
        # Last frame
        axes[1, i].imshow(frames[-1], cmap='gray')
        axes[1, i].set_title(f'Digit {label} - End')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_dataset(hdf5_path, output_dir, num_samples=10, formats=['gif', 'mp4'], 
                     create_montage=True, fps=10, gif_duration=200):
    """Visualize VideoMNIST dataset by creating sample videos."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        train_data = f['train_data'][:]
        train_labels = f['train_labels'][:]
        train_idx = f['train_idx'][:]
        sequence_length = f.attrs['sequence_length']
        resolution = f.attrs['resolution']
    
    print(f"Dataset loaded: {len(train_idx)} videos, {sequence_length} frames each, {resolution}x{resolution} resolution")
    
    # Sample videos from each digit class
    samples_per_class = max(1, num_samples // 10)
    selected_videos = []
    selected_labels = []
    
    for digit in range(10):
        # Find videos of this digit
        digit_indices = np.where(train_labels == digit)[0]
        if len(digit_indices) > 0:
            # Sample random videos
            sample_indices = np.random.choice(digit_indices, 
                                            min(samples_per_class, len(digit_indices)), 
                                            replace=False)
            
            for idx in sample_indices:
                start_idx = train_idx[idx]
                end_idx = train_idx[idx + 1] if idx < len(train_idx) - 1 else len(train_data)
                video_frames = train_data[start_idx:end_idx]
                selected_videos.append(video_frames)
                selected_labels.append(digit)
    
    print(f"Selected {len(selected_videos)} videos for visualization")
    
    # Create individual video files
    for i, (frames, label) in enumerate(tqdm(zip(selected_videos, selected_labels), 
                                           desc="Creating videos")):
        base_name = f"digit_{label}_video_{i:03d}"
        
        if 'gif' in formats:
            gif_path = os.path.join(output_dir, f"{base_name}.gif")
            frames_to_gif(frames, gif_path, duration=gif_duration)
        
        if 'mp4' in formats:
            mp4_path = os.path.join(output_dir, f"{base_name}.mp4")
            frames_to_mp4(frames, mp4_path, fps=fps)
    
    # Create montage
    if create_montage:
        montage_path = os.path.join(output_dir, "videomnist_montage.png")
        create_montage_image(selected_videos[:min(10, len(selected_videos))], 
                           selected_labels[:min(10, len(selected_labels))], 
                           montage_path)
        print(f"Montage saved to {montage_path}")
    
    # Create summary statistics
    stats_path = os.path.join(output_dir, "dataset_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("VideoMNIST Dataset Statistics\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total videos: {len(train_idx)}\n")
        f.write(f"Sequence length: {sequence_length} frames\n")
        f.write(f"Resolution: {resolution}x{resolution}\n")
        f.write(f"Frames per second: {fps}\n")
        f.write(f"Video duration: {sequence_length/fps:.1f} seconds\n\n")
        
        f.write("Videos per digit class:\n")
        for digit in range(10):
            count = np.sum(train_labels == digit)
            f.write(f"  Digit {digit}: {count} videos\n")
        
        f.write(f"\nGenerated {len(selected_videos)} sample videos\n")
        f.write(f"Output formats: {', '.join(formats)}\n")
    
    print(f"Visualization complete! Files saved to {output_dir}")
    print(f"Generated {len(selected_videos)} sample videos in {', '.join(formats)} format")


def create_class_comparison(hdf5_path, output_dir, num_per_class=3):
    """Create a comparison showing different motion patterns for each digit class."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    with h5py.File(hdf5_path, 'r') as f:
        train_data = f['train_data'][:]
        train_labels = f['train_labels'][:]
        train_idx = f['train_idx'][:]
        sequence_length = f.attrs['sequence_length']
    
    # Create comparison for each digit class
    for digit in range(10):
        digit_indices = np.where(train_labels == digit)[0]
        if len(digit_indices) == 0:
            continue
        
        # Sample videos for this digit
        sample_indices = np.random.choice(digit_indices, 
                                        min(num_per_class, len(digit_indices)), 
                                        replace=False)
        
        videos = []
        for idx in sample_indices:
            start_idx = train_idx[idx]
            end_idx = train_idx[idx + 1] if idx < len(train_idx) - 1 else len(train_data)
            video_frames = train_data[start_idx:end_idx]
            videos.append(video_frames)
        
        # Create GIF showing different motion patterns
        gif_path = os.path.join(output_dir, f"digit_{digit}_comparison.gif")
        
        # Combine videos horizontally
        combined_frames = []
        for frame_idx in range(sequence_length):
            frame_row = []
            for video in videos:
                frame_row.append(video[frame_idx])
            combined_frame = np.hstack(frame_row)
            combined_frames.append(combined_frame)
        
        # Convert to numpy array
        combined_frames = np.array(combined_frames)
        frames_to_gif(combined_frames, gif_path, duration=200)
        print(f"Digit {digit} comparison saved to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize VideoMNIST dataset')
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='Path to VideoMNIST HDF5 file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualization files')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sample videos to generate')
    parser.add_argument('--formats', nargs='+', default=['gif', 'mp4'],
                       choices=['gif', 'mp4'],
                       help='Output formats for videos')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for MP4 videos')
    parser.add_argument('--gif_duration', type=int, default=200,
                       help='Duration per frame in GIF (milliseconds)')
    parser.add_argument('--create_montage', action='store_true', default=True,
                       help='Create montage image')
    parser.add_argument('--create_comparison', action='store_true',
                       help='Create class comparison GIFs')
    parser.add_argument('--num_per_class', type=int, default=3,
                       help='Number of videos per class for comparison')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create main visualization
    visualize_dataset(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        formats=args.formats,
        create_montage=args.create_montage,
        fps=args.fps,
        gif_duration=args.gif_duration
    )
    
    # Create class comparison if requested
    if args.create_comparison:
        comparison_dir = os.path.join(args.output_dir, "class_comparisons")
        create_class_comparison(
            hdf5_path=args.hdf5_path,
            output_dir=comparison_dir,
            num_per_class=args.num_per_class
        )


if __name__ == '__main__':
    main()
