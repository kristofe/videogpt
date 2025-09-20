#!/usr/bin/env python3
"""
Convert VideoMNIST videos to HDF5 format for VideoGPT training.

This script processes the generated VideoMNIST videos and converts them
to the HDF5 format expected by VideoGPT.
"""

import os
import argparse
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import glob
import pickle
from torchvision.datasets.video_utils import VideoClips


def extract_frames_from_video(video_path, sequence_length=16):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    
    cap.release()
    
    # Pad with last frame if needed
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    
    return np.array(frames[:sequence_length])


def process_video_dataset(data_dir, output_file, sequence_length=16, resolution=64):
    """Process video dataset and save to HDF5 format."""
    
    # Find all video files
    video_files = []
    labels = []
    
    for class_dir in sorted(glob.glob(os.path.join(data_dir, 'class_*'))):
        class_name = os.path.basename(class_dir)
        class_label = int(class_name.split('_')[1])
        
        class_videos = glob.glob(os.path.join(class_dir, '*.avi'))
        video_files.extend(class_videos)
        labels.extend([class_label] * len(class_videos))
    
    print(f"Found {len(video_files)} videos")
    
    # Extract all frames
    all_frames = []
    all_labels = []
    video_start_indices = []
    
    current_idx = 0
    
    for i, (video_path, label) in enumerate(tqdm(zip(video_files, labels), 
                                                desc="Processing videos")):
        try:
            frames = extract_frames_from_video(video_path, sequence_length)
            
            # Resize frames to target resolution
            resized_frames = []
            for frame in frames:
                resized = cv2.resize(frame, (resolution, resolution))
                resized_frames.append(resized)
            
            frames_array = np.array(resized_frames, dtype=np.uint8)
            
            # Store frames
            all_frames.append(frames_array)
            all_labels.append(label)
            video_start_indices.append(current_idx)
            current_idx += len(frames_array)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    # Convert to numpy arrays
    all_frames = np.concatenate(all_frames, axis=0)
    all_labels = np.array(all_labels, dtype=np.int64)
    video_start_indices = np.array(video_start_indices, dtype=np.int64)
    
    print(f"Total frames: {len(all_frames)}")
    print(f"Frame shape: {all_frames.shape}")
    print(f"Number of videos: {len(video_start_indices)}")
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=all_frames, compression='gzip')
        f.create_dataset('idx', data=video_start_indices, compression='gzip')
        f.create_dataset('labels', data=all_labels, compression='gzip')
        
        # Add metadata
        f.attrs['sequence_length'] = sequence_length
        f.attrs['resolution'] = resolution
        f.attrs['n_classes'] = len(set(all_labels))
        f.attrs['n_videos'] = len(video_start_indices)
    
    print(f"Dataset saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert VideoMNIST videos to HDF5')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing VideoMNIST videos')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output HDF5 file path')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Number of frames per video sequence')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Resolution of video frames')
    
    args = parser.parse_args()
    
    process_video_dataset(
        data_dir=args.data_dir,
        output_file=args.output_file,
        sequence_length=args.sequence_length,
        resolution=args.resolution
    )


if __name__ == '__main__':
    main()
