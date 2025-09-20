#!/usr/bin/env python3
"""
Convert VizDoom video dataset to HDF5 format for VideoGPT training.
This script processes the video files and creates the required HDF5 structure.
"""

import argparse
import os
import sys
import h5py
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import glob


def get_parent_dir(file_path):
    """Get the parent directory name of a file path."""
    return os.path.basename(os.path.dirname(file_path))


def load_video_frames(video_path, target_frames=16):
    """Load frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    
    cap.release()
    
    # Pad with last frame if needed
    while len(frames) < target_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            # Create black frame if no frames
            frames.append(np.zeros((64, 64), dtype=np.uint8))
    
    return np.array(frames[:target_frames])


def load_actions(actions_path, target_frames=16):
    """Load actions from a numpy file."""
    try:
        actions = np.load(actions_path)
        # Pad with last action if needed
        while len(actions) < target_frames:
            if len(actions) > 0:
                actions = np.vstack([actions, actions[-1:].copy()])
            else:
                # Create no-op action if no actions
                actions = np.array([[0, 0, 0, 0, 0, 0, 0]])
        return actions[:target_frames]
    except Exception as e:
        print(f"Error loading actions from {actions_path}: {e}")
        # Return no-op actions
        return np.tile([[0, 0, 0, 0, 0, 0, 0]], (target_frames, 1))


def process_video_dataset(data_dir, output_file, sequence_length=16, resolution=64):
    """Process video dataset and convert to HDF5 format."""
    data_dir = Path(data_dir)
    
    # Find all video files
    train_files = []
    test_files = []
    
    # Get training videos
    train_pattern = str(data_dir / "train" / "**" / "*.mp4")
    train_files = glob.glob(train_pattern, recursive=True)
    
    # Get test videos
    test_pattern = str(data_dir / "test" / "**" / "*.mp4")
    test_files = glob.glob(test_pattern, recursive=True)
    
    print(f"Found {len(train_files)} training videos and {len(test_files)} test videos")
    
    # Get class information
    train_classes = list(set([get_parent_dir(f) for f in train_files]))
    test_classes = list(set([get_parent_dir(f) for f in test_files]))
    all_classes = sorted(list(set(train_classes + test_classes)))
    
    class_to_label = {c: i for i, c in enumerate(all_classes)}
    print(f"Classes found: {all_classes}")
    
    # Process training data
    print("Processing training videos...")
    train_data = []
    train_actions = []
    train_idx = [0]
    train_labels = []
    
    for video_path in tqdm(train_files, desc="Training videos"):
        try:
            frames = load_video_frames(video_path, sequence_length)
            if frames.shape[0] == sequence_length:
                # Load corresponding actions
                video_name = Path(video_path).stem
                actions_path = data_dir / "train" / "actions" / f"{video_name}.npy"
                actions = load_actions(actions_path, sequence_length)
                
                train_data.append(frames)
                train_actions.append(actions)
                class_name = get_parent_dir(video_path)
                train_labels.append(class_to_label[class_name])
                train_idx.append(train_idx[-1] + sequence_length)
        except Exception as e:
            print(f"Error processing training video {video_path}: {e}")
            continue
    
    # Process test data
    print("Processing test videos...")
    test_data = []
    test_actions = []
    test_idx = [0]
    test_labels = []
    
    for video_path in tqdm(test_files, desc="Test videos"):
        try:
            frames = load_video_frames(video_path, sequence_length)
            if frames.shape[0] == sequence_length:
                # Load corresponding actions
                video_name = Path(video_path).stem
                actions_path = data_dir / "test" / "actions" / f"{video_name}.npy"
                actions = load_actions(actions_path, sequence_length)
                
                test_data.append(frames)
                test_actions.append(actions)
                class_name = get_parent_dir(video_path)
                test_labels.append(class_to_label[class_name])
                test_idx.append(test_idx[-1] + sequence_length)
        except Exception as e:
            print(f"Error processing test video {video_path}: {e}")
            continue
    
    # Convert to numpy arrays
    train_data = np.concatenate(train_data, axis=0) if train_data else np.array([])
    test_data = np.concatenate(test_data, axis=0) if test_data else np.array([])
    train_actions = np.concatenate(train_actions, axis=0) if train_actions else np.array([])
    test_actions = np.concatenate(test_actions, axis=0) if test_actions else np.array([])
    train_idx = np.array(train_idx[:-1])  # Remove last element (total count)
    test_idx = np.array(test_idx[:-1])
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Training actions shape: {train_actions.shape}")
    print(f"Test actions shape: {test_actions.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Save to HDF5
    print(f"Saving dataset to {output_file}")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train_data', data=train_data, compression='gzip')
        f.create_dataset('train_actions', data=train_actions, compression='gzip')
        f.create_dataset('train_idx', data=train_idx)
        f.create_dataset('train_labels', data=train_labels)
        f.create_dataset('test_data', data=test_data, compression='gzip')
        f.create_dataset('test_actions', data=test_actions, compression='gzip')
        f.create_dataset('test_idx', data=test_idx)
        f.create_dataset('test_labels', data=test_labels)
        
        # Add metadata
        f.attrs['sequence_length'] = sequence_length
        f.attrs['resolution'] = resolution
        f.attrs['n_classes'] = len(all_classes)
        f.attrs['class_names'] = [c.encode('utf-8') for c in all_classes]
        f.attrs['action_dim'] = 7  # Number of action dimensions
        f.attrs['action_names'] = [b'move_left', b'move_right', b'attack', b'move_forward', b'move_backward', b'turn_left', b'turn_right']
    
    print(f"Dataset saved to {output_file}")
    print(f"Training videos: {len(train_labels)}")
    print(f"Test videos: {len(test_labels)}")
    print(f"Classes: {all_classes}")


def main():
    parser = argparse.ArgumentParser(description="Convert VizDoom videos to HDF5 format")
    parser.add_argument("--data_dir", required=True, help="Directory containing train/test video folders")
    parser.add_argument("--output_file", required=True, help="Output HDF5 file path")
    parser.add_argument("--sequence_length", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--resolution", type=int, default=64, help="Video resolution")
    
    args = parser.parse_args()
    
    process_video_dataset(
        data_dir=args.data_dir,
        output_file=args.output_file,
        sequence_length=args.sequence_length,
        resolution=args.resolution
    )


if __name__ == "__main__":
    main()
