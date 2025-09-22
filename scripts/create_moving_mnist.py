#!/usr/bin/env python3
"""
Create Moving MNIST dataset for VideoGPT training.

This script generates a Moving MNIST dataset and saves it in HDF5 format
that's compatible with VideoGPT's data loading framework.
"""

import argparse
import os
import sys

# Add the parent directory to the path so we can import videogpt modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogpt.moving_mnist import create_moving_mnist_dataset


def main():
    parser = argparse.ArgumentParser(description='Create Moving MNIST dataset for VideoGPT')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output HDF5 file path')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Number of frames per video')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Target resolution for videos (will be scaled from 64x64)')
    parser.add_argument('--num_digits', type=int, default=2, choices=[1, 2],
                       help='Number of digits per video')
    parser.add_argument('--videos_per_digit', type=int, default=1000,
                       help='Number of videos per digit class')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data to use for training')
    
    args = parser.parse_args()
    
    print("Creating Moving MNIST dataset...")
    print(f"Output file: {args.output_file}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Digits per video: {args.num_digits}")
    print(f"Videos per digit: {args.videos_per_digit}")
    print(f"Train split: {args.train_split}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Generate the dataset
    create_moving_mnist_dataset(
        output_file=args.output_file,
        sequence_length=args.sequence_length,
        resolution=args.resolution,
        num_digits=args.num_digits,
        videos_per_digit=args.videos_per_digit,
        train_split=args.train_split
    )
    
    print(f"\nDataset created successfully at: {args.output_file}")
    print("\nTo use this dataset with VideoGPT, run:")
    print(f"python scripts/train_videogpt.py --data_path {args.output_file} --resolution {args.resolution}")


if __name__ == '__main__':
    main()
