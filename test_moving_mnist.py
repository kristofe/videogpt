#!/usr/bin/env python3
"""
Simple test script for Moving MNIST dataset without full VideoGPT dependencies.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_moving_mnist_basic():
    """Test basic Moving MNIST functionality without PyTorch dependencies."""
    print("Testing Moving MNIST basic functionality...")
    
    # Test the core generation logic
    from videogpt.moving_mnist import MovingMNISTDataset
    
    try:
        # Create a small dataset
        dataset = MovingMNISTDataset(
            sequence_length=8,  # Shorter for testing
            train=True,
            resolution=32,  # Smaller resolution for testing
            num_digits=1,  # Single digit for simplicity
            videos_per_digit=10  # Very small dataset
        )
        
        print(f"Dataset created successfully with {len(dataset)} videos")
        
        # Test getting a sample
        sample = dataset[0]
        video = sample['video']
        print(f"Sample video shape: {video.shape}")
        print(f"Video value range: [{video.min():.3f}, {video.max():.3f}]")
        
        # Test scaling
        dataset_64 = MovingMNISTDataset(
            sequence_length=8,
            train=True,
            resolution=64,  # Test scaling
            num_digits=1,
            videos_per_digit=10
        )
        
        sample_64 = dataset_64[0]
        video_64 = sample_64['video']
        print(f"Scaled video shape: {video_64.shape}")
        
        print("✅ Basic Moving MNIST test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Moving MNIST test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization of Moving MNIST samples."""
    print("\nTesting Moving MNIST visualization...")
    
    try:
        from videogpt.moving_mnist import MovingMNISTDataset
        
        # Create dataset
        dataset = MovingMNISTDataset(
            sequence_length=8,
            train=True,
            resolution=32,
            num_digits=1,
            videos_per_digit=4
        )
        
        # Create visualization
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        
        for i in range(2):
            sample = dataset[i]
            video = sample['video'].squeeze(0)  # [T, H, W]
            
            for t in range(8):
                # Ensure we have a 2D array for imshow
                frame = video[t].numpy()
                if len(frame.shape) == 3:
                    frame = frame.squeeze()  # Remove any extra dimensions
                axes[i, t].imshow(frame, cmap='gray')
                axes[i, t].set_title(f'Frame {t}')
                axes[i, t].axis('off')
        
        plt.tight_layout()
        plt.savefig('moving_mnist_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization test passed! Saved to moving_mnist_test.png")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scaling():
    """Test resolution scaling functionality."""
    print("\nTesting resolution scaling...")
    
    try:
        from videogpt.moving_mnist import MovingMNISTDataset
        
        # Test different resolutions
        resolutions = [32, 64, 128]
        
        for res in resolutions:
            dataset = MovingMNISTDataset(
                sequence_length=4,
                train=True,
                resolution=res,
                num_digits=1,
                videos_per_digit=2
            )
            
            sample = dataset[0]
            video = sample['video']
            # The dataset returns [C, T, H, W] format
            expected_shape = (1, 4, res, res)
            
            if video.shape == expected_shape:
                print(f"✅ Resolution {res}x{res}: {video.shape}")
            else:
                print(f"❌ Resolution {res}x{res}: Expected {expected_shape}, got {video.shape}")
                return False
        
        print("✅ Scaling test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Scaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running Moving MNIST integration tests...\n")
    
    tests = [
        test_moving_mnist_basic,
        test_visualization,
        test_scaling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Moving MNIST integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == '__main__':
    main()

