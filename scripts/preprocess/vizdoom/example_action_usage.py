#!/usr/bin/env python3
"""
Example script showing how to use the action data from VizDoom dataset.
This demonstrates loading action sequences and using them for world model training.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_vizdoom_data(hdf5_path, video_idx=0, split='train'):
    """Load a single video and its actions from the VizDoom dataset."""
    with h5py.File(hdf5_path, 'r') as f:
        # Get video indices
        if split == 'train':
            start_idx = f['train_idx'][video_idx]
            end_idx = f['train_idx'][video_idx + 1] if video_idx + 1 < len(f['train_idx']) else start_idx + 16
            frames = f['train_data'][start_idx:end_idx]
            actions = f['train_actions'][start_idx:end_idx]
            label = f['train_labels'][video_idx]
        else:
            start_idx = f['test_idx'][video_idx]
            end_idx = f['test_idx'][video_idx + 1] if video_idx + 1 < len(f['test_idx']) else start_idx + 16
            frames = f['test_data'][start_idx:end_idx]
            actions = f['test_actions'][start_idx:end_idx]
            label = f['test_labels'][video_idx]
        
        # Get metadata
        action_names = [name.decode('utf-8') for name in f.attrs['action_names']]
        class_names = [name.decode('utf-8') for name in f.attrs['class_names']]
        
        return frames, actions, label, action_names, class_names


def visualize_actions(actions, action_names, title="Action Sequence"):
    """Visualize the action sequence over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a heatmap of actions over time
    im = ax.imshow(actions.T, cmap='Blues', aspect='auto')
    
    # Set labels
    ax.set_xlabel('Frame')
    ax.set_ylabel('Action')
    ax.set_title(title)
    ax.set_yticks(range(len(action_names)))
    ax.set_yticklabels(action_names)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Action Active (0/1)')
    
    plt.tight_layout()
    return fig


def analyze_action_patterns(actions, action_names):
    """Analyze patterns in the action sequences."""
    print("Action Analysis:")
    print("=" * 50)
    
    # Action frequency
    action_freq = np.mean(actions, axis=0)
    print("Action frequencies (0-1):")
    for i, (name, freq) in enumerate(zip(action_names, action_freq)):
        print(f"  {name}: {freq:.3f}")
    
    # Action combinations
    print(f"\nAction combinations:")
    unique_combinations = np.unique(actions, axis=0)
    print(f"  Total unique action combinations: {len(unique_combinations)}")
    
    # Most common combinations
    from collections import Counter
    action_tuples = [tuple(row) for row in actions]
    most_common = Counter(action_tuples).most_common(5)
    print(f"  Most common action combinations:")
    for i, (combo, count) in enumerate(most_common):
        action_str = " + ".join([action_names[j] for j, active in enumerate(combo) if active])
        if not action_str:
            action_str = "no action"
        print(f"    {i+1}. {action_str} ({count} times)")


def create_action_sequence_example():
    """Create an example of how to use actions for world model training."""
    print("World Model Training Example:")
    print("=" * 50)
    
    # Simulate loading data
    print("1. Loading video and action data...")
    # frames, actions, label, action_names, class_names = load_vizdoom_data('datasets/vizdoom/vizdoom.h5')
    
    # For demonstration, create dummy data
    frames = np.random.rand(16, 64, 64)  # Dummy frames
    actions = np.random.randint(0, 2, (16, 7))  # Dummy actions
    action_names = ['move_left', 'move_right', 'attack', 'move_forward', 'move_backward', 'turn_left', 'turn_right']
    
    print(f"   Frames shape: {frames.shape}")
    print(f"   Actions shape: {actions.shape}")
    
    print("\n2. Preparing data for world model training...")
    # Split into current frame + action -> next frame
    current_frames = frames[:-1]  # [15, 64, 64]
    next_frames = frames[1:]      # [15, 64, 64]
    current_actions = actions[:-1]  # [15, 7]
    
    print(f"   Current frames: {current_frames.shape}")
    print(f"   Next frames: {next_frames.shape}")
    print(f"   Current actions: {current_actions.shape}")
    
    print("\n3. World model training loop:")
    print("   for epoch in range(num_epochs):")
    print("       for i in range(len(current_frames)):")
    print("           # Get current frame and action")
    print("           frame = current_frames[i]")
    print("           action = current_actions[i]")
    print("           target = next_frames[i]")
    print("           ")
    print("           # Predict next frame")
    print("           predicted = world_model(frame, action)")
    print("           ")
    print("           # Compute loss and update model")
    print("           loss = mse_loss(predicted, target)")
    print("           loss.backward()")
    print("           optimizer.step()")


def main():
    """Main function demonstrating action data usage."""
    print("VizDoom Action Data Usage Example")
    print("=" * 50)
    
    # Check if dataset exists
    hdf5_path = Path("datasets/vizdoom/vizdoom.h5")
    if not hdf5_path.exists():
        print(f"Dataset not found at {hdf5_path}")
        print("Please run: sh scripts/preprocess/vizdoom/create_vizdoom_dataset.sh datasets/vizdoom")
        print("\nUsing dummy data for demonstration...")
        use_dummy_data = True
    else:
        use_dummy_data = False
    
    if not use_dummy_data:
        # Load real data
        try:
            frames, actions, label, action_names, class_names = load_vizdoom_data(str(hdf5_path))
            print(f"Loaded video from class: {class_names[label]}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using dummy data for demonstration...")
            use_dummy_data = True
    
    if use_dummy_data:
        # Create dummy data for demonstration
        frames = np.random.rand(16, 64, 64)
        actions = np.random.randint(0, 2, (16, 7))
        action_names = ['move_left', 'move_right', 'attack', 'move_forward', 'move_backward', 'turn_left', 'turn_right']
        class_names = ['basic', 'defend_the_center', 'defend_the_line', 'health_gathering']
    
    # Analyze actions
    analyze_action_patterns(actions, action_names)
    
    # Create action visualization
    print("\n4. Creating action visualization...")
    fig = visualize_actions(actions, action_names, "Example Action Sequence")
    
    # Save visualization
    output_dir = Path("vizdoom_examples")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "action_sequence.png", dpi=150, bbox_inches='tight')
    print(f"   Saved action visualization to {output_dir / 'action_sequence.png'}")
    
    # Show world model example
    print("\n" + "="*50)
    create_action_sequence_example()
    
    print(f"\nExample completed! Check {output_dir} for visualizations.")


if __name__ == "__main__":
    main()
