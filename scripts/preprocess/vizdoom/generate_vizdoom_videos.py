#!/usr/bin/env python3
"""
Generate VizDoom videos for VideoGPT training.
This script creates video sequences by running VizDoom scenarios and recording gameplay.
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path
import random
import time

try:
    import vizdoom as vzd
except ImportError:
    print("Error: VizDoom not installed. Please install with: pip install vizdoom")
    sys.exit(1)


def create_doom_game(scenario="basic", resolution=(64, 64)):
    """Create and configure a VizDoom game instance."""
    game = vzd.DoomGame()
    
    # Set basic configuration
    game.set_doom_scenario_path(vzd.scenarios_path + f"/{scenario}.wad")
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)
    
    # Set episode settings
    game.set_episode_timeout(3000)
    game.set_episode_start_time(10)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_living_reward(-1)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_ticrate(35)
    
    game.init()
    return game


def resize_frame(frame, target_size=(64, 64)):
    """Resize frame to target resolution."""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def generate_random_action():
    """Generate a random action for the agent."""
    return [random.choice([0, 1]) for _ in range(7)]  # 7 available buttons


def record_video_episode(game, output_path, sequence_length=16, resolution=(64, 64)):
    """Record a single video episode with actions."""
    frames = []
    actions = []
    game.new_episode()
    
    frame_count = 0
    while not game.is_episode_finished() and frame_count < sequence_length:
        state = game.get_state()
        if state is not None:
            frame = state.screen_buffer
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = resize_frame(frame, resolution)
            frames.append(frame)
            frame_count += 1
        
        # Perform random action and record it
        action = generate_random_action()
        actions.append(action)
        game.make_action(action)
    
    # Pad with last frame and action if needed
    while len(frames) < sequence_length:
        if frames:
            frames.append(frames[-1].copy())
        else:
            # Create black frame if no frames captured
            frames.append(np.zeros(resolution, dtype=np.uint8))
    
    while len(actions) < sequence_length:
        if actions:
            actions.append(actions[-1].copy())
        else:
            # Create no-op action if no actions captured
            actions.append([0, 0, 0, 0, 0, 0, 0])
    
    return np.array(frames[:sequence_length]), np.array(actions[:sequence_length])


def save_video_as_mp4(frames, output_path, fps=10):
    """Save frames as MP4 video."""
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    for frame in frames:
        out.write(frame)
    out.release()


def save_actions_as_npy(actions, output_path):
    """Save actions as numpy array."""
    np.save(output_path, actions)


def generate_vizdoom_dataset(output_dir, num_train_videos=1000, num_test_videos=200, 
                           sequence_length=16, resolution=(64, 64), scenarios=None):
    """Generate the complete VizDoom dataset."""
    if scenarios is None:
        scenarios = ["basic", "defend_the_center", "defend_the_line", "health_gathering"]
    
    # Create directory structure
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class directories based on scenarios
    for scenario in scenarios:
        (train_dir / f"class_{scenarios.index(scenario)}").mkdir(exist_ok=True)
        (test_dir / f"class_{scenarios.index(scenario)}").mkdir(exist_ok=True)
    
    # Create actions directories
    (train_dir / "actions").mkdir(exist_ok=True)
    (test_dir / "actions").mkdir(exist_ok=True)
    
    print(f"Generating {num_train_videos} training videos and {num_test_videos} test videos...")
    
    # Generate training videos
    print("Generating training videos...")
    for i in range(num_train_videos):
        scenario = random.choice(scenarios)
        class_id = scenarios.index(scenario)
        
        try:
            game = create_doom_game(scenario, resolution)
            frames, actions = record_video_episode(game, None, sequence_length, resolution)
            game.close()
            
            # Save video as MP4
            video_path = train_dir / f"class_{class_id}" / f"video_{i:04d}.mp4"
            save_video_as_mp4(frames, str(video_path))
            
            # Save actions as NPY
            actions_path = train_dir / "actions" / f"video_{i:04d}.npy"
            save_actions_as_npy(actions, str(actions_path))
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_train_videos} training videos")
                
        except Exception as e:
            print(f"Error generating training video {i}: {e}")
            continue
    
    # Generate test videos
    print("Generating test videos...")
    for i in range(num_test_videos):
        scenario = random.choice(scenarios)
        class_id = scenarios.index(scenario)
        
        try:
            game = create_doom_game(scenario, resolution)
            frames, actions = record_video_episode(game, None, sequence_length, resolution)
            game.close()
            
            # Save video as MP4
            video_path = test_dir / f"class_{class_id}" / f"video_{i:04d}.mp4"
            save_video_as_mp4(frames, str(video_path))
            
            # Save actions as NPY
            actions_path = test_dir / "actions" / f"video_{i:04d}.npy"
            save_actions_as_npy(actions, str(actions_path))
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{num_test_videos} test videos")
                
        except Exception as e:
            print(f"Error generating test video {i}: {e}")
            continue
    
    print("VizDoom dataset generation completed!")


def main():
    parser = argparse.ArgumentParser(description="Generate VizDoom videos for VideoGPT")
    parser.add_argument("--output_dir", required=True, help="Output directory for videos")
    parser.add_argument("--num_train_videos", type=int, default=1000, help="Number of training videos")
    parser.add_argument("--num_test_videos", type=int, default=200, help="Number of test videos")
    parser.add_argument("--sequence_length", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--resolution", type=int, default=64, help="Video resolution")
    parser.add_argument("--scenarios", nargs="+", 
                       default=["basic", "defend_the_center", "defend_the_line", "health_gathering"],
                       help="VizDoom scenarios to use")
    
    args = parser.parse_args()
    
    generate_vizdoom_dataset(
        output_dir=args.output_dir,
        num_train_videos=args.num_train_videos,
        num_test_videos=args.num_test_videos,
        sequence_length=args.sequence_length,
        resolution=(args.resolution, args.resolution),
        scenarios=args.scenarios
    )


if __name__ == "__main__":
    main()
