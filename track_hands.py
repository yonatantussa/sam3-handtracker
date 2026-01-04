#!/usr/bin/env python3
"""Track hands on a limited number of frames to avoid OOM."""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from PIL import Image
from samgeo import SamGeo3Video

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Track hands through video frames using SAM3')
    parser.add_argument('--max-frames', type=int, default=1000,
                        help='Maximum number of frames to process (default: 1000)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='Ending frame index, None = start+max_frames (default: None)')
    parser.add_argument('--frames-dir', type=str, default='../frames_preview',
                        help='Directory containing video frames (default: ../frames_preview)')
    parser.add_argument('--coords-file', type=str, default='hand_coords.json',
                        help='JSON file with hand coordinates (default: hand_coords.json)')
    parser.add_argument('--output-dir', type=str, default='output/masks',
                        help='Output directory for masks (default: output/masks)')
    parser.add_argument('--temp-dir', type=str, default='frames_temp',
                        help='Temporary directory for frame copies (default: frames_temp)')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Configuration
MAX_FRAMES = args.max_frames
FRAMES_DIR = args.frames_dir
TEMP_DIR = args.temp_dir
OUTPUT_DIR = args.output_dir
COORDS_FILE = args.coords_file

# Load coordinates
with open(COORDS_FILE, 'r') as f:
    coords = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Pre-calculate frame range for display
all_frames_list = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
start_idx = args.start_frame
if args.end_frame is not None:
    end_idx = min(args.end_frame, len(all_frames_list))
else:
    end_idx = min(start_idx + MAX_FRAMES, len(all_frames_list))

print("\n" + "=" * 70)
print("SAM3 Hand Tracking (Limited)")
print("=" * 70)
print(f"Processing frames {start_idx} to {end_idx-1} ({end_idx-start_idx} total)")
print(f"Right hand: {len(coords['right'])} points")
print(f"Left hand: {len(coords['left'])} points")
print("=" * 70 + "\n")

# Check if masks already exist
if os.path.exists(OUTPUT_DIR):
    existing_masks = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    if existing_masks:
        print(f"WARNING:  WARNING: Found {len(existing_masks)} existing masks in {OUTPUT_DIR}/")
        
        # Handle non-interactive mode
        if not sys.stdin.isatty():
            print("Running in non-interactive mode, skipping existing masks.")
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
            exit(0)
        
        response = input("Delete and rerun tracking? (y/N): ")
        if response.lower() != 'y':
            print("Skipping tracking. Keeping existing masks.")
            # Clean up temp dir and exit
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
            exit(0)
        else:
            print(f"Deleting {len(existing_masks)} existing masks...")
            for f in existing_masks:
                os.remove(os.path.join(OUTPUT_DIR, f))
            print("> Deleted existing masks\n")

# Copy frames in the specified range to temp directory
all_frames = all_frames_list  # Already sorted and loaded
selected_frames = all_frames[start_idx:end_idx]
print(f"Copying {len(selected_frames)} frames to temp directory...")
for frame_file in selected_frames:
    shutil.copy(os.path.join(FRAMES_DIR, frame_file), os.path.join(TEMP_DIR, frame_file))
print("> Frames copied\n")

# Initialize SAM3
print("Initializing SAM3...")
sam = SamGeo3Video()
sam.set_video(TEMP_DIR)
sam.init_tracker()
print("> Initialized\n")

# Determine mode
mode = coords.get('mode', 'points')  # Default to points for backward compatibility

# Add prompts for right hand
if coords['right']:
    print(f"Adding right hand prompts ({mode})...")
    if mode == 'boxes':
        # Box prompt: convert from [x_min, y_min, x_max, y_max] to normalized [x, y, w, h]
        # SAM3 expects [xmin, ymin, width, height] in normalized coords (0-1 range)
        # It will convert to center format internally
        x_min, y_min, x_max, y_max = coords['right']
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize to 0-1 range
        norm_x = x_min / 1408
        norm_y = y_min / 1408
        norm_w = width / 1408
        norm_h = height / 1408
        
        # Call predictor directly (samgeo has a bug - it passes 'box' instead of 'bounding_boxes')
        import torch
        box_tensor = torch.tensor([[norm_x, norm_y, norm_w, norm_h]], dtype=torch.float32)  # 2D: [1, 4]
        sam.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=sam.session_id,
                frame_index=0,
                bounding_boxes=box_tensor,
                bounding_box_labels=torch.tensor([1], dtype=torch.int32),  # Label 1 = foreground
                obj_id=1,
            )
        )
    else:
        # Point prompts
        sam.add_point_prompts(
            points=coords['right'],
            labels=[1] * len(coords['right']),
            obj_id=1,
            frame_idx=0
        )
    print("> Right hand prompts added")

# Add prompts for left hand
if coords['left']:
    print(f"Adding left hand prompts ({mode})...")
    if mode == 'boxes':
        # Box prompt: convert from [x_min, y_min, x_max, y_max] to normalized [x, y, w, h]
        # SAM3 expects [xmin, ymin, width, height] in normalized coords (0-1 range)
        # It will convert to center format internally
        x_min, y_min, x_max, y_max = coords['left']
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize to 0-1 range
        norm_x = x_min / 1408
        norm_y = y_min / 1408
        norm_w = width / 1408
        norm_h = height / 1408
        
        # Call predictor directly (samgeo has a bug - it passes 'box' instead of 'bounding_boxes')
        import torch
        box_tensor = torch.tensor([[norm_x, norm_y, norm_w, norm_h]], dtype=torch.float32)  # 2D: [1, 4]
        sam.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=sam.session_id,
                frame_index=0,
                bounding_boxes=box_tensor,
                bounding_box_labels=torch.tensor([1], dtype=torch.int32),  # Label 1 = foreground
                obj_id=2,
            )
        )
    else:
        # Point prompts
        sam.add_point_prompts(
            points=coords['left'],
            labels=[1] * len(coords['left']),
            obj_id=2,
            frame_idx=0
        )
    print("> Left hand prompts added\n")

# Propagate
print("Propagating masks through video...")
sam.propagate()
print("> Propagation complete\n")

# Save masks
print("Saving masks...")
sam.save_masks(OUTPUT_DIR)
print(f"> Masks saved to {OUTPUT_DIR}")

# Check mask content
mask_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
if mask_files:
    first_mask = np.array(Image.open(os.path.join(OUTPUT_DIR, mask_files[0])))
    last_mask = np.array(Image.open(os.path.join(OUTPUT_DIR, mask_files[-1])))
    print(f"\nFirst mask unique values: {np.unique(first_mask)}")
    print(f"Last mask unique values: {np.unique(last_mask)}")
    print(f"Total mask files: {len(mask_files)}")

# Cleanup
sam.close()
shutil.rmtree(TEMP_DIR)
print("\n> Done!")
print(f"Processed {len(mask_files)} frames successfully")
