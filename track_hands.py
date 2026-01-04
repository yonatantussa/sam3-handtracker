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
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--max-frames', type=int, default=1000,
                        help='Maximum number of frames to process (default: 1000)')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='Ending frame index (exclusive, overrides --max-frames if set)')
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
START_FRAME = args.start_frame
MAX_FRAMES = args.max_frames
END_FRAME = args.end_frame
FRAMES_DIR = args.frames_dir
TEMP_DIR = args.temp_dir
OUTPUT_DIR = args.output_dir
COORDS_FILE = args.coords_file

# Load coordinates
with open(COORDS_FILE, 'r') as f:
    coords = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print("\n" + "=" * 70)
print("SAM3 Hand Tracking (Limited)")
print("=" * 70)
if END_FRAME is not None:
    num_frames = END_FRAME - START_FRAME
    print(f"Processing frames {START_FRAME} to {END_FRAME-1} ({num_frames} frames)")
else:
    print(f"Processing {MAX_FRAMES} frames starting from frame {START_FRAME}")
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

# Copy frames to temp directory based on start/end range
all_frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])

if END_FRAME is not None:
    frames_to_process = all_frames[START_FRAME:END_FRAME]
else:
    frames_to_process = all_frames[START_FRAME:START_FRAME + MAX_FRAMES]

print(f"Copying {len(frames_to_process)} frames to temp directory...")
for frame_file in frames_to_process:
    shutil.copy(os.path.join(FRAMES_DIR, frame_file), os.path.join(TEMP_DIR, frame_file))
print("> Frames copied\n")

# Initialize SAM3
print("Initializing SAM3...")
sam = SamGeo3Video()
sam.set_video(TEMP_DIR)
sam.init_tracker()
print("> Initialized\n")

# Add prompts for right hand
if coords['right']:
    print(f"Adding right hand prompts...")
    sam.add_point_prompts(
        points=coords['right'],
        labels=[1] * len(coords['right']),
        obj_id=1,
        frame_idx=0
    )
    print("> Right hand prompts added")

# Add prompts for left hand
if coords['left']:
    print(f"Adding left hand prompts...")
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
