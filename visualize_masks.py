#!/usr/bin/env python3
"""Visualize hand masks overlaid on original frames."""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize hand masks on frames')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--frames-dir', type=str, default='../frames_preview',
                        help='Directory containing video frames (default: ../frames_preview)')
    parser.add_argument('--masks-dir', type=str, default='output/masks',
                        help='Directory containing masks (default: output/masks)')
    parser.add_argument('--output-dir', type=str, default='output/visualizations',
                        help='Output directory for visualizations (default: output/visualizations)')
    return parser.parse_args()

args = parse_args()

FRAMES_DIR = args.frames_dir
MASKS_DIR = args.masks_dir
OUTPUT_DIR = args.output_dir

# Check if output directory already has visualizations
if os.path.exists(OUTPUT_DIR):
    existing_vis = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
    if existing_vis:
        print(f"\nWARNING:  WARNING: Found {len(existing_vis)} existing visualizations in {OUTPUT_DIR}/")
        
        # Handle non-interactive mode
        if not sys.stdin.isatty():
            print("Running in non-interactive mode, skipping existing visualizations.")
            exit(0)
        
        response = input("Delete and recreate visualizations? (y/N): ")
        if response.lower() != 'y':
            print("Skipping visualization. Keeping existing files.")
            exit(0)
        else:
            print(f"Deleting {len(existing_vis)} existing visualizations...")
            for f in existing_vis:
                os.remove(os.path.join(OUTPUT_DIR, f))
            print("> Deleted existing visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get list of masks and frames
mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith('.png')])
all_frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])

# Select frames starting from the specified index
start_idx = args.start_frame
frame_files = all_frame_files[start_idx:start_idx + len(mask_files)]

print(f"\nVisualizing {len(mask_files)} masks...")
print(f"  Masks: {len(mask_files)} files")
print(f"  Frames: Using indices {start_idx} to {start_idx + len(mask_files) - 1}")
print(f"  Total available frames: {len(all_frame_files)}")

if len(frame_files) < len(mask_files):
    print(f"\nERROR: Not enough frames! Need {len(mask_files)}, but only {len(frame_files)} available from index {start_idx}")
    exit(1)

# Create colormap for hands
colors = {
    0: [0, 0, 0],        # Background: black
    1: [0, 255, 0],      # Right hand: green
    2: [0, 0, 255]       # Left hand: red
}

for i, (mask_file, frame_file) in enumerate(zip(mask_files, frame_files)):
    # Load mask and frame
    mask = np.array(Image.open(os.path.join(MASKS_DIR, mask_file)))
    frame = cv2.imread(os.path.join(FRAMES_DIR, frame_file))
    
    # Create colored mask
    colored_mask = np.zeros_like(frame)
    for obj_id, color in colors.items():
        colored_mask[mask == obj_id] = color
    
    # Blend with frame
    alpha = 0.5
    overlay = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
    
    # Save visualization
    output_path = os.path.join(OUTPUT_DIR, f"vis_{i:04d}.jpg")
    cv2.imwrite(output_path, overlay)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(mask_files)}")

print(f"\n> Saved visualizations to {OUTPUT_DIR}/")
print(f"  Total files: {len(mask_files)}")
print(f"\nTo view:")
print(f"  ls {OUTPUT_DIR}/")
print(f"  # Open any vis_XXXX.jpg file to see the masks overlaid")
