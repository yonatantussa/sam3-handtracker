#!/usr/bin/env python3
"""
Simple pipeline script: Run the complete hand tracking workflow.
Runs: label_hands.py -> track_hands.py -> visualize_masks.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    """Print colored header."""
    print(f"\n{BLUE}{BOLD}{'=' * 70}")
    print(f"{text}")
    print(f"{'=' * 70}{RESET}\n")

def print_success(text):
    """Print success message."""
    print(f"{GREEN}> {text}{RESET}")

def print_error(text):
    """Print error message."""
    print(f"{RED}ERROR: {text}{RESET}")

def print_info(text):
    """Print info message."""
    print(f"{YELLOW}> {text}{RESET}")

def run_command(cmd, description):
    """Run a command and check if it succeeded."""
    print_info(f"Running: {description}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print_error(f"Failed: {description}")
        return False
    
    print_success(f"Completed: {description}")
    return True

def main():
    """Run the complete pipeline."""
    print_header("SAM3 Hand Tracking Pipeline")
    
    print(f"{BOLD}This pipeline will:{RESET}")
    print("  1. Label hands (interactive - you click points)")
    print("  2. Track hands through video")
    print("  3. Generate visualizations")
    print()
    
    # Check if we should re-label
    coords_file = Path("hand_coords.json")
    skip_labeling = False
    
    if coords_file.exists():
        print_info("Found existing hand_coords.json")
        response = input(f"{YELLOW}Skip labeling and use existing coordinates? (Y/n): {RESET}")
        if response.lower() != 'n':
            skip_labeling = True
            print_success("Using existing coordinates")
    
    # Step 1: Label hands
    if not skip_labeling:
        print_header("STEP 1: Label Hands")
        print_info("Instructions:")
        print("  Click points on the RIGHT hand (shown in green)")
        print("  Press SPACEBAR to switch to LEFT hand")
        print("  Click points on the LEFT hand (shown in red)")
        print("  Press ENTER when done")
        print()
        input(f"{YELLOW}Press Enter to start labeling...{RESET}")
        
        if not run_command(
            ["python", "label_hands.py"],
            "Hand labeling"
        ):
            print_error("Labeling failed. Exiting.")
            return 1
        
        if not coords_file.exists():
            print_error("hand_coords.json not created. Labeling may have failed.")
            return 1
    
    # Step 2: Track hands
    print_header("STEP 2: Track Hands")
    print_info("This will track hands through video frames")
    
    # Ask for max frames
    print()
    print(f"{YELLOW}How many frames to process?{RESET}")
    print("  Default: 1000 frames (recommended, ~30-40GB RAM)")
    print("  Custom: Specify a number")
    print()

    max_frames_input = input(f"{YELLOW}Max frames (press Enter for 1000): {RESET}").strip()

    max_frames = 1000  # default

    if max_frames_input:
        try:
            max_frames = int(max_frames_input)
        except ValueError:
            print_error("Invalid number, using default 1000")
            max_frames = 1000

    print_info(f"Processing {max_frames} frames")
    print()

    if not run_command(
        ["python", "track_hands.py", "--max-frames", str(max_frames)],
        f"Hand tracking ({max_frames} frames)"
    ):
        print_error("Tracking failed. Exiting.")
        return 1
    
    # Check if masks were created
    masks_dir = Path("output/masks")
    if not masks_dir.exists() or not list(masks_dir.glob("*.png")):
        print_error("No masks generated. Tracking may have failed.")
        return 1
    
    mask_count = len(list(masks_dir.glob("*.png")))
    print_success(f"Generated {mask_count} masks")
    
    # Step 3: Visualize
    print_header("STEP 3: Visualize Results")
    print_info("Creating colored overlays (green=right hand, red=left hand)")
    print()
    
    if not run_command(
        ["python", "visualize_masks.py"],
        "Visualization"
    ):
        print_error("Visualization failed. Exiting.")
        return 1
    
    # Check results
    vis_dir = Path("output/visualizations")
    if vis_dir.exists():
        vis_count = len(list(vis_dir.glob("*.jpg")))
        print_success(f"Generated {vis_count} visualizations")
    
    # Success!
    print_header("PIPELINE COMPLETE")
    print(f"{GREEN}{BOLD}Results:{RESET}")
    print(f"  Masks: {masks_dir}/")
    print(f"  Visualizations: {vis_dir}/")
    print()
    print(f"{YELLOW}To view results:{RESET}")
    print(f"  ls {masks_dir}/")
    print(f"  ls {vis_dir}/")
    print()
    print(f"{YELLOW}To process more frames:{RESET}")
    print(f"  python run_pipeline.py")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_error("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
