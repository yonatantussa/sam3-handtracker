#!/usr/bin/env python3
"""
Body tracking pipeline using SAM 3D Body.
Runs: tracking -> visualization -> analysis
"""

import subprocess
import sys
import os
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
    """Run the complete body tracking pipeline."""
    print_header("SAM 3D Body Tracking Pipeline")

    print(f"{BOLD}This pipeline will:{RESET}")
    print("  1. Track bodies using SAM 3D Body (automatic detection)")
    print("  2. Generate 2D masks from 3D mesh projections")
    print("  3. Analyze frames to identify body presence")
    print("  4. Create visualizations")
    print()

    # Ask for frame range
    print(f"{YELLOW}Frame range to process:{RESET}")
    print("  Start frame (default: 0)")
    print("  Max frames (default: 1000)")
    print()

    start_frame_input = input(f"{YELLOW}Start frame (press Enter for 0): {RESET}").strip()
    max_frames_input = input(f"{YELLOW}Max frames (press Enter for 1000): {RESET}").strip()

    start_frame = 0  # default
    max_frames = 1000  # default

    if start_frame_input:
        try:
            start_frame = int(start_frame_input)
        except ValueError:
            print_error("Invalid number, using default 0")
            start_frame = 0

    if max_frames_input:
        try:
            max_frames = int(max_frames_input)
        except ValueError:
            print_error("Invalid number, using default 1000")
            max_frames = 1000

    print_info(f"Will process frames {start_frame} to {start_frame + max_frames - 1} ({max_frames} total)")
    print()

    # Step 1: Track bodies
    print_header("STEP 1: Track Bodies with SAM 3D Body")
    print_info(f"Processing frames {start_frame} to {start_frame + max_frames - 1}")
    print_info("This will automatically detect bodies and generate 3D meshes")
    print_info("No manual labeling required!")
    print()

    tracking_cmd = [
        "python", "track_body.py",
        "--start-frame", str(start_frame),
        "--max-frames", str(max_frames)
    ]

    if not run_command(tracking_cmd, f"Body tracking ({max_frames} frames)"):
        print_error("Tracking failed. Exiting.")
        return 1

    # Check if masks were created
    masks_dir = Path("output/body_masks")
    if not masks_dir.exists() or not list(masks_dir.glob("*.png")):
        print_error("No masks generated. Tracking may have failed.")
        return 1

    mask_count = len(list(masks_dir.glob("*.png")))
    print_success(f"Generated {mask_count} masks")

    # Step 2: Analyze frames
    print_header("STEP 2: Analyze Body Presence")
    print_info("Identifying frames with visible bodies")
    print()

    if not run_command(
        ["python", "analyze_body_frames.py"],
        "Frame analysis"
    ):
        print_error("Analysis failed. Continuing anyway...")

    # Step 3: Visualize
    print_header("STEP 3: Visualize Results")
    print_info("Creating colored overlays (cyan=body)")
    print()

    vis_cmd = [
        "python", "visualize_masks.py",
        "--start-frame", str(start_frame),
        "--masks-dir", "output/body_masks",
        "--output-dir", "output/body_visualizations"
    ]

    if not run_command(vis_cmd, "Visualization"):
        print_error("Visualization failed. Continuing anyway...")

    # Check results
    vis_dir = Path("output/body_visualizations")
    if vis_dir.exists():
        vis_count = len(list(vis_dir.glob("*.jpg")))
        print_success(f"Generated {vis_count} visualizations")

    # Success!
    print_header("PIPELINE COMPLETE")
    print(f"{GREEN}{BOLD}Results:{RESET}")
    print(f"  Body masks: output/body_masks/")
    print(f"  Frame analysis: body_frames.json")
    print(f"  Visualizations: output/body_visualizations/")
    print()
    print(f"{YELLOW}Next steps:{RESET}")
    print("  - Check body_frames.json for frames with visible bodies")
    print("  - Use masks in output/body_masks/ for diffusion inpainting")
    print("  - Review visualizations to verify detection quality")
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
