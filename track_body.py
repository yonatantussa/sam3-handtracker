#!/usr/bin/env python3
"""
Track bodies using SAM 3D Body and project 3D meshes to 2D masks.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

# SAM 3D Body imports
try:
    from notebook.utils import setup_sam_3d_body
except ImportError:
    print("ERROR: SAM 3D Body not found. Please install:")
    print("  git clone https://github.com/facebookresearch/sam-3d-body.git")
    print("  cd sam-3d-body && pip install -e .")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Track bodies using SAM 3D Body')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--max-frames', type=int, default=1000,
                        help='Maximum number of frames to process (default: 1000)')
    parser.add_argument('--frames-dir', type=str, default='../frames_preview',
                        help='Directory containing video frames (default: ../frames_preview)')
    parser.add_argument('--output-dir', type=str, default='output/body_masks',
                        help='Output directory for masks (default: output/body_masks)')
    parser.add_argument('--model', type=str, default='facebook/sam-3d-body-dinov3',
                        help='Hugging Face model repo (default: facebook/sam-3d-body-dinov3)')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help='Threshold for mask projection (default: 0.5)')
    return parser.parse_args()


def project_mesh_to_mask(outputs, img_shape, threshold=0.5):
    """
    Project 3D mesh to 2D binary mask.

    Args:
        outputs: SAM 3D Body output dict with mesh/vertices
        img_shape: Tuple of (height, width)
        threshold: Threshold for creating binary mask

    Returns:
        Binary mask (H, W) with 0=background, 1=body, or None if detection failed
    """
    # Check if body was detected
    if outputs is None or 'vertices' not in outputs:
        return None

    height, width = img_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Get 2D keypoints if available (faster than full mesh projection)
    if 'keypoints_2d' in outputs:
        kpts = outputs['keypoints_2d']

        # Create convex hull from keypoints
        if kpts is not None and len(kpts) > 0:
            # Filter valid keypoints
            valid_kpts = kpts[kpts[:, 2] > threshold][:, :2]  # confidence threshold

            if len(valid_kpts) >= 3:
                # Create filled polygon from keypoints
                hull = cv2.convexHull(valid_kpts.astype(np.int32))
                cv2.fillPoly(mask, [hull], 1)
                return mask

    # Fallback: Try to get rendered mask if available
    if 'mask' in outputs:
        return (outputs['mask'] > threshold).astype(np.uint8)

    # If no 2D representation available, return None (failed detection)
    return None


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAM 3D Body Tracking")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Frames: {args.start_frame} to {args.start_frame + args.max_frames - 1}")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")

    # Check if masks already exist
    if os.path.exists(args.output_dir):
        existing_masks = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
        if existing_masks:
            print(f"WARNING: Found {len(existing_masks)} existing masks in {args.output_dir}/")

            if not sys.stdin.isatty():
                print("Running in non-interactive mode, skipping existing masks.")
                return 0

            response = input("Delete and rerun tracking? (y/N): ")
            if response.lower() != 'y':
                print("Skipping tracking. Keeping existing masks.")
                return 0
            else:
                print(f"Deleting {len(existing_masks)} existing masks...")
                for f in existing_masks:
                    os.remove(os.path.join(args.output_dir, f))
                print("> Deleted existing masks\n")

    # Initialize SAM 3D Body
    print("Initializing SAM 3D Body...")
    try:
        estimator = setup_sam_3d_body(hf_repo_id=args.model)
        print("> Initialized\n")
    except Exception as e:
        print(f"ERROR: Failed to initialize SAM 3D Body: {e}")
        print("Make sure you've downloaded the model checkpoint:")
        print(f"  hf download {args.model} --local-dir checkpoints/{args.model.split('/')[-1]}")
        return 1

    # Get frame list
    frames_dir = Path(args.frames_dir)
    all_frames = sorted([f for f in frames_dir.glob("*.jpg")])

    if not all_frames:
        print(f"ERROR: No frames found in {args.frames_dir}")
        return 1

    # Select frame range
    end_idx = min(args.start_frame + args.max_frames, len(all_frames))
    frames_to_process = all_frames[args.start_frame:end_idx]

    print(f"Processing {len(frames_to_process)} frames...")
    print()

    # Track statistics
    successful = 0
    failed = 0

    # Process each frame
    for i, frame_path in enumerate(frames_to_process):
        frame_idx = args.start_frame + i

        # Load image
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"Frame {frame_idx}: Failed to load {frame_path}")
            failed += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process with SAM 3D Body
        try:
            outputs = estimator.process_one_image(img_rgb)

            # Project to 2D mask
            mask = project_mesh_to_mask(outputs, img.shape[:2], args.mask_threshold)

            if mask is not None:
                # Save mask
                mask_filename = f"{frame_idx:04d}.png"
                mask_path = os.path.join(args.output_dir, mask_filename)
                Image.fromarray(mask * 255).save(mask_path)
                successful += 1

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(frames_to_process)} frames (Success: {successful}, Failed: {failed})")
            else:
                print(f"Frame {frame_idx}: No body detected")
                # Save empty mask
                mask_filename = f"{frame_idx:04d}.png"
                mask_path = os.path.join(args.output_dir, mask_filename)
                Image.fromarray(np.zeros(img.shape[:2], dtype=np.uint8)).save(mask_path)
                failed += 1

        except Exception as e:
            print(f"Frame {frame_idx}: Processing error - {e}")
            # Save empty mask
            mask_filename = f"{frame_idx:04d}.png"
            mask_path = os.path.join(args.output_dir, mask_filename)
            Image.fromarray(np.zeros(img.shape[:2], dtype=np.uint8)).save(mask_path)
            failed += 1

    print()
    print("=" * 70)
    print("TRACKING COMPLETE")
    print("=" * 70)
    print(f"Total frames processed: {len(frames_to_process)}")
    print(f"Successful detections: {successful}")
    print(f"Failed detections: {failed}")
    print(f"Success rate: {successful / len(frames_to_process) * 100:.1f}%")
    print(f"Masks saved to: {args.output_dir}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
