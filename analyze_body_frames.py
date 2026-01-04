#!/usr/bin/env python3
"""Analyze body masks to identify frames with visible body."""

import os
import argparse
import json
import numpy as np
from PIL import Image
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze body masks for frame detection')
    parser.add_argument('--masks-dir', type=str, default='output/body_masks',
                        help='Directory containing body masks (default: output/body_masks)')
    parser.add_argument('--output', type=str, default='body_frames.json',
                        help='Output JSON file (default: body_frames.json)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Minimum body pixel ratio to count as visible (default: 0.01 = 1%%)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Body Frame Analysis")
    print("=" * 70)
    print(f"Masks directory: {args.masks_dir}")
    print(f"Visibility threshold: {args.threshold * 100:.1f}%")
    print("=" * 70 + "\n")

    # Get all mask files
    masks_dir = Path(args.masks_dir)
    if not masks_dir.exists():
        print(f"ERROR: Masks directory not found: {args.masks_dir}")
        return 1

    mask_files = sorted([f for f in masks_dir.glob("*.png")])

    if not mask_files:
        print(f"ERROR: No mask files found in {args.masks_dir}")
        return 1

    print(f"Analyzing {len(mask_files)} masks...")

    # Initialize results
    results = {
        "frames_with_body": [],
        "frames_without_body": [],
        "frame_details": {},
        "statistics": {}
    }

    # Process each mask
    for mask_file in mask_files:
        # Extract frame index from filename
        frame_idx = int(mask_file.stem)

        # Load mask
        mask = np.array(Image.open(mask_file))

        # Calculate body presence
        total_pixels = mask.size
        body_pixels = np.sum(mask > 0)
        body_ratio = body_pixels / total_pixels if total_pixels > 0 else 0

        # Store details
        results["frame_details"][frame_idx] = {
            "body_pixels": int(body_pixels),
            "total_pixels": int(total_pixels),
            "body_ratio": float(body_ratio),
            "has_body": body_ratio >= args.threshold
        }

        # Categorize frame
        if body_ratio >= args.threshold:
            results["frames_with_body"].append(frame_idx)
        else:
            results["frames_without_body"].append(frame_idx)

    # Calculate statistics
    total_frames = len(mask_files)
    frames_with_body = len(results["frames_with_body"])
    frames_without_body = len(results["frames_without_body"])

    results["statistics"] = {
        "total_frames": total_frames,
        "frames_with_body": frames_with_body,
        "frames_without_body": frames_without_body,
        "body_presence_ratio": frames_with_body / total_frames if total_frames > 0 else 0,
        "threshold_used": args.threshold,
        "frame_range": {
            "start": min([int(f.stem) for f in mask_files]),
            "end": max([int(f.stem) for f in mask_files])
        }
    }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames with body: {frames_with_body} ({frames_with_body/total_frames*100:.1f}%)")
    print(f"Frames without body: {frames_without_body} ({frames_without_body/total_frames*100:.1f}%)")
    print(f"Frame range: {results['statistics']['frame_range']['start']} - {results['statistics']['frame_range']['end']}")
    print(f"\nResults saved to: {args.output}")
    print("=" * 70 + "\n")

    # Print first few frames with/without body for quick check
    if frames_with_body:
        print(f"First frames with body: {results['frames_with_body'][:10]}")
    if frames_without_body:
        print(f"First frames without body: {results['frames_without_body'][:10]}")
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
