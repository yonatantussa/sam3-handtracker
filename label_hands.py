#!/usr/bin/env python3
"""
Interactive coordinate picker using matplotlib.
Click on hands to get coordinates for SAM3 tracking.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json

# Global state
right_hand_points = []
left_hand_points = []
current_hand = "right"
img = None
ax = None
fig = None


def onclick(event):
    """Handle mouse click events."""
    global right_hand_points, left_hand_points, current_hand, ax, fig
    
    if event.xdata is None or event.ydata is None:
        return
    
    x, y = int(event.xdata), int(event.ydata)
    
    if current_hand == "right":
        right_hand_points.append([x, y])
        color = 'lime'
        label = f"R{len(right_hand_points)}"
    else:
        left_hand_points.append([x, y])
        color = 'red'
        label = f"L{len(left_hand_points)}"
    
    # Draw point
    ax.plot(x, y, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.text(x+10, y-10, label, color=color, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    print(f"{current_hand.upper()} hand: [{x}, {y}]")
    
    # Update title
    title = f"Right: {len(right_hand_points)} points (green) | Left: {len(left_hand_points)} points (red)\n"
    title += "Press SPACEBAR to switch hands | Press ENTER to finish"
    ax.set_title(title, fontsize=12, pad=20)
    
    fig.canvas.draw()


def onkey(event):
    """Handle keyboard events."""
    global current_hand, ax, fig, img, right_hand_points, left_hand_points
    
    if event.key == ' ':  # Spacebar to switch hands
        if current_hand == "right":
            current_hand = "left"
            print("\n>>> Switched to LEFT hand (red dots)")
            # Redraw image and existing points
            ax.clear()
            ax.imshow(img)
            ax.axis('off')
            
            # Redraw right hand points
            for point in right_hand_points:
                ax.plot(point[0], point[1], 'go', markersize=10, markeredgewidth=2, markeredgecolor='white')
            
            title = f"Right: {len(right_hand_points)} points (green) | Left: {len(left_hand_points)} points (red)\n"
            title += "Now clicking LEFT hand | Press SPACE when done, ENTER to finish"
            ax.set_title(title, fontsize=12, pad=20)
            fig.canvas.draw_idle()
        else:
            current_hand = "right"
            print("\n>>> Switched to RIGHT hand (green dots)")
            # Redraw image and existing points
            ax.clear()
            ax.imshow(img)
            ax.axis('off')
            
            # Redraw both hands points
            for point in right_hand_points:
                ax.plot(point[0], point[1], 'go', markersize=10, markeredgewidth=2, markeredgecolor='white')
            for point in left_hand_points:
                ax.plot(point[0], point[1], 'ro', markersize=10, markeredgewidth=2, markeredgecolor='white')
            
            title = f"Right: {len(right_hand_points)} points (green) | Left: {len(left_hand_points)} points (red)\n"
            title += "Now clicking RIGHT hand | Press SPACE when done, ENTER to finish"
            ax.set_title(title, fontsize=12, pad=20)
            fig.canvas.draw_idle()
    
    elif event.key == 'enter':
        plt.close()


def main():
    global img, ax, fig
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Label hands on a video frame')
    parser.add_argument('--frame-index', type=int, default=0,
                        help='Frame index to label (default: 0)')
    parser.add_argument('--frames-dir', type=str, default='../frames_preview',
                        help='Directory containing video frames (default: ../frames_preview)')
    parser.add_argument('--output', type=str, default='hand_coords.json',
                        help='Output JSON file (default: hand_coords.json)')
    args = parser.parse_args()
    
    FRAMES_DIR = args.frames_dir
    FRAME_IDX = args.frame_index
    OUTPUT_FILE = args.output
    
    # Load image - try different naming patterns
    frame_path = Path(FRAMES_DIR) / f"{FRAME_IDX}.jpg"
    if not frame_path.exists():
        # Try frame_XXXXXXXXXX.jpg format
        frame_path = Path(FRAMES_DIR) / f"frame_{FRAME_IDX:010d}.jpg"
    
    if not frame_path.exists():
        # Try listing first available frame
        frames_dir = Path(FRAMES_DIR)
        if frames_dir.exists():
            jpg_files = sorted(frames_dir.glob("*.jpg"))
            if jpg_files:
                frame_path = jpg_files[FRAME_IDX] if FRAME_IDX < len(jpg_files) else jpg_files[0]
    
    if not frame_path.exists():
        print(f"Error: No frames found in {FRAMES_DIR}/")
        print(f"Checked for: {FRAME_IDX}.jpg or frame_{FRAME_IDX:010d}.jpg")
        return
    
    print("\n" + "=" * 70)
    print("SAM3 Hand Coordinate Picker")
    print("=" * 70)
    print("Instructions:")
    print("  1. Click points on the RIGHT hand (green dots)")
    print("  2. Press SPACEBAR to switch to LEFT hand")
    print("  3. Click points on the LEFT hand (red dots)")
    print("  4. Press SPACEBAR again to go back to RIGHT hand if needed")
    print("  5. Press ENTER when done")
    print("\nNote: Works best with around 8 points per hand")
    print("=" * 70 + "\n")
    
    img = mpimg.imread(frame_path)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    ax.axis('off')
    
    title = f"RIGHT hand mode (green) | Press SPACEBAR to switch, ENTER when done\n"
    title += f"Image size: {img.shape[1]}x{img.shape[0]}"
    ax.set_title(title, fontsize=12, pad=20, color='green')
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    if right_hand_points or left_hand_points:
        coords = {
            "right": right_hand_points,
            "left": left_hand_points,
            "frame_idx": FRAME_IDX
        }
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(coords, f, indent=2)
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Right hand: {len(right_hand_points)} points")
        print(f"Left hand: {len(left_hand_points)} points")
        print(f"\nSaved to: {OUTPUT_FILE}")
        print("\nTo run tracking, use:")
        print(f"  python track_hands.py --max-frames 1000")
        print("=" * 70 + "\n")
    else:
        print("\nNo points captured!")


if __name__ == "__main__":
    main()
