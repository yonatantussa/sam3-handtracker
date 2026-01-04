#!/usr/bin/env python3
"""
Interactive coordinate picker using matplotlib.
Click on hands to get coordinates for SAM3 tracking.
Supports both point prompts and bounding box prompts.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json
import sys
from matplotlib.patches import Rectangle

# Configuration
FRAMES_DIR = "../frames_preview"
FRAME_IDX = 0
OUTPUT_FILE = "hand_coords.json"

# Global state
right_hand_points = []
left_hand_points = []
right_hand_box = None
left_hand_box = None
current_hand = "right"
use_boxes = False
box_start = None
box_rect = None
img = None
ax = None
fig = None


def onclick(event):
    """Handle mouse click events."""
    global right_hand_points, left_hand_points, current_hand, ax, fig
    global right_hand_box, left_hand_box, box_start, box_rect, use_boxes
    
    if event.xdata is None or event.ydata is None:
        return
    
    x, y = int(event.xdata), int(event.ydata)
    
    if use_boxes:
        # Box mode: click twice to draw box
        if box_start is None:
            # First click - start of box
            box_start = [x, y]
            print(f"{current_hand.upper()} hand box - first corner: [{x}, {y}]")
        else:
            # Second click - complete the box
            x_min = min(box_start[0], x)
            y_min = min(box_start[1], y)
            x_max = max(box_start[0], x)
            y_max = max(box_start[1], y)
            
            box = [x_min, y_min, x_max, y_max]
            
            if current_hand == "right":
                right_hand_box = box
                color = 'lime'
            else:
                left_hand_box = box
                color = 'red'
            
            # Draw rectangle
            width = x_max - x_min
            height = y_max - y_min
            rect = Rectangle((x_min, y_min), width, height, 
                           linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            print(f"{current_hand.upper()} hand box: {box}")
            box_start = None
            
            # Update title
            title = f"Right: {'BOX' if right_hand_box else 'none'} | Left: {'BOX' if left_hand_box else 'none'}\n"
            title += "Press SPACEBAR to switch hands | Press ENTER to finish"
            ax.set_title(title, fontsize=12, pad=20)
            fig.canvas.draw()
    else:
        # Point mode
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
    global right_hand_box, left_hand_box, use_boxes
    
    if event.key == ' ':  # Spacebar to switch hands
        if current_hand == "right":
            current_hand = "left"
            print(f"\n>>> Switched to LEFT hand ({'box' if use_boxes else 'points'})")
            # Redraw image and existing annotations
            ax.clear()
            ax.imshow(img)
            ax.axis('off')
            
            if use_boxes:
                # Redraw right hand box
                if right_hand_box:
                    x_min, y_min, x_max, y_max = right_hand_box
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = Rectangle((x_min, y_min), width, height,
                                   linewidth=3, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                title = f"Right: {'BOX' if right_hand_box else 'none'} | Left: {'BOX' if left_hand_box else 'none'}\n"
            else:
                # Redraw right hand points
                for point in right_hand_points:
                    ax.plot(point[0], point[1], 'go', markersize=10, markeredgewidth=2, markeredgecolor='white')
                title = f"Right: {len(right_hand_points)} points | Left: {len(left_hand_points)} points\n"
            
            title += "Now clicking LEFT hand | Press SPACE to switch, ENTER to finish"
            ax.set_title(title, fontsize=12, pad=20)
            fig.canvas.draw_idle()
        else:
            current_hand = "right"
            print(f"\n>>> Switched to RIGHT hand ({'box' if use_boxes else 'points'})")
            # Redraw image and existing annotations
            ax.clear()
            ax.imshow(img)
            ax.axis('off')
            
            if use_boxes:
                # Redraw both boxes
                if right_hand_box:
                    x_min, y_min, x_max, y_max = right_hand_box
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = Rectangle((x_min, y_min), width, height,
                                   linewidth=3, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                if left_hand_box:
                    x_min, y_min, x_max, y_max = left_hand_box
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = Rectangle((x_min, y_min), width, height,
                                   linewidth=3, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                title = f"Right: {'BOX' if right_hand_box else 'none'} | Left: {'BOX' if left_hand_box else 'none'}\n"
            else:
                # Redraw both hands points
                for point in right_hand_points:
                    ax.plot(point[0], point[1], 'go', markersize=10, markeredgewidth=2, markeredgecolor='white')
                for point in left_hand_points:
                    ax.plot(point[0], point[1], 'ro', markersize=10, markeredgewidth=2, markeredgecolor='white')
                title = f"Right: {len(right_hand_points)} points | Left: {len(left_hand_points)} points\n"
            
            title += "Now clicking RIGHT hand | Press SPACE to switch, ENTER to finish"
            ax.set_title(title, fontsize=12, pad=20)
            fig.canvas.draw_idle()
    
    elif event.key == 'enter':
        plt.close()


def main():
    global img, ax, fig, use_boxes
    
    # Ask for mode
    print("\n" + "=" * 70)
    print("SAM3 Hand Labeling Tool")
    print("=" * 70)
    mode = input("Use (p)oints or (b)oxes? [p=default]: ").strip().lower()
    use_boxes = (mode == 'b')
    
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
    print(f"Mode: {'Bounding Boxes' if use_boxes else 'Point Prompts'}")
    print("=" * 70)
    
    if use_boxes:
        print("Instructions:")
        print("  1. Click TWO corners to draw a box around the RIGHT hand (green)")
        print("  2. Press SPACEBAR to switch to LEFT hand")
        print("  3. Click TWO corners to draw a box around the LEFT hand (red)")
        print("  4. Press SPACEBAR again to go back to RIGHT hand if needed")
        print("  5. Press ENTER when done")
        print("\nNote: Click diagonal corners of a rectangle around each hand")
    else:
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
    
    if use_boxes:
        title = f"RIGHT hand mode (green box) | Press SPACEBAR to switch, ENTER when done\n"
    else:
        title = f"RIGHT hand mode (green) | Press SPACEBAR to switch, ENTER when done\n"
    title += f"Image size: {img.shape[1]}x{img.shape[0]}"
    ax.set_title(title, fontsize=12, pad=20, color='green')
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    if use_boxes:
        if right_hand_box or left_hand_box:
            coords = {
                "mode": "boxes",
                "right": right_hand_box if right_hand_box else [],
                "left": left_hand_box if left_hand_box else [],
                "frame_idx": FRAME_IDX
            }
            
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(coords, f, indent=2)
            
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            print(f"Right hand: {'Box' if right_hand_box else 'none'} {right_hand_box if right_hand_box else ''}")
            print(f"Left hand: {'Box' if left_hand_box else 'none'} {left_hand_box if left_hand_box else ''}")
            print(f"\nSaved to: {OUTPUT_FILE}")
            print("\nTo run tracking, use:")
            print(f"  python track_hands.py --max-frames 1000")
            print("=" * 70 + "\n")
        else:
            print("\nNo boxes captured!")
    else:
        if right_hand_points or left_hand_points:
            coords = {
                "mode": "points",
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
