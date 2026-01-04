# SAM3 Hand Tracker

Hand segmentation and tracking for egocentric video using SAM3.

## Requirements

- Linux with NVIDIA GPU (CUDA 12.6+)
- Python 3.12+
- 24GB VRAM (tested on RTX 3090)
- 46GB RAM (for 1000 frames)

## Installation

```bash
# Create environment
conda create -n sam3 python=3.12
conda activate sam3

# Install SAM3
cd /path/to/sam3
pip install -e .

# Install dependencies
cd /path/to/sam3-handtracker
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python run_pipeline.py
```

Interactive pipeline that runs: labeling -> tracking -> visualization.

### Step-by-Step

**1. Label hands**
```bash
python label_hands.py
```
- Click points on right hand (green)
- Press spacebar to switch to left hand (red)
- Press Enter when done

Saves to `hand_coords.json`.

**2. Track hands**
```bash
python track_hands.py --max-frames 1000
```

Generates binary masks (0=background, 1=right, 2=left) in `output/masks/`.

**3. Visualize**
```bash
python visualize_masks.py
```

Creates colored overlays in `output/visualizations/`.

## Configuration

Edit these variables in scripts:

- `FRAMES_DIR`: Input frames directory (default: `../frames_preview`)
- `MAX_FRAMES`: Frames to process (default: 1000)
- `OUTPUT_DIR`: Output directory (default: `output/`)

Or use command-line arguments:
```bash
python track_hands.py --max-frames 500 --frames-dir ./frames
```

## Notes

- Frame naming: `frame_NNNNNNNNNN.jpg` or `N.jpg` format
- Works best with around 8 points per hand
- Processing all frames may cause OOM - use batches instead
- Scripts check for existing output and prompt before overwriting
