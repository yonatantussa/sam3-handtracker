# SAM3 Hand Tracker

Hand segmentation and tracking for egocentric video using SAM3.

## Requirements

- Linux with NVIDIA GPU (CUDA 12.6+)
- Python 3.12+
- 24GB VRAM (tested on RTX 3090)
- ~46GB RAM (for 1000 frames at 1408x1408)

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

Two modes available:
- **Points (recommended)**: Click 4-8 points per hand. Press spacebar to switch hands, Enter to save.
- **Boxes (experimental)**: Draw bounding boxes (two clicks per hand). Note: produces empty masks.

Saves to `hand_coords.json`.

**2. Track hands**
```bash
python track_hands.py --start-frame 0 --max-frames 1000
```

Generates binary masks (0=background, 1=right hand, 2=left hand) in `output/masks/`.

**3. Visualize**
```bash
python visualize_masks.py --start-frame 0
```

Creates colored overlays (green=right, red=left) in `output/visualizations/`.

## Notes

- Frame naming: `frame_NNNNNNNNNN.jpg` or `N.jpg` format
- Works best with around 8 points per hand
- Processing all frames may cause OOM - use batches instead
- Scripts check for existing output and prompt before overwriting
- **Point prompts**: ~96% detection rate (recommended)
- **Box prompts**: Implementation complete but SAM3 produces empty masks. See `BOXES_IMPLEMENTATION.md` for details.
- Use `--start-frame` and `--max-frames` to process specific ranges
- Match `--start-frame` in visualize_masks.py to track_hands.py for correct alignment