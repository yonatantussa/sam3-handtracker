# SAM 3D Body Tracker

Body detection and segmentation for egocentric video using SAM 3D Body.

## Goals

1. **Primary:** Identify which frames have a visible body
2. **Secondary:** Generate body masks for diffusion-based inpainting (to remove wrist box/wires)

## Requirements

- Linux with NVIDIA GPU (CUDA 12.6+)
- Python 3.12+
- 24GB+ VRAM (SAM 3D Body is a large model)

## Installation

```bash
# Create environment
conda create -n sam3d python=3.12
conda activate sam3d

# Install dependencies
pip install -r requirements_body.txt

# Install SAM 3D Body
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body
pip install -e .
cd ..

# Download model checkpoint
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

## Usage

### Quick Start

```bash
python run_body_pipeline.py
```

Interactive pipeline that runs: tracking → analysis → visualization.

**No manual labeling required!** SAM 3D Body automatically detects bodies.

### Step-by-Step

**1. Track bodies**
```bash
python track_body.py --start-frame 0 --max-frames 1000
```

- Automatically detects bodies in each frame
- Generates 3D meshes and projects to 2D masks
- Saves binary masks (0=background, 1=body) to `output/body_masks/`

**2. Analyze frames**
```bash
python analyze_body_frames.py
```

- Identifies which frames have visible bodies
- Outputs `body_frames.json` with frame lists and statistics

**3. Visualize**
```bash
python visualize_masks.py --masks-dir output/body_masks --output-dir output/body_visualizations
```

- Creates colored overlays (cyan=body)
- Saves to `output/body_visualizations/`

## Output Structure

```
output/
├── body_masks/              # Binary masks for inpainting
│   ├── 0000.png            # 0=background, 1=body
│   ├── 0001.png
│   └── ...
├── body_visualizations/     # Visual inspection
│   ├── vis_0000.jpg
│   ├── vis_0001.jpg
│   └── ...
└── ../body_frames.json     # Frame detection results
```

### body_frames.json Format

```json
{
  "frames_with_body": [0, 1, 2, 5, 6, ...],
  "frames_without_body": [3, 4, 8, ...],
  "frame_details": {
    "0": {
      "body_pixels": 123456,
      "total_pixels": 2073600,
      "body_ratio": 0.0595,
      "has_body": true
    },
    ...
  },
  "statistics": {
    "total_frames": 1000,
    "frames_with_body": 847,
    "frames_without_body": 153,
    "body_presence_ratio": 0.847,
    "threshold_used": 0.01
  }
}
```

## Configuration

### Detection Threshold

Adjust sensitivity for "body present" detection:

```bash
python analyze_body_frames.py --threshold 0.05  # Stricter (5% of pixels)
python analyze_body_frames.py --threshold 0.001 # Lenient (0.1% of pixels)
```

### Mask Projection Threshold

Control quality of 3D→2D projection:

```bash
python track_body.py --mask-threshold 0.7  # Higher confidence
```

## Use Cases

### 1. Frame Filtering

Filter video to frames with visible bodies:

```python
import json

# Load frame analysis
with open('body_frames.json') as f:
    data = json.load(f)

# Get frames with bodies
valid_frames = data['frames_with_body']
print(f"Processing {len(valid_frames)} frames with visible bodies")
```

### 2. Diffusion Inpainting

Remove wrist box/wires using generated masks:

```python
from PIL import Image
import numpy as np

# Load frame and body mask
frame = np.array(Image.open('frames/0000.jpg'))
body_mask = np.array(Image.open('output/body_masks/0000.png'))

# Invert mask to inpaint body (remove wires/box on body)
inpaint_mask = (body_mask > 0).astype(np.uint8)

# Use with your diffusion model
# inpainted = diffusion_model(frame, mask=inpaint_mask, prompt="...")
```

## Notes

- SAM 3D Body works on **single images** (not video sequences)
- Each frame is processed independently
- Detection failures result in empty masks (all zeros)
- Frame naming: `frame_NNNNNNNNNN.jpg` or `N.jpg` format
- Processing is slower than SAM3 (3D mesh estimation per frame)

## Comparison with Hand Tracking

| Feature | Hand Tracking (SAM3) | Body Tracking (SAM 3D Body) |
|---------|---------------------|----------------------------|
| Input | Point prompts required | Automatic detection |
| Output | 2D masks | 3D mesh → 2D masks |
| Speed | Fast (video optimized) | Slower (per-frame 3D) |
| Objects | 2 (left/right hands) | 1 (body) |
| Use case | Hand segmentation | Body detection + inpainting |

## Troubleshooting

- **CUDA OOM**: Reduce batch size or frame resolution
- **Empty masks**: Body not visible or detection failed, check visualizations
- **Import errors**: Make sure SAM 3D Body is installed correctly
- **Model not found**: Download checkpoint with `hf download`
