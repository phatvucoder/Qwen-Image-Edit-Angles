# Qwen Image Edit Camera Control

🎬 **AI-powered camera angle manipulation and batch processing for Qwen Image Edit**

This project provides comprehensive camera angle control for images using the Qwen-Image-Edit-2509 model with camera angle LoRA adapters. It includes both a web interface and command-line tools for batch processing.

## 🌟 Features

- **🎯 Camera Angle Control**: Rotate, zoom, tilt, and apply wide-angle effects
- **🚀 Batch Processing**: Generate multiple camera angles from a single image
- **🖥️ Multiple Interfaces**: Gradio web UI and command-line interface
- **⚡ Fast Inference**: 4-step inference with optimized transformer
- **📊 Custom Configs**: JSON configuration support for complex batch jobs

## 📋 System Requirements

### Hardware
- **GPU**: CUDA-capable GPU recommended
- **VRAM**: 8GB+ recommended for 1024x1024 images
- **Disk**: 40GB+ available for model downloads

### Software
- **Python**: 3.9+
- **CUDA**: 11.8+ (for GPU acceleration)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Qwen-Image-Edit-Angles.git
cd Qwen-Image-Edit-Angles
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n qwen-camera python=3.10
conda activate qwen-camera

# OR using venv
python -m venv qwen-camera
source qwen-camera/bin/activate  # On Windows: qwen-camera\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Web Interface

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

### Command-Line Interface

```bash
# Basic usage with default camera angles
python cli_camera.py --input image.jpg

# Process specific rotation angles
python cli_camera.py --input image.jpg --rotate -45 0 45 90

# Process from URL
python cli_camera.py --url https://example.com/image.jpg --output results/

# Generate all preset angles
python cli_camera.py --input image.jpg --preset all
```

## 📊 Common Use Cases

### 1. Product Photography

Generate multiple product shots from a single image:

```bash
python cli_camera.py --input product.jpg --rotate -45 -30 0 30 45 --move 3 6 --wide
```

### 2. Architectural Visualization

Create different architectural perspectives:

```bash
python cli_camera.py --input building.jpg --rotate -90 -45 0 45 90 --tilt -1 0 1
```

### 3. Portrait Photography

Generate various portrait angles:

```bash
python cli_camera.py --input portrait.jpg --rotate -15 0 15 --move 0 5 8
```

## 🔧 Custom Configuration

### JSON Configuration File

Create a JSON configuration file for complex batch processing:

```json
{
  "input_path": "image.jpg",
  "input_url": "",
  "output_dir": "output",
  "camera_configs": [
    {
      "rotate_deg": -90,
      "move_forward": 0,
      "vertical_tilt": 0,
      "wideangle": false,
      "seed": 42,
      "randomize_seed": false,
      "true_guidance_scale": 1.0,
      "num_inference_steps": 4,
      "height": 1024,
      "width": 1024
    },
    {
      "rotate_deg": 45,
      "move_forward": 5,
      "vertical_tilt": 1,
      "wideangle": true,
      "seed": 123,
      "randomize_seed": false,
      "true_guidance_scale": 1.5,
      "num_inference_steps": 6,
      "height": 1024,
      "width": 1024
    }
  ]
}
```

Then run:

```bash
python cli_camera.py --config my_config.json
```

### CLI Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--input` | Input image path | Required | File path |
| `--rotate` | Rotation angles in degrees | 0 | -180 to 180 |
| `--move` | Forward movement (0-10 scale) | 0 | 0 to 10 |
| `--tilt` | Vertical tilt (-1, 0, 1) | 0 | -1, 0, 1 |
| `--wide` | Enable wide-angle lens | false | flag |
| `--steps` | Inference steps | 4 | 1 to 40 |
| `--guidance` | Guidance scale | 1.0 | 1.0 to 10.0 |
| `--height` | Image height | 1024 | 256 to 2048 |
| `--width` | Image width | 1024 | 256 to 2048 |

## 📁 Output Structure

```
output/
├── 00_original.png          # Original input image
├── 01_rot_-45deg.png        # 45-degree left rotation
├── 02_move_3.png           # Forward movement
├── 03_tilt_-1.png          # Bird's eye view
├── 04_wide.png             # Wide angle lens
├── 05_rot_45deg_wide.png   # Combination effects
└── metadata.json           # Processing metadata
```

## 🤝 Citation

This project is a fork and extension of [Qwen Image Edit Angles](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles) by [linoyts](https://huggingface.co/linoyts).

**Original Space:** https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles

**Core Models:**
- **Base Model:** [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- **LoRA:** [dx8152/Qwen-Edit-2509-Multiple-angles](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles)
- **Transformer:** [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO)

If you use this project in your research, please cite:

```bibtex
@software{qwen_image_edit_angles_2024,
  title={Qwen Image Edit Angles - Camera Control Extension},
  author={linoyts},
  year={2024},
  url={https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
title: Qwen Image Edit Camera Control
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: Fast 4 step inference with Qwen Image Edit 2509
---