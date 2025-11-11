# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Qwen Image Edit Camera Control** application that provides AI-powered camera angle manipulation and video generation capabilities. The project uses:

- **Core Model**: Qwen-Image-Edit-2509 for image editing with camera controls
- **LoRA**: dx8152/Qwen-Edit-2509-Multiple-angles for camera angle transformations
- **Optimized Transformer**: Phr00t/Qwen-Image-Edit-Rapid-AIO for fast 4-step inference
- **UI**: Gradio web interface with real-time camera controls
- **CLI**: Command-line interface for batch processing
- **Video Generation**: External service integration for video creation between images

## Architecture

### Key Components

- **[`app.py`](app.py)**: Main Gradio application with UI and inference logic
- **[`cli_camera.py`](cli_camera.py)**: Command-line interface for batch processing
- **[`camera_pipeline.py`](camera_pipeline.py)**: Core pipeline module extracted for CLI usage
- **[`qwenimage/`](qwenimage/)**: Custom module containing the Qwen Image Edit implementation
  - [`pipeline_qwenimage_edit_plus.py`](qwenimage/pipeline_qwenimage_edit_plus.py): Main diffusion pipeline
  - [`transformer_qwenimage.py`](qwenimage/transformer_qwenimage.py): Custom transformer model implementation
  - [`qwen_fa3_processor.py`](qwenimage/qwen_fa3_processor.py): Flash Attention 3 processor
- **[`optimization.py`](optimization.py)**: Pipeline optimization using torch.compile and AO quantization
- **[`example_config.json`](example_config.json)**: Example configuration file for CLI batch processing

### Model Architecture

The application loads a hybrid pipeline:
1. Base pipeline from `Qwen/Qwen-Image-Edit-2509`
2. Custom transformer from `Phr00t/Qwen-Image-Edit-Rapid-AIO`
3. Camera angle LoRA adapter fused into the transformer
4. FA3 attention processor for optimized inference
5. Compiled with torch.compile for performance

## Common Development Tasks

### Running the Application

#### Web Interface (Gradio)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

#### Command-Line Interface (CLI)

```bash
# Install dependencies
pip install -r requirements.txt

# Basic usage with default angles
python cli_camera.py --input image.jpg

# Process specific rotation angles
python cli_camera.py --input image.jpg --rotate -45 0 45 90

# Process from URL
python cli_camera.py --url https://example.com/image.jpg --output results/

# Use configuration file
python cli_camera.py --config example_config.json

# Custom parameters
python cli_camera.py --input image.jpg --rotate 45 --move 3 --wide --steps 8 --guidance 1.5

# Verbose output
python cli_camera.py --input image.jpg --preset default --verbose
```

### Testing Camera Controls

The application provides these camera manipulations:
- **Rotation**: -180° to +180° (left/right rotation)
- **Movement**: Forward movement to close-up (0-10 scale)
- **Vertical Tilt**: Bird's eye view (-1), normal (0), worm's eye view (1)
- **Lens**: Wide-angle lens toggle

#### CLI Parameters

| Parameter | CLI Flag | Type | Default | Range/Options |
|-----------|----------|------|---------|---------------|
| Input Image | `--input` / `-i` | str | Required | File path |
| Input URL | `--url` / `-u` | str | Optional | HTTP URL |
| Output Directory | `--output` / `-o` | str | output | Directory path |
| Rotation | `--rotate` | float[] | 0 | -180 to 180 |
| Movement | `--move` | float[] | 0 | 0 to 10 |
| Vertical Tilt | `--tilt` | int[] | 0 | -1, 0, 1 |
| Wide Angle | `--wide` | flag | false | - |
| Seed | `--seed` | int | 0 | -1 for random |
| Inference Steps | `--steps` | int | 4 | 1 to 40 |
| Guidance Scale | `--guidance` | float | 1.0 | 1.0 to 10.0 |
| Image Height | `--height` | int | 1024 | 256 to 2048 |
| Image Width | `--width` | int | 1024 | 256 to 2048 |
| Device | `--device` | str | auto | auto, cuda, cpu |
| Preset | `--preset` | str | default | default, rotation_only, all |

#### CLI Presets

- **default**: Common camera angles (rotations, tilts, movement, wide)
- **rotation_only**: Only rotation angles (-90, -45, 0, 45, 90)
- **all**: Comprehensive angles including extended rotations

### Adding New Camera Effects

Camera prompts are built in `build_camera_prompt()` function ([`app.py:70`](app.py#L70)):
1. Add new parameters to the function signature
2. Update prompt generation logic with Chinese and English descriptions
3. Update the UI sliders/controls in the Gradio interface
4. Modify examples array to showcase new capabilities

### Model Updates

When updating models or LoRAs:
1. Update model loading in [`app.py:29-37`](app.py#L29-L37)
2. Adjust LoRA configuration and weights ([`app.py:44-47`](app.py#L44-L47))
3. Re-run optimization with new pipeline configuration

## Development Environment

### Dependencies
- **PyTorch**: Core ML framework with CUDA support
- **Diffusers**: Hugging Face diffusion library (from git source)
- **Gradio**: Web UI framework with Spaces integration
- **Transformers**: Hugging Face transformer models
- **Flash Attention**: Optimized attention implementation
- **torchao**: PyTorch optimization and quantization

### Hardware Requirements
- CUDA-capable GPU (application defaults to CUDA, falls back to CPU)
- Recommended VRAM: 8GB+ for optimal performance with 1024x1024 images

### Configuration Notes
- Uses `torch.bfloat16` dtype for memory efficiency
- 4-step inference by default (configurable via UI)
- Dynamic shape compilation for variable image sizes
- External video generation via Hugging Face Spaces API

## File Structure

```
├── app.py                    # Main Gradio application
├── cli_camera.py             # Command-line interface
├── camera_pipeline.py        # Core pipeline module for CLI
├── optimization.py           # Pipeline optimization utilities
├── requirements.txt          # Python dependencies
├── example_config.json       # Example CLI configuration
├── qwenimage/               # Custom Qwen Image Edit module
│   ├── __init__.py
│   ├── pipeline_qwenimage_edit_plus.py
│   ├── transformer_qwenimage.py
│   └── qwen_fa3_processor.py
└── *.jpg, *.png             # Sample images for examples
```

## CLI Batch Processing

The CLI tool supports batch processing of multiple camera angles with automatic output organization:

### Output Structure

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

### Configuration File Format

JSON configuration files support batch processing:

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
    }
  ]
}
```

## Key Functions

### Web Interface (Gradio)
- **`infer_camera_edit()`** ([`app.py:103`](app.py#L103)): Main inference function for camera edits
- **`build_camera_prompt()`** ([`app.py:70`](app.py#L70)): Constructs bilingual camera movement prompts
- **`_generate_video_segment()`** ([`app.py:59`](app.py#L59)): External video generation API call

### CLI and Pipeline
- **`CameraPipeline.process_batch()`** ([`camera_pipeline.py:156`](camera_pipeline.py#L156)): Batch process multiple camera configurations
- **`CameraPipeline.build_camera_prompt()`** ([`camera_pipeline.py:85`](camera_pipeline.py#L85)): CLI version of prompt generation
- **`create_default_camera_configs()`** ([`camera_pipeline.py:218`](camera_pipeline.py#L218)): Generate preset camera configurations
- **`optimize_pipeline_()`** ([`optimization.py:48`](optimization.py#L48)): Compiles and optimizes the transformer model