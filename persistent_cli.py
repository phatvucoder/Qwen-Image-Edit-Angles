#!/usr/bin/env python3
"""
Persistent CLI Camera Control for Qwen Image Edit
Keeps model loaded in RAM for fast sequential processing
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List
import signal

from camera_pipeline import CameraPipelineFactory, CameraConfig, BatchConfig, create_default_camera_configs


def validate_resolution(height: int, width: int) -> tuple[int, int]:
    """Validate and adjust resolution to minimum usable values"""
    min_resolution = 256

    if height < min_resolution or width < min_resolution:
        print(f"⚠️  Warning: Resolution {width}x{height} is too small for meaningful results")
        print(f"   Auto-adjusting to minimum recommended resolution {min_resolution}x{min_resolution}")
        return min_resolution, min_resolution

    return height, width


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process images sequentially with persistent model (model stays in RAM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python persistent_cli.py --input image.jpg --rotate -45 0 45

  # Process multiple images sequentially (model stays loaded)
  python persistent_cli.py --input image1.jpg --rotate -45 0 45
  python persistent_cli.py --input image2.jpg --rotate -45 0 45
  python persistent_cli.py --input image3.jpg --rotate -45 0 45

  # Clean up model from RAM when done
  python persistent_cli.py --cleanup

Resolution Notes:
  - Minimum recommended resolution: 256x256
  - Default resolution: 1024x1024 (best quality)
  - Smaller resolutions will be auto-adjusted
        """
    )

    # Special cleanup mode
    parser.add_argument("--cleanup", action="store_true",
                       help="Remove model from RAM and exit")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", type=str, help="Input image path")
    input_group.add_argument("--url", "-u", type=str, help="Input image URL")
    input_group.add_argument("--config", "-c", type=str, help="Configuration JSON file")

    # Output options
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Output directory (default: output)")

    # Camera parameters
    parser.add_argument("--rotate", nargs="+", type=float,
                       help="Rotation angles in degrees (e.g., -45 0 45 90)")
    parser.add_argument("--move", nargs="+", type=float,
                       help="Forward movement values (0-10)")
    parser.add_argument("--tilt", nargs="+", type=int, choices=[-1, 0, 1],
                       help="Vertical tilt (-1: bird's eye, 0: normal, 1: worm's eye)")
    parser.add_argument("--wide", action="store_true",
                       help="Use wide-angle lens")

    # Generation parameters
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0, use -1 for random)")
    parser.add_argument("--randomize-seed", action="store_true", default=True,
                       help="Randomize seed (default: True)")
    parser.add_argument("--steps", type=int, default=4,
                       help="Number of inference steps (default: 4)")
    parser.add_argument("--guidance", type=float, default=1.0,
                       help="True guidance scale (default: 1.0)")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height (default: 1024, minimum: 256)")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width (default: 1024, minimum: 256)")

    # Device options
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to use (default: auto)")

    # Preset options
    parser.add_argument("--preset", choices=["default", "rotation_only", "all"],
                       default="default", help="Use preset configurations")

    # Verbose output
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    return parser.parse_args()


def create_camera_configs_from_args(args) -> List[CameraConfig]:
    """Create camera configurations from command line arguments"""
    configs = []

    # Use preset if specified
    if args.preset == "default":
        base_configs = create_default_camera_configs()
    elif args.preset == "rotation_only":
        base_configs = [CameraConfig(rotate_deg=angle) for angle in [-90, -45, 0, 45, 90]]
    elif args.preset == "all":
        base_configs = create_default_camera_configs()
        # Add more rotation angles
        for angle in [-180, -135, 135, 180]:
            base_configs.append(CameraConfig(rotate_deg=angle))
    else:
        base_configs = [CameraConfig()]  # Single config with no changes

    # Filter based on provided arguments
    if args.rotate or args.move or args.tilt or args.wide:
        base_configs = []  # Clear presets if custom parameters provided

        # Create combinations of provided parameters
        rotates = args.rotate or [0]
        moves = args.move or [0]
        tilts = args.tilt or [0]
        wides = [args.wide]

        for rotate in rotates:
            for move in moves:
                for tilt in tilts:
                    for wide in wides:
                        config = CameraConfig(
                            rotate_deg=rotate,
                            move_forward=move,
                            vertical_tilt=tilt,
                            wideangle=wide,
                            seed=args.seed,
                            randomize_seed=args.randomize_seed,
                            true_guidance_scale=args.guidance,
                            num_inference_steps=args.steps,
                            height=args.height,
                            width=args.width
                        )
                        configs.append(config)

        # If no specific parameters provided, add at least one config
        if not configs:
            configs.append(CameraConfig(
                seed=args.seed,
                randomize_seed=args.randomize_seed,
                true_guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width
            ))
    else:
        # Update base configs with generation parameters
        for config in base_configs:
            config.seed = args.seed
            config.randomize_seed = args.randomize_seed
            config.true_guidance_scale = args.guidance
            config.num_inference_steps = args.steps
            config.height = args.height
            config.width = args.width
        configs = base_configs

    return configs


def load_config_file(config_path: str) -> BatchConfig:
    """Load batch configuration from JSON file"""
    with open(config_path, 'r') as f:
        data = json.load(f)

    # Convert dict configs to CameraConfig objects
    camera_configs = []
    for config_data in data.get('camera_configs', []):
        config = CameraConfig(**config_data)
        camera_configs.append(config)

    return BatchConfig(
        input_path=data.get('input_path', ''),
        input_url=data.get('input_url', ''),
        output_dir=data.get('output_dir', 'output'),
        camera_configs=camera_configs
    )


def process_single_image(args):
    """Process a single image using persistent pipeline"""

    # Validate resolution
    args.height, args.width = validate_resolution(args.height, args.width)

    try:
        # Create camera configurations
        if args.preset or args.rotate or args.move or args.tilt or args.wide:
            camera_configs = create_camera_configs_from_args(args)
        else:
            camera_configs = [CameraConfig(
                seed=args.seed,
                randomize_seed=args.randomize_seed,
                true_guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width
            )]

        # Validate resolution for all configs
        for config in camera_configs:
            config.height, config.width = validate_resolution(config.height, config.width)

        if args.verbose:
            print(f"Camera configurations: {len(camera_configs)}")
            for i, config in enumerate(camera_configs[:5]):  # Show first 5
                print(f"  Config {i+1}: rotate={config.rotate_deg}°, "
                      f"move={config.move_forward}, tilt={config.vertical_tilt}, "
                      f"wide={config.wideangle}")
            if len(camera_configs) > 5:
                print(f"  ... and {len(camera_configs) - 5} more")

        # Get or create persistent pipeline
        print("🔄 Accessing model pipeline...")
        start_time = time.time()

        pipeline = CameraPipelineFactory.get_pipeline(device=args.device)

        access_time = time.time() - start_time
        if access_time < 1.0:  # Very fast access means model was already loaded
            print(f"✅ Model already loaded in RAM (instant access: {access_time:.2f}s)")
        else:  # Slower access means model had to be loaded
            print(f"✅ Model loaded in {access_time:.1f}s")

        # Create batch config
        batch_config = BatchConfig(
            input_path=args.input or "",
            input_url=args.url or "",
            output_dir=args.output,
            camera_configs=camera_configs
        )

        # Process the image
        print(f"📸 Processing: {args.input or args.url}")
        process_start = time.time()

        results = pipeline.process_batch(batch_config, verbose=args.verbose)

        process_time = time.time() - process_start

        print(f"✅ Generated {len(results)} images in {process_time:.1f} seconds")
        print(f"📁 Output directory: {Path(args.output).absolute()}")

        # Show model persistence info
        print(f"💡 Model remains loaded in RAM for next image")
        print(f"   Use 'python persistent_cli.py --cleanup' to free RAM when done")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def cleanup_model():
    """Clean up model from RAM"""
    print("🧹 Cleaning up model from RAM...")
    CameraPipelineFactory.reset_cache()
    print("✅ Model removed from RAM")
    print("💡 Next run will reload the model")


def setup_signal_handlers():
    """Setup signal handlers for graceful cleanup"""
    def signal_handler(signum, frame):
        print(f"\n📡 Received signal {signum}, cleaning up...")
        cleanup_model()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination


def main():
    """Main persistent CLI function"""
    setup_signal_handlers()

    args = parse_args()

    if args.cleanup:
        cleanup_model()
        return

    # Process single image with persistent model
    process_single_image(args)


if __name__ == "__main__":
    main()