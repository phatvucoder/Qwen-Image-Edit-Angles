#!/usr/bin/env python3
"""
CLI Camera Control for Qwen Image Edit

Command-line interface for batch processing camera angles on images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from camera_pipeline import CameraPipeline, CameraConfig, BatchConfig, create_default_camera_configs


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process images with camera angle controls using Qwen Image Edit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default angles
  python cli_camera.py --input image.jpg

  # Process specific rotation angles
  python cli_camera.py --input image.jpg --rotate -45 0 45 90

  # Process from URL
  python cli_camera.py --url https://example.com/image.jpg --output results/

  # Custom configuration
  python cli_camera.py --input image.jpg --rotate 45 --move 3 --wide --steps 8 --guidance 1.5

  # Use config file
  python cli_camera.py --config my_config.json
        """
    )

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
                       help="Randomize seed (default: True, like GUI)")
    parser.add_argument("--steps", type=int, default=4,
                       help="Number of inference steps (default: 4)")
    parser.add_argument("--guidance", type=float, default=1.0,
                       help="True guidance scale (default: 1.0)")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height (default: 1024)")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width (default: 1024)")

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
        # More comprehensive preset
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
                            randomize_seed=args.randomize_seed,  # Use new parameter
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
                randomize_seed=args.randomize_seed,  # Use new parameter
                true_guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width
            ))
    else:
        # Update base configs with generation parameters
        for config in base_configs:
            config.seed = args.seed
            config.randomize_seed = args.randomize_seed  # Use new parameter
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


def save_config_file(batch_config: BatchConfig, output_path: str):
    """Save batch configuration to JSON file"""
    data = {
        'input_path': batch_config.input_path,
        'input_url': batch_config.input_url,
        'output_dir': batch_config.output_dir,
        'camera_configs': [
            {
                'rotate_deg': config.rotate_deg,
                'move_forward': config.move_forward,
                'vertical_tilt': config.vertical_tilt,
                'wideangle': config.wideangle,
                'seed': config.seed,
                'randomize_seed': config.randomize_seed,
                'true_guidance_scale': config.true_guidance_scale,
                'num_inference_steps': config.num_inference_steps,
                'height': config.height,
                'width': config.width
            }
            for config in batch_config.camera_configs
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Main CLI function"""
    args = parse_args()

    try:
        if args.config:
            # Load from config file
            print(f"Loading configuration from: {args.config}")
            batch_config = load_config_file(args.config)

            # Override output directory if specified
            if args.output != "output":
                batch_config.output_dir = args.output

        else:
            # Create configuration from command line arguments
            camera_configs = create_camera_configs_from_args(args)

            batch_config = BatchConfig(
                input_path=args.input or "",
                input_url=args.url or "",
                output_dir=args.output,
                camera_configs=camera_configs
            )

        if args.verbose:
            print(f"Batch configuration:")
            print(f"  Input: {batch_config.input_path or batch_config.input_url}")
            print(f"  Output: {batch_config.output_dir}")
            print(f"  Number of configurations: {len(batch_config.camera_configs)}")
            if args.verbose:
                for i, config in enumerate(batch_config.camera_configs):
                    print(f"  Config {i+1}: rotate={config.rotate_deg}°, "
                          f"move={config.move_forward}, tilt={config.vertical_tilt}, "
                          f"wide={config.wideangle}")

        # Initialize pipeline
        print("\nInitializing camera pipeline...")
        pipeline = CameraPipeline(device=args.device)

        # Process batch with verbose parameter
        print(f"\nProcessing {len(batch_config.camera_configs)} configurations...")
        results = pipeline.process_batch(batch_config, verbose=args.verbose)

        print(f"\n✅ Processing complete! Generated {len(results)} images.")
        print(f"📁 Output directory: {Path(batch_config.output_dir).absolute()}")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()