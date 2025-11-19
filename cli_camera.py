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

from camera_pipeline import CameraPipeline, CameraPipelineFactory, CameraConfig, BatchConfig, create_default_camera_configs


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

Resolution Notes:
  - Minimum recommended resolution: 256x256
  - Default resolution: 1024x1024 (best quality)
  - Smaller resolutions may produce poor quality results
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", type=str, help="Input image path")
    input_group.add_argument("--url", "-u", type=str, help="Input image URL")
    input_group.add_argument("--config", "-c", type=str, help="Configuration JSON file")
    input_group.add_argument("--batch", nargs="+", help="Multiple image paths for batch processing")
    input_group.add_argument("--image-dir", type=str, help="Directory containing images for batch processing")

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


def process_multi_images(image_paths: List[str], camera_configs: List[CameraConfig],
                         output_dir: str, device: str, verbose: bool = False):
    """Process multiple images efficiently using factory pattern"""

    print(f"🔄 Loading shared model pipeline for {len(image_paths)} images...")
    start_time = time.time()

    # Use factory to get shared pipeline
    pipeline = CameraPipelineFactory.get_pipeline(device=device)

    load_time = time.time() - start_time
    print(f"✅ Pipeline loaded in {load_time:.1f} seconds")
    print(f"   Processing {len(image_paths)} images with {len(camera_configs)} configurations each")

    successful = 0
    failed = 0
    total_start = time.time()

    for i, input_path in enumerate(image_paths, 1):
        if not os.path.exists(input_path):
            print(f"⚠️  Skipping missing file: {input_path}")
            failed += 1
            continue

        print(f"\n📸 [{i}/{len(image_paths)}] Processing: {input_path}")

        # Create individual image output directory
        path_parts = Path(input_path).parts
        img_id = f"{path_parts[-2]}_{Path(input_path).stem.split('_')[-1]}"
        image_output_dir = Path(output_dir) / img_id

        try:
            # Create batch config for this image
            batch_config = BatchConfig(
                input_path=input_path,
                output_dir=str(image_output_dir),
                camera_configs=camera_configs
            )

            # Process this image
            results = pipeline.process_batch(batch_config, verbose=verbose)

            successful += 1
            if verbose:
                print(f"   ✅ Generated {len(results)} images in {image_output_dir}")
            else:
                print(f"   ✅ Generated {len(results)} images")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1

    total_time = time.time() - total_start

    print(f"\n📊 Multi-Image Processing Summary:")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Successful: {successful}/{len(image_paths)} images")
    print(f"   Failed: {failed} images")
    print(f"   Average time per image: {total_time / max(len(image_paths), 1):.1f} seconds")
    print(f"   Total variations generated: {successful * len(camera_configs)}")


def get_images_from_directory(directory: str) -> List[str]:
    """Get all image files from a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    image_paths = []

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"❌ Directory not found: {directory}")
        return []

    for ext in image_extensions:
        image_paths.extend(dir_path.glob(f"*{ext}"))
        image_paths.extend(dir_path.glob(f"*{ext.upper()}"))

    return [str(p) for p in sorted(image_paths)]


# Add time import for the new functions
import time


def main():
    """Main CLI function"""
    args = parse_args()

    # Validate resolution
    args.height, args.width = validate_resolution(args.height, args.width)

    try:
        # Handle batch processing options
        if args.batch or args.image_dir:
            # Multi-image batch processing
            if args.batch:
                image_paths = args.batch
                print(f"Processing {len(image_paths)} specified images...")
            else:
                image_paths = get_images_from_directory(args.image_dir)
                print(f"Found {len(image_paths)} images in directory: {args.image_dir}")

            if not image_paths:
                print("❌ No images to process")
                sys.exit(1)

            # Create camera configurations
            camera_configs = create_camera_configs_from_args(args)

            if not camera_configs:
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

            # Process all images efficiently
            process_multi_images(
                image_paths=image_paths,
                camera_configs=camera_configs,
                output_dir=args.output,
                device=args.device,
                verbose=args.verbose
            )

        elif args.config:
            # Load from config file
            print(f"Loading configuration from: {args.config}")
            batch_config = load_config_file(args.config)

            # Override output directory if specified
            if args.output != "output":
                batch_config.output_dir = args.output

            # Validate resolution for config file too
            for config in batch_config.camera_configs:
                config.height, config.width = validate_resolution(config.height, config.width)

            if args.verbose:
                print(f"Batch configuration:")
                print(f"  Input: {batch_config.input_path or batch_config.input_url}")
                print(f"  Output: {batch_config.output_dir}")
                print(f"  Number of configurations: {len(batch_config.camera_configs)}")
                for i, config in enumerate(batch_config.camera_configs):
                    print(f"  Config {i+1}: rotate={config.rotate_deg}°, "
                          f"move={config.move_forward}, tilt={config.vertical_tilt}, "
                          f"wide={config.wideangle}")

            # Use factory for efficient single image processing
            print("\nInitializing camera pipeline...")
            pipeline = CameraPipelineFactory.get_pipeline(device=args.device)

            # Process batch
            print(f"\nProcessing {len(batch_config.camera_configs)} configurations...")
            results = pipeline.process_batch(batch_config, verbose=args.verbose)

            print(f"\n✅ Processing complete! Generated {len(results)} images.")
            print(f"📁 Output directory: {Path(batch_config.output_dir).absolute()}")

        else:
            # Single image processing (original behavior)
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
                for i, config in enumerate(batch_config.camera_configs):
                    print(f"  Config {i+1}: rotate={config.rotate_deg}°, "
                          f"move={config.move_forward}, tilt={config.vertical_tilt}, "
                          f"wide={config.wideangle}")

            # Use factory for efficiency
            print("\nInitializing camera pipeline...")
            pipeline = CameraPipelineFactory.get_pipeline(device=args.device)

            # Process batch
            print(f"\nProcessing {len(batch_config.camera_configs)} configurations...")
            results = pipeline.process_batch(batch_config, verbose=args.verbose)

            print(f"\n✅ Processing complete! Generated {len(results)} images.")
            print(f"📁 Output directory: {Path(batch_config.output_dir).absolute()}")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()