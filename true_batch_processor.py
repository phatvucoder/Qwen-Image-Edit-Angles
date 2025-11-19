#!/usr/bin/env python3
"""
True Batch Processor - Single Process, Multiple Images
Keeps model loaded once for all images in a single process
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List
from camera_pipeline import CameraPipelineFactory, CameraConfig, BatchConfig, create_default_camera_configs


def validate_resolution(height: int, width: int) -> tuple[int, int]:
    """Validate and adjust resolution to minimum usable values"""
    min_resolution = 256

    if height < min_resolution or width < min_resolution:
        print(f"⚠️  Warning: Resolution {width}x{height} is too small for meaningful results")
        print(f"   Auto-adjusting to minimum recommended resolution {min_resolution}x{min_resolution}")
        return min_resolution, min_resolution

    return height, width


def process_images_list(image_paths: List[str], rotate_angles: List[float], tilt_angles: List[int],
                         height: int, width: int, output_dir: str, device: str = "auto") -> None:
    """
    Process a list of images with camera controls in a single process
    """
    print(f"🚀 Starting true batch processing:")
    print(f"   Images to process: {len(image_paths)}")
    print(f"   Rotation angles: {rotate_angles}")
    print(f"   Tilt angles: {tilt_angles}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Output directory: {output_dir}")

    # Validate resolution
    height, width = validate_resolution(height, width)

    # Create camera configurations
    camera_configs = []

    # Generate all combinations of rotation and tilt
    for rotate in rotate_angles:
        for tilt in tilt_angles:
            config = CameraConfig(
                rotate_deg=rotate,
                vertical_tilt=tilt,
                seed=0,
                randomize_seed=True,
                true_guidance_scale=1.0,
                num_inference_steps=4,
                height=height,
                width=width
            )
            camera_configs.append(config)

    print(f"   Total configurations: {len(camera_configs)}")

    # Load model once for all images
    print(f"\n🔄 Loading model pipeline for all {len(image_paths)} images...")
    start_time = time.time()

    try:
        pipeline = CameraPipelineFactory.get_pipeline(device=device)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        print(f"   Model remains in memory for all images")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Process each image in sequence
    total_start = time.time()
    successful = 0
    failed = 0

    for i, input_path in enumerate(image_paths, 1):
        if not os.path.exists(input_path):
            print(f"\n⚠️  Skipping missing file: {input_path}")
            failed += 1
            continue

        print(f"\n📸 [{i}/{len(image_paths)}] Processing: {input_path}")

        # Create image ID
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
            img_start_time = time.time()
            results = pipeline.process_batch(batch_config)
            img_time = time.time() - img_start_time

            successful += 1
            print(f"   ✅ Generated {len(results)} images in {img_time:.1f} seconds")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1

    # Final summary
    total_time = time.time() - total_start
    total_images = successful * len(camera_configs)

    print(f"\n📊 Batch Processing Summary:")
    print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Successful images: {successful}/{len(image_paths)}")
    print(f"   Failed images: {failed}")
    print(f"   Average time per image: {total_time/max(len(image_paths), 1):.1f} seconds")
    print(f"   Total variations generated: {total_images}")
    print(f"   Processing speed: {total_images/total_time:.1f} images/second")

    # Cleanup model from RAM when done
    print(f"\n🧹 Cleaning up model from RAM...")
    CameraPipelineFactory.reset_cache()
    print(f"✅ Model removed from RAM")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process multiple images in a single process with persistent model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process multiple images with default angles
  python true_batch_processor.py --images img1.jpg img2.jpg img3.png

  # Process all images in a directory
  python true_batch_processor.py --image-dir "dataset/images/" --output "results/"

  # Custom camera angles
  python true_batch_processor.py --images img.jpg --rotate -45 0 45 --tilt -1 0 1

  # Custom resolution
  python true_batch_processor.py --images img.jpg --height 512 --width 512
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--images", nargs="+", help="List of image paths to process")
    input_group.add_argument("--image-dir", type=str, help="Directory containing images to process")

    # Camera parameters
    parser.add_argument("--rotate", nargs="+", type=float, default=[-180, -145, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180],
                       help="Rotation angles in degrees (default: all common angles)")
    parser.add_argument("--tilt", nargs="+", type=int, choices=[-1, 0, 1], default=[-1, 0, 1],
                       help="Vertical tilt angles (-1: bird's eye, 0: normal, 1: worm's eye)")

    # Output options
    parser.add_argument("--output", "-o", type=str, default="batch_output",
                       help="Output directory (default: batch_output)")

    # Resolution options
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height (default: 1024, minimum: 256)")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width (default: 1024, minimum: 256)")

    # Device options
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to use (default: auto)")

    return parser.parse_args()


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


def main():
    """Main function"""
    args = parse_args()

    # Get image paths
    if args.images:
        image_paths = args.images
        print(f"Processing {len(image_paths)} specified images...")
    else:
        image_paths = get_images_from_directory(args.image_dir)
        print(f"Found {len(image_paths)} images in directory: {args.image_dir}")

    if not image_paths:
        print("❌ No images to process")
        sys.exit(1)

    # Process all images in a single process
    process_images_list(
        image_paths=image_paths,
        rotate_angles=args.rotate,
        tilt_angles=args.tilt,
        height=args.height,
        width=args.width,
        output_dir=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()