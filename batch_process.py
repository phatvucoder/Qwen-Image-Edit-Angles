#!/usr/bin/env python3
"""
Optimized batch processing script for Qwen Image Edit Camera Control
Loads model once and processes multiple images efficiently
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from camera_pipeline import CameraPipeline, CameraPipelineFactory, create_default_camera_configs


class BatchProcessor:
    """Efficient batch processor that loads model once and processes multiple images"""

    def __init__(self, device: str = "auto"):
        """Initialize the batch processor with a single model loading"""
        self.device = self._determine_device(device)
        self.pipeline = None
        self.model_loaded = False

    def _determine_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _validate_resolution(self, height: int, width: int) -> tuple[int, int]:
        """Validate and adjust resolution to minimum usable values"""
        min_resolution = 256

        if height < min_resolution or width < min_resolution:
            print(f"⚠️  Warning: Resolution {width}x{height} is too small for meaningful results")
            print(f"   Auto-adjusting to {min_resolution}x{min_resolution}")
            return min_resolution, min_resolution

        return height, width

    def load_model(self):
        """Load the model pipeline once using factory pattern"""
        if self.model_loaded:
            return

        print("🔄 Loading model pipeline (this will take 1-2 minutes)...")
        start_time = time.time()

        try:
            # Use factory to get or create shared pipeline
            self.pipeline = CameraPipelineFactory.get_pipeline(device=self.device)
            self.model_loaded = True

            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.1f} seconds")
            print(f"   Device: {self.device}")
            print(f"   Pipeline optimized and ready for batch processing")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def process_single_image(
        self,
        input_path: str,
        output_dir: str,
        camera_configs: List[Dict[str, Any]],
        height: int = 1024,
        width: int = 1024
    ) -> Dict[str, Any]:
        """Process a single image with multiple camera configurations"""

        # Validate resolution
        height, width = self._validate_resolution(height, width)

        # Generate image identifier
        path_parts = Path(input_path).parts
        img_id = f"{path_parts[-2]}_{Path(input_path).stem.split('_')[-1]}"

        # Create output directory
        image_output_dir = Path(output_dir) / img_id
        image_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📸 Processing: {img_id}")
        print(f"   Input: {input_path}")
        print(f"   Output: {image_output_dir}")
        print(f"   Generating {len(camera_configs)} variations...")

        try:
            # Process batch for this image
            start_time = time.time()

            results = self.pipeline.process_batch(
                input_path=input_path,
                output_dir=str(image_output_dir),
                camera_configs=camera_configs,
                height=height,
                width=width
            )

            process_time = time.time() - start_time

            # Save processing metadata
            metadata = {
                "input_path": input_path,
                "image_id": img_id,
                "output_directory": str(image_output_dir),
                "processing_time_seconds": round(process_time, 2),
                "num_variations": len(camera_configs),
                "resolution": f"{width}x{height}",
                "device": self.device,
                "generated_images": results.get("saved_files", []),
                "camera_configs": camera_configs
            }

            metadata_file = image_output_dir / "batch_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"✅ Completed in {process_time:.1f}s - {len(camera_configs)} images generated")

            return {
                "success": True,
                "image_id": img_id,
                "processing_time": process_time,
                "num_variations": len(camera_configs),
                "output_dir": str(image_output_dir)
            }

        except Exception as e:
            print(f"❌ Failed to process {input_path}: {e}")
            return {
                "success": False,
                "image_id": img_id,
                "error": str(e)
            }

    def process_batch(
        self,
        input_paths: List[str],
        output_dir: str,
        camera_configs: Optional[List[Dict[str, Any]]] = None,
        height: int = 1024,
        width: int = 1024
    ) -> Dict[str, Any]:
        """Process multiple images with the same camera configurations"""

        # Load model once
        self.load_model()

        # Use default camera configs if none provided
        if camera_configs is None:
            camera_configs = create_default_camera_configs("all")
            print(f"Using default camera configurations ({len(camera_configs)} variants)")

        # Validate resolution
        height, width = self._validate_resolution(height, width)

        print(f"\n🚀 Starting batch processing:")
        print(f"   Images to process: {len(input_paths)}")
        print(f"   Variations per image: {len(camera_configs)}")
        print(f"   Total images to generate: {len(input_paths) * len(camera_configs)}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Output directory: {output_dir}")

        # Process all images
        batch_start_time = time.time()
        results = []

        for input_path in tqdm(input_paths, desc="Processing images"):
            if not os.path.exists(input_path):
                print(f"⚠️  Skipping missing file: {input_path}")
                continue

            result = self.process_single_image(
                input_path=input_path,
                output_dir=output_dir,
                camera_configs=camera_configs,
                height=height,
                width=width
            )
            results.append(result)

        # Batch summary
        batch_time = time.time() - batch_start_time
        successful = sum(1 for r in results if r.get("success", False))
        total_generated = successful * len(camera_configs)

        # Save batch summary
        summary = {
            "batch_processing_time_seconds": round(batch_time, 2),
            "total_input_images": len(input_paths),
            "successful_images": successful,
            "failed_images": len(input_paths) - successful,
            "variations_per_image": len(camera_configs),
            "total_images_generated": total_generated,
            "average_time_per_image": round(batch_time / max(len(input_paths), 1), 2),
            "resolution": f"{width}x{height}",
            "device": self.device,
            "results": results
        }

        summary_file = Path(output_dir) / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total time: {batch_time:.1f} seconds")
        print(f"   Successful images: {successful}/{len(input_paths)}")
        print(f"   Total variations generated: {total_generated}")
        print(f"   Average time per image: {batch_time / max(len(input_paths), 1):.1f} seconds")
        print(f"   Summary saved to: {summary_file}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Optimized batch processing for Qwen Image Edit Camera Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process multiple images with default angles
  python batch_process.py --images image1.jpg image2.jpg image3.png

  # Process all images in a directory
  python batch_process.py --image-dir "dataset/images" --output results/

  # Custom camera angles and resolution
  python batch_process.py --images img.jpg --rotate -45 0 45 --height 512 --width 512
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--images",
        nargs="+",
        help="List of image paths to process"
    )
    input_group.add_argument(
        "--image-dir",
        help="Directory containing images to process (supports jpg, png, webp)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        default="batch_output",
        help="Output directory for processed images (default: batch_output)"
    )

    # Camera options
    parser.add_argument(
        "--preset",
        choices=["default", "rotation_only", "all"],
        default="all",
        help="Camera angle preset to use (default: all)"
    )

    parser.add_argument(
        "--rotate",
        nargs="+",
        type=float,
        help="Custom rotation angles in degrees"
    )

    parser.add_argument(
        "--tilt",
        nargs="+",
        type=int,
        choices=[-1, 0, 1],
        help="Vertical tilt angles (-1: bird's eye, 0: normal, 1: worm's eye)"
    )

    # Resolution options
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height (default: 1024, minimum: 256)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width (default: 1024, minimum: 256)"
    )

    # Device option
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for processing (default: auto)"
    )

    args = parser.parse_args()

    # Get input image paths
    if args.image_dir:
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        input_paths = []
        for ext in image_extensions:
            input_paths.extend(Path(args.image_dir).glob(f"*{ext}"))
            input_paths.extend(Path(args.image_dir).glob(f"*{ext.upper()}"))
        input_paths = [str(p) for p in input_paths]

        if not input_paths:
            print(f"❌ No images found in directory: {args.image_dir}")
            return

    else:
        input_paths = args.images

    print(f"Found {len(input_paths)} images to process")

    # Create camera configurations
    if args.rotate or args.tilt:
        # Custom configuration
        from itertools import product

        rotations = args.rotate if args.rotate else [0]
        tilts = args.tilt if args.tilt else [0]

        camera_configs = []
        for rot, tilt in product(rotations, tilts):
            config = {
                "rotate_deg": rot,
                "move_forward": 0,
                "vertical_tilt": tilt,
                "wideangle": False,
                "seed": 0,
                "randomize_seed": True,
                "true_guidance_scale": 1.0,
                "num_inference_steps": 4,
                "height": args.height,
                "width": args.width
            }
            camera_configs.append(config)

    else:
        # Use preset
        camera_configs = create_default_camera_configs(args.preset)

    # Create batch processor and run
    processor = BatchProcessor(device=args.device)

    try:
        results = processor.process_batch(
            input_paths=input_paths,
            output_dir=args.output,
            camera_configs=camera_configs,
            height=args.height,
            width=args.width
        )

        print(f"\n🎉 Batch processing completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Batch processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Batch processing failed: {e}")
        raise


if __name__ == "__main__":
    main()