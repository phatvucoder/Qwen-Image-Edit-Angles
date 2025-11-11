#!/usr/bin/env python3
"""
Camera Control Pipeline Module

Extracted from app.py to provide CLI functionality for batch camera angle processing.
"""

import os
import random
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import requests
from urllib.parse import urlparse
import json

from diffusers import FlowMatchEulerDiscreteScheduler
from optimization import optimize_pipeline_
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

MAX_SEED = np.iinfo(np.int32).max


@dataclass
class CameraConfig:
    """Configuration for camera parameters"""
    rotate_deg: float = 0.0
    move_forward: float = 0.0
    vertical_tilt: int = 0  # -1, 0, or 1
    wideangle: bool = False
    seed: int = 0
    true_guidance_scale: float = 1.0
    num_inference_steps: int = 4
    height: int = 1024
    width: int = 1024
    randomize_seed: bool = True


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    input_path: str = ""
    input_url: str = ""
    output_dir: str = "output"
    camera_configs: List[CameraConfig] = None

    def __post_init__(self):
        if self.camera_configs is None:
            self.camera_configs = []


class CameraPipeline:
    """Main camera control pipeline class"""

    def __init__(self, device: str = "auto"):
        """
        Initialize the camera pipeline

        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.device = self._get_device(device)
        self.dtype = torch.bfloat16
        self.pipe = None
        self._load_pipeline()

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_pipeline(self):
        """Load and configure the Qwen Image Edit pipeline"""
        print(f"Loading pipeline on device: {self.device}")

        # Load base pipeline
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            transformer=QwenImageTransformer2DModel.from_pretrained(
                "linoyts/Qwen-Image-Edit-Rapid-AIO",
                subfolder='transformer',
                torch_dtype=self.dtype,
                device_map='cuda' if self.device == 'cuda' else None
            ),
            torch_dtype=self.dtype
        ).to(self.device)

        # Load LoRA weights
        self.pipe.load_lora_weights(
            "dx8152/Qwen-Edit-2509-Multiple-angles",
            weight_name="镜头转换.safetensors",
            adapter_name="angles"
        )

        # Configure adapters
        self.pipe.set_adapters(["angles"], adapter_weights=[1.])
        self.pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
        self.pipe.unload_lora_weights()

        # Set attention processor
        self.pipe.transformer.__class__ = QwenImageTransformer2DModel
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

        # Optimize pipeline
        print("Optimizing pipeline...")
        optimize_pipeline_(
            self.pipe,
            image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))],
            prompt="prompt"
        )

        print("Pipeline loaded successfully!")

    @staticmethod
    def build_camera_prompt(config: CameraConfig) -> str:
        """Build camera movement prompt from configuration"""
        prompt_parts = []

        # Rotation
        if config.rotate_deg != 0:
            direction = "left" if config.rotate_deg > 0 else "right"
            if direction == "left":
                prompt_parts.append(f"将镜头向左旋转{abs(config.rotate_deg)}度 Rotate the camera {abs(config.rotate_deg)} degrees to the left.")
            else:
                prompt_parts.append(f"将镜头向右旋转{abs(config.rotate_deg)}度 Rotate the camera {abs(config.rotate_deg)} degrees to the right.")

        # Move forward / close-up
        if config.move_forward > 5:
            prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
        elif config.move_forward >= 1:
            prompt_parts.append("将镜头向前移动 Move the camera forward.")

        # Vertical tilt
        if config.vertical_tilt <= -1:
            prompt_parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
        elif config.vertical_tilt >= 1:
            prompt_parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")

        # Lens option
        if config.wideangle:
            prompt_parts.append(" 将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

        final_prompt = " ".join(prompt_parts).strip()
        return final_prompt if final_prompt else "no camera movement"

    @staticmethod
    def load_image(input_path: str, input_url: str = "") -> Image.Image:
        """
        Load image from file path or URL

        Args:
            input_path: Local file path
            input_url: URL to download image from

        Returns:
            PIL Image object
        """
        if input_url:
            print(f"Downloading image from URL: {input_url}")
            response = requests.get(input_url, stream=True)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")
        elif input_path:
            print(f"Loading image from path: {input_path}")
            return Image.open(input_path).convert("RGB")
        else:
            raise ValueError("Either input_path or input_url must be provided")

    def process_single_image(self, image: Image.Image, config: CameraConfig) -> tuple[Image.Image, int, str]:
        """
        Process a single image with camera configuration

        Args:
            image: Input PIL Image
            config: Camera configuration

        Returns:
            Tuple of (output_image, seed_used, prompt_used)
        """
        prompt = self.build_camera_prompt(config)
        print(f"Generated Prompt: {prompt}")

        if prompt == "no camera movement":
            return image, config.seed, prompt

        if config.randomize_seed:
            seed = random.randint(0, MAX_SEED)
        else:
            seed = config.seed

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Process image
        result = self.pipe(
            image=[image],
            prompt=prompt,
            height=config.height if config.height != 0 else None,
            width=config.width if config.width != 0 else None,
            num_inference_steps=config.num_inference_steps,
            generator=generator,
            true_cfg_scale=config.true_guidance_scale,
            num_images_per_prompt=1,
        ).images[0]

        return result, seed, prompt

    def process_batch(self, batch_config: BatchConfig) -> List[Dict[str, Any]]:
        """
        Process batch of camera configurations

        Args:
            batch_config: Batch processing configuration

        Returns:
            List of results with metadata
        """
        # Load input image
        input_image = self.load_image(batch_config.input_path, batch_config.input_url)

        # Create output directory
        output_dir = Path(batch_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        # Save original image
        original_path = output_dir / "00_original.png"
        input_image.save(original_path)
        print(f"Saved original image to: {original_path}")

        # Process each camera configuration
        for i, config in enumerate(batch_config.camera_configs):
            print(f"\nProcessing configuration {i+1}/{len(batch_config.camera_configs)}")

            # Update config dimensions based on input image if not specified
            if config.height == 1024 and config.width == 1024:
                orig_width, orig_height = input_image.size
                if orig_width > orig_height:
                    new_width = 1024
                    aspect_ratio = orig_height / orig_width
                    new_height = int(new_width * aspect_ratio)
                else:
                    new_height = 1024
                    aspect_ratio = orig_width / orig_height
                    new_width = int(new_height * aspect_ratio)

                # Ensure dimensions are multiples of 8
                config.width = (new_width // 8) * 8
                config.height = (new_height // 8) * 8

            # Process image
            output_image, seed_used, prompt_used = self.process_single_image(input_image, config)

            # Generate filename
            filename_parts = []
            if config.rotate_deg != 0:
                filename_parts.append(f"rot_{config.rotate_deg}deg")
            if config.move_forward != 0:
                filename_parts.append(f"move_{config.move_forward}")
            if config.vertical_tilt != 0:
                filename_parts.append(f"tilt_{config.vertical_tilt}")
            if config.wideangle:
                filename_parts.append("wide")

            filename = f"{i+1:02d}_" + "_".join(filename_parts) if filename_parts else f"{i+1:02d}_no_change"
            filename += ".png"

            # Save result
            output_path = output_dir / filename
            output_image.save(output_path)

            # Store result metadata
            result = {
                "filename": str(output_path),
                "config": config,
                "seed_used": seed_used,
                "prompt_used": prompt_used,
                "index": i + 1
            }
            results.append(result)

            print(f"Saved result to: {output_path}")
            print(f"Seed used: {seed_used}")

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        metadata = {
            "input_path": batch_config.input_path,
            "input_url": batch_config.input_url,
            "output_dir": batch_config.output_dir,
            "results": [
                {
                    "filename": r["filename"],
                    "index": r["index"],
                    "config": {
                        "rotate_deg": r["config"].rotate_deg,
                        "move_forward": r["config"].move_forward,
                        "vertical_tilt": r["config"].vertical_tilt,
                        "wideangle": r["config"].wideangle,
                        "seed": r["config"].seed,
                        "true_guidance_scale": r["config"].true_guidance_scale,
                        "num_inference_steps": r["config"].num_inference_steps,
                        "height": r["config"].height,
                        "width": r["config"].width
                    },
                    "seed_used": r["seed_used"],
                    "prompt_used": r["prompt_used"]
                }
                for r in results
            ]
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nSaved metadata to: {metadata_path}")
        print(f"Processing complete! Generated {len(results)} images in {output_dir}")

        return results


def create_default_camera_configs() -> List[CameraConfig]:
    """Create default camera configurations for common angles"""
    configs = []

    # Rotation angles
    for angle in [-90, -45, 45, 90]:
        configs.append(CameraConfig(rotate_deg=angle))

    # Vertical tilts
    configs.append(CameraConfig(vertical_tilt=-1))  # Bird's eye view
    configs.append(CameraConfig(vertical_tilt=1))   # Worm's eye view

    # Movement
    configs.append(CameraConfig(move_forward=5))    # Forward
    configs.append(CameraConfig(move_forward=8))    # Close-up

    # Wide angle
    configs.append(CameraConfig(wideangle=True))

    # Combinations
    configs.append(CameraConfig(rotate_deg=45, wideangle=True))
    configs.append(CameraConfig(rotate_deg=-45, move_forward=3))

    return configs