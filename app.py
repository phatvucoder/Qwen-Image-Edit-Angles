import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from optimization import optimize_pipeline_
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3


import math
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import os

import tempfile
from PIL import Image
import os
import gradio as gr


# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", 
                                                transformer= QwenImageTransformer2DModel.from_pretrained("linoyts/Qwen-Image-Edit-Rapid-AIO", 
                                                                                                         subfolder='transformer',
                                                                                                         torch_dtype=dtype,
                                                                                                         device_map='cuda'),torch_dtype=dtype).to(device)

pipe.load_lora_weights(
        "dx8152/Qwen-Edit-2509-Multiple-angles", 
        weight_name="镜头转换.safetensors", adapter_name="angles"
    )
pipe.set_adapters(["angles"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.)
pipe.unload_lora_weights()


# Apply the same optimizations from the first version
pipe.transformer.__class__ = QwenImageTransformer2DModel
pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

# --- Ahead-of-time compilation ---
optimize_pipeline_(pipe, image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))], prompt="prompt")

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

def use_output_as_input(output_images):
    """Convert output images to input format for the gallery"""
    if output_images is None or len(output_images) == 0:
        return []
    return output_images

def suggest_next_scene_prompt(images):
    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item[0], Image.Image):
                    pil_images.append(item[0].convert("RGB"))
                elif isinstance(item[0], str):
                    pil_images.append(Image.open(item[0]).convert("RGB"))
                elif hasattr(item, "name"):
                    pil_images.append(Image.open(item.name).convert("RGB"))
            except Exception:
                continue
    if len(pil_images) > 0:
        prompt = next_scene_prompt("", pil_images)
    else:
        prompt = ""
    print("next scene prompt: ", prompt)
    return prompt

# --- Main Inference Function (with hardcoded negative prompt) ---
@spaces.GPU(duration=300)
def infer(
    images,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generates an image using the local Qwen-Image diffusers pipeline.
    """
    # Hardcode the negative prompt as requested
    negative_prompt = " "
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Load input images into PIL Images
    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item[0], Image.Image):
                    pil_images.append(item[0].convert("RGB"))
                elif isinstance(item[0], str):
                    pil_images.append(Image.open(item[0]).convert("RGB"))
                elif hasattr(item, "name"):
                    pil_images.append(Image.open(item.name).convert("RGB"))
            except Exception:
                continue

    if height==256 and width==256:
        height, width = None, None
    print(f"Calling pipeline with prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}, Size: {width}x{height}")
    

    # Generate the image
    image = pipe(
        image=pil_images if len(pil_images) > 0 else None,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    ).images

    # Return images, seed, and make button visible
    return image, seed, gr.update(visible=True), gr.update(visible=True)


# --- Examples and UI Layout ---
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#logo-title {
    text-align: center;
}
#logo-title img {
    width: 400px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <div id="logo-title">
            <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Edit Logo" width="400" style="display: block; margin: 0 auto;">
            <h2 style="font-style: italic;color: #5b47d1;margin-top: -27px !important;margin-left: 96px">Next Scene 🎬</h2>
        </div>
        """)
        gr.Markdown("""
        This demo uses the new [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) with [lovis93/next-scene-qwen-image-lora](https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509) for cinematic image sequences with natural visual progression from frame to frame 🎥 and [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main) + [AoT compilation & FA3](https://huggingface.co/blog/zerogpu-aoti) for accelerated 4-step inference.
        Try on [Qwen Chat](https://chat.qwen.ai/), or [download model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) to run locally with ComfyUI or diffusers.
        """)
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input Images", 
                                          show_label=False, 
                                          type="pil", 
                                          interactive=True)

                prompt = gr.Text(
                    label="Prompt 🪄",
                    show_label=True,
                    placeholder="Next scene: The camera dollies in to a tight close-up...",
            )
                run_button = gr.Button("Edit!", variant="primary")
                
                with gr.Accordion("Advanced Settings", open=False):
                    
        
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
        
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        
                    with gr.Row():
        
                        true_guidance_scale = gr.Slider(
                            label="True guidance scale",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            value=1.0
                        )

                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=40,
                            step=1,
                            value=4,
                        )
                        
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=2048,
                            step=8,
                            value=None,
                        )
                        
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=2048,
                            step=8,
                            value=None,
                        )
                        

        

            with gr.Column():
                result = gr.Gallery(label="Result", show_label=False, type="pil")
                with gr.Row():
                    use_output_btn = gr.Button("↗️ Use as input", variant="secondary", size="sm", visible=False)

                

        gr.Examples(examples=[
            [["disaster_girl.jpg", "grumpy.png"], "Next Scene: the camera zooms in, showing the cat walking away from the fire"],
            [["wednesday.png"], "Next Scene: The camera pulls back and rises to an elevated angle, revealing the full dance floor with the choreographed movements of all dancers as the central figure becomes part of the larger ensemble."],
            ],
                inputs=[input_images, prompt], 
                outputs=[result, seed], 
                fn=infer, 
                cache_examples="lazy")


        

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_images,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            height,
            width,
        ],
        outputs=[result, seed, use_output_btn],

    )

    # Add the new event handler for the "Use Output as Input" button
    use_output_btn.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_images]
    )


if __name__ == "__main__":
    demo.launch()