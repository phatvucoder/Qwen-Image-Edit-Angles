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

# pipe.load_lora_weights(
#         "lovis93/next-scene-qwen-image-lora-2509", 
#         weight_name="next-scene_lora-v2-3000.safetensors", adapter_name="next-scene"
#     )
pipe.set_adapters(["angles"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
# pipe.fuse_lora(adapter_names=["next-scene"], lora_scale=1.)
pipe.unload_lora_weights()



pipe.transformer.__class__ = QwenImageTransformer2DModel
pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

optimize_pipeline_(pipe, image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))], prompt="prompt")


MAX_SEED = np.iinfo(np.int32).max

def build_camera_prompt(rotate_deg, move_forward, vertical_tilt, wideangle):
    prompt_parts = []

    # Rotation
    if rotate_deg != 0:
        direction = "left" if rotate_deg > 0 else "right"
        if direction == "left":
            prompt_parts.append(f"将镜头向左旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the left.")
        else:
            prompt_parts.append(f"将镜头向右旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the right.")


    # Move forward / close-up
    if move_forward >= 5:
        prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
    elif move_forward >= 1:
        prompt_parts.append("将镜头向前移动 Move the camera forward.")

    # Vertical tilt
    if vertical_tilt <= -1:
        prompt_parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
    elif vertical_tilt >= 1:
        prompt_parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")

    # Lens option
    if wideangle:
        prompt_parts.append(" 将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

    final_prompt = " ".join(prompt_parts).strip()
    return final_prompt if final_prompt else "no camera movement"


@spaces.GPU
def infer_camera_edit(
    image,
    prev_output,
    rotate_deg,
    move_forward,
    vertical_tilt,
    wideangle,
    seed,
    randomize_seed,
    true_guidance_scale,
    num_inference_steps,
    height,
    width,
):
    prompt = build_camera_prompt(rotate_deg, move_forward, vertical_tilt, wideangle)
    print(f"Generated Prompt: {prompt}")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Choose input image (prefer uploaded, else last output)
    pil_images = []
    if image is not None:
        if isinstance(image, Image.Image):
            pil_images.append(image.convert("RGB"))
        elif hasattr(image, "name"):
            pil_images.append(Image.open(image.name).convert("RGB"))
    elif prev_output is not None:
        pil_images.append(prev_output.convert("RGB"))

    if len(pil_images) == 0:
        raise gr.Error("Please upload an image first.")

    if prompt == "no camera movement":
        return image, seed, prompt
    result = pipe(
        image=pil_images,
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    return result, seed, prompt


# --- UI ---
css = "#col-container { max-width: 800px; margin: 0 auto; }"

is_reset = gr.State(value=False)

def reset_all():
    return [0, 0, 0, 0, False, True]

def end_reset():
    return False


with gr.Blocks(theme=gr.themes.Citrus(), css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("## 🎬 Qwen Image Edit — Camera Angle Control")
        gr.Markdown("""
            Qwen Image Edit 2509 for Camera Control ✨ 
            Using [dx8152's Qwen-Edit-2509-Multiple-angles LoRA](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles) and [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main) for 4-step inference 💨
            """
        )

        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input Image", type="pil", sources=["upload"])
                prev_output = gr.State(value=None)
                is_reset = gr.State(value=False)

                with gr.Tab("Camera Controls"):
                    rotate_deg = gr.Slider(label="Rotate Left–Right (°)", minimum=-90, maximum=90, step=45, value=0)
                    move_forward = gr.Slider(label="Move Forward → Close-Up", minimum=0, maximum=10, step=5, value=0)
                    vertical_tilt = gr.Slider(label="Vertical Angle (Bird ↔ Worm)", minimum=-1, maximum=1, step=1, value=0)
                    wideangle = gr.Checkbox(label="Wide-Angle Lens", value=False)
                with gr.Row():
                        reset_btn = gr.Button("Reset")
                        run_btn = gr.Button("Generate", variant="primary")

                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    true_guidance_scale = gr.Slider(label="True Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=40, step=1, value=4)
                    height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024)
                    width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024)

                
                    

            with gr.Column():
                result = gr.Image(label="Output Image")
                prompt_preview = gr.Textbox(label="Processed Prompt", interactive=False)
                #gr.Markdown("_Each change applies a fresh camera instruction to the last output image._")

    inputs = [
        image, prev_output, rotate_deg, move_forward,
        vertical_tilt, wideangle,
        seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width
    ]
    outputs = [result, seed, prompt_preview]

    # Reset behavior
    reset_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[rotate_deg, move_forward, vertical_tilt, wideangle, is_reset],
        queue=False
    ).then(fn=end_reset, inputs=None, outputs=[is_reset], queue=False)

    # Manual generation
    run_event = run_btn.click(fn=infer_camera_edit, inputs=inputs, outputs=outputs)

    # Image upload resets
    image.change(
        fn=reset_all,
        inputs=None,
        outputs=[rotate_deg, move_forward, vertical_tilt, wideangle, is_reset],
        queue=False
    ).then(fn=end_reset, inputs=None, outputs=[is_reset], queue=False)

    # Live updates
    def maybe_infer(is_reset, *args):
        if is_reset:
            return gr.update(), gr.update(), gr.update()
        else:
            return infer_camera_edit(*args)

    control_inputs = [
        image, prev_output, rotate_deg, move_forward,
        vertical_tilt, wideangle,
        seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width
    ]
    control_inputs_with_flag = [is_reset] + control_inputs

    for control in [rotate_deg, move_forward, vertical_tilt, wideangle]:
        control.change(fn=maybe_infer, inputs=control_inputs_with_flag, outputs=outputs, show_progress="minimal")

    run_event.then(lambda img, *_: img, inputs=[result], outputs=[prev_output])

demo.launch()