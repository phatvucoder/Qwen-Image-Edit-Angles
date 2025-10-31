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

from huggingface_hub import InferenceClient
import math
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import os
import base64
from io import BytesIO
import json
import time  # Added for history update delay

from gradio_client import Client, handle_file
import tempfile
from PIL import Image
import os
import gradio as gr

def turn_into_video(input_images, output_images, prompt, progress=gr.Progress(track_tqdm=True)):
    if not input_images or not output_images:
        raise gr.Error("Please generate an output image first.")

    progress(0.02, desc="Preparing images...")

    def extract_pil(img_entry):
        if isinstance(img_entry, tuple) and isinstance(img_entry[0], Image.Image):
            return img_entry[0]
        elif isinstance(img_entry, Image.Image):
            return img_entry
        elif isinstance(img_entry, str):
            return Image.open(img_entry)
        else:
            raise gr.Error(f"Unsupported image format: {type(img_entry)}")

    start_img = extract_pil(input_images[0])
    end_img   = extract_pil(output_images[0])

    progress(0.10, desc="Saving temp files...")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_start, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_end:
        start_img.save(tmp_start.name)
        end_img.save(tmp_end.name)

    progress(0.20, desc="Connecting to Wan space...")

    client = Client("multimodalart/wan-2-2-first-last-frame")  

    progress(0.35, desc="Generating video...")

    video_path, seed = client.predict(
        start_image_pil=handle_file(tmp_start.name),
        end_image_pil=handle_file(tmp_end.name),
        prompt=prompt or "smooth cinematic transition",
        api_name="/generate_video"
    )

    progress(0.95, desc="Finalizing...")
    print(video_path)
    return video_path['video']




SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  
Please strictly follow the rewriting rules below:
## 1. General Principles
- Keep the rewritten prompt **concise and comprehensive**. Avoid overly long sentences and unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the main part of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the scene in the input images.  
- If multiple sub-images are to be generated, describe the content of each sub-image individually.  
## 2. Task-Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  
### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Keep the original language of the text, and keep the capitalization.  
- Both adding new text and replacing existing text are text replacement tasks, For example:  
    - Replace "xx" to "yy"  
    - Replace the mask / bounding box to "yy"  
    - Replace the visual object to "yy"  
- Specify text position, color, and layout only if user has required.  
- If font is specified, keep the original language of the font.  
### 3. Human Editing Tasks
- Make the smallest changes to the given user's prompt.  
- If changes to background, action, expression, camera shot, or ambient lighting are required, please list each modification individually.
- **Edits to makeup or facial features / expression must be subtle, not exaggerated, and must preserve the subject's identity consistency.**
    > Original: "Add eyebrows to the face"  
    > Rewritten: "Slightly thicken the person's eyebrows with little change, look natural."
### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco style: flashing lights, disco ball, mirrored walls, vibrant colors"  
- For style reference, analyze the original image and extract key characteristics (color, composition, texture, lighting, artistic style, etc.), integrating them into the instruction.  
- **Colorization tasks (including old photo restoration) must use the fixed template:**  
  "Restore and colorize the old photo."  
- Clearly specify the object to be modified. For example:  
    > Original: Modify the subject in Picture 1 to match the style of Picture 2.  
    > Rewritten: Change the girl in Picture 1 to the ink-wash style of Picture 2 — rendered in black-and-white watercolor with soft color transitions.
### 5. Material Replacement
- Clearly specify the object and the material. For example: "Change the material of the apple to papercut style."
- For text material replacement, use the fixed template:
    "Change the material of text "xxxx" to laser style"
### 6. Logo/Pattern Editing
- Material replacement should preserve the original shape and structure as much as possible. For example:
   > Original: "Convert to sapphire material"  
   > Rewritten: "Convert the main subject in the image to sapphire material, preserving similar shape and structure"
- When migrating logos/patterns to new scenes, ensure shape and structure consistency. For example:
   > Original: "Migrate the logo in the image to a new scene"  
   > Rewritten: "Migrate the logo in the image to a new scene, preserving similar shape and structure"
### 7. Multi-Image Tasks
- Rewritten prompts must clearly point out which image's element is being modified. For example:  
    > Original: "Replace the subject of picture 1 with the subject of picture 2"  
    > Rewritten: "Replace the girl of picture 1 with the boy of picture 2, keeping picture 2's background unchanged"  
- For stylization tasks, describe the reference image's style in the rewritten prompt, while preserving the visual content of the source image.  
## 3. Rationale and Logic Check
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" requires logical correction.
- Supplement missing critical information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, blank space, center/edge, etc.).
# Output Format Example
```json
{
   "Rewritten": "..."
}
'''


NEXT_SCENE_SYSTEM_PROMPT = '''
# Next Scene Prompt Generator
You are a cinematic AI director assistant. Your task is to analyze the provided image and generate a compelling "Next Scene" prompt that describes the natural cinematic progression from the current frame.
## Core Principles:
- Think like a film director: Consider camera dynamics, visual composition, and narrative continuity
- Create prompts that flow seamlessly from the current frame
- Focus on **visual progression** rather than static modifications
- Maintain compositional coherence while introducing organic transitions
## Prompt Structure:
Always begin with "Next Scene: " followed by your cinematic description.
## Key Elements to Include:
1. **Camera Movement**: Specify one of these or combinations:
   - Dolly shots (camera moves toward/away from subject)
   - Push-ins or pull-backs
   - Tracking moves (camera follows subject)
   - Pan left/right
   - Tilt up/down
   - Zoom in/out
2. **Framing Evolution**: Describe how the shot composition changes:
   - Wide to close-up transitions
   - Angle shifts (high angle to eye level, etc.)
   - Reframing of subjects
   - Revealing new elements in frame
3. **Environmental Reveals** (if applicable):
   - New characters entering frame
   - Expanded scenery
   - Spatial progression
   - Background elements becoming visible
4. **Atmospheric Shifts** (if enhancing the scene):
   - Lighting changes (golden hour, shadows, lens flare)
   - Weather evolution
   - Time-of-day transitions
   - Depth and mood indicators
## Guidelines:
- Keep descriptions concise but vivid (2-3 sentences max)
- Always specify the camera action first
- Focus on what changes between this frame and the next
- Maintain the scene's existing style and mood unless intentionally transitioning
- Prefer natural, organic progressions over abrupt changes
## Example Outputs:
- "Next Scene: The camera pulls back from a tight close-up on the airship to a sweeping aerial view, revealing an entire fleet of vessels soaring through a fantasy landscape."
- "Next Scene: The camera tracks forward and tilts down, bringing the sun and helicopters closer into frame as a strong lens flare intensifies."
- "Next Scene: The camera pans right, removing the dragon and rider from view while revealing more of the floating mountain range in the distance."
- "Next Scene: The camera moves slightly forward as sunlight breaks through the clouds, casting a soft glow around the character's silhouette in the mist. Realistic cinematic style, atmospheric depth."
## Output Format:
Return ONLY the next scene prompt as plain text, starting with "Next Scene: "
Do NOT include JSON formatting or additional explanations.
'''

# --- Prompt Enhancement using Hugging Face InferenceClient ---
def polish_prompt_hf(original_prompt, img_list):
    """
    Rewrites the prompt using a Hugging Face InferenceClient.
    """
    # Ensure HF_TOKEN is set
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Warning: HF_TOKEN not set. Falling back to original prompt.")
        return original_prompt

    try:
        # Initialize the client
        prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {original_prompt}\n\nRewritten Prompt:"
        client = InferenceClient(
            provider="nebius",
            api_key=api_key,
        )

        # Format the messages for the chat completions API
        sys_promot = "you are a helpful assistant, you should provide useful answers to users."
        messages = [
            {"role": "system", "content": sys_promot},
            {"role": "user", "content": []}]
        for img in img_list:
            messages[1]["content"].append(
                {"image": f"data:image/png;base64,{encode_image(img)}"})
        messages[1]["content"].append({"text": f"{prompt}"})

        # Call the API
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=messages,
        )
        
        # Parse the response
        result = completion.choices[0].message.content
        
        # Try to extract JSON if present
        if '"Rewritten"' in result:
            try:
                # Clean up the response
                result = result.replace('```json', '').replace('```', '')
                result_json = json.loads(result)
                polished_prompt = result_json.get('Rewritten', result)
            except:
                polished_prompt = result
        else:
            polished_prompt = result
            
        polished_prompt = polished_prompt.strip().replace("\n", " ")
        return polished_prompt
        
    except Exception as e:
        print(f"Error during API call to Hugging Face: {e}")
        # Fallback to original prompt if enhancement fails
        return original_prompt
    
def next_scene_prompt(original_prompt, img_list):
    """
    Rewrites the prompt using a Hugging Face InferenceClient.
    Supports multiple images via img_list.
    """
    # Ensure HF_TOKEN is set
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Warning: HF_TOKEN not set. Falling back to original prompt.")
        return original_prompt
    prompt = f"{NEXT_SCENE_SYSTEM_PROMPT}"
    system_prompt = "you are a helpful assistant, you should provide useful answers to users."
    try:
        # Initialize the client
        client = InferenceClient(
            provider="nebius",
            api_key=api_key,
        )

        # Convert list of images to base64 data URLs
        image_urls = []
        if img_list is not None:
            # Ensure img_list is actually a list
            if not isinstance(img_list, list):
                img_list = [img_list]
            
            for img in img_list:
                image_url = None
                # If img is a PIL Image
                if hasattr(img, 'save'):  # Check if it's a PIL Image
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_url = f"data:image/png;base64,{img_base64}"
                # If img is already a file path (string)
                elif isinstance(img, str):
                    with open(img, "rb") as image_file:
                        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    image_url = f"data:image/png;base64,{img_base64}"
                else:
                    print(f"Warning: Unexpected image type: {type(img)}, skipping...")
                    continue
                
                if image_url:
                    image_urls.append(image_url)

        # Build the content array with text first, then all images
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        # Add all images to the content
        for image_url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })

        # Format the messages for the chat completions API
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content
            }
        ]

        # Call the API
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=messages,
        )
        
        # Parse the response
        result = completion.choices[0].message.content
        
        # Try to extract JSON if present
        if '"Rewritten"' in result:
            try:
                # Clean up the response
                result = result.replace('```json', '').replace('```', '')
                result_json = json.loads(result)
                polished_prompt = result_json.get('Rewritten', result)
            except:
                polished_prompt = result
        else:
            polished_prompt = result
            
        polished_prompt = polished_prompt.strip().replace("\n", " ")
        return polished_prompt
        
    except Exception as e:
        print(f"Error during API call to Hugging Face: {e}")
        # Fallback to original prompt if enhancement fails
        return original_prompt 


def update_history(new_images, history):
    """Updates the history gallery with the new images."""
    time.sleep(0.5)  # Small delay to ensure images are ready
    if history is None:
        history = []
    if new_images is not None and len(new_images) > 0:
        if not isinstance(history, list):
            history = list(history) if history else []
        for img in new_images:
            history.insert(0, img)
    history = history[:20]  # Keep only last 20 images
    return history

def use_history_as_input(evt: gr.SelectData):
    """Sets the selected history image as the new input image."""
    if evt.value is not None:
        return gr.update(value=[(evt.value,)])
    return gr.update()
    
def encode_image(pil_image):
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", 
                                                transformer= QwenImageTransformer2DModel.from_pretrained("linoyts/Qwen-Image-Edit-Rapid-AIO", 
                                                                                                         subfolder='transformer',
                                                                                                         torch_dtype=dtype,
                                                                                                         device_map='cuda'),torch_dtype=dtype).to(device)

pipe.load_lora_weights(
        "lovis93/next-scene-qwen-image-lora-2509", 
        weight_name="next-scene_lora-v2-3000.safetensors", adapter_name="next-scene"
    )
pipe.set_adapters(["next-scene"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["next-scene"], lora_scale=1.)
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
    rewrite_prompt=True,
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
    if rewrite_prompt and len(pil_images) > 0:
        prompt = polish_prompt_hf(prompt, pil_images)
        print(f"Rewritten Prompt: {prompt}")
    

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
                        
                        
                        rewrite_prompt = gr.Checkbox(label="Rewrite prompt", value=False)

        

            with gr.Column():
                result = gr.Gallery(label="Result", show_label=False, type="pil")
                with gr.Row():
                    use_output_btn = gr.Button("↗️ Use as input", variant="secondary", size="sm", visible=False)
                    turn_video_btn = gr.Button("🎬 Turn into Video", variant="secondary", size="sm", visible=False)
                output_video = gr.Video(label="Generated Video", autoplay=True, visible=False)

                with gr.Row(visible=False):
                    gr.Markdown("### 📜 History")
                    clear_history_button = gr.Button("🗑️ Clear History", size="sm", variant="stop")
                
                history_gallery = gr.Gallery(
                    label="Click any image to use as input", 
                    interactive=False,
                    show_label=True,
                    visible=False
                )

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
            rewrite_prompt,
        ],
        outputs=[result, seed, use_output_btn, turn_video_btn],

    ).then(
    fn=update_history,
    inputs=[result, history_gallery],
    outputs=history_gallery,

    )

    # Add the new event handler for the "Use Output as Input" button
    use_output_btn.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_images]
    )

    # History gallery event handlers
    history_gallery.select(
        fn=use_history_as_input,
        inputs=None,
        outputs=[input_images],

    )
    
    clear_history_button.click(
        fn=lambda: [],
        inputs=None,
        outputs=history_gallery,

    )

    input_images.change(fn=suggest_next_scene_prompt, inputs=[input_images], outputs=[prompt])

    turn_video_btn.click(
    fn=lambda: gr.update(visible=True),   
    inputs=None,
    outputs=[output_video],
).then(
    fn=turn_into_video,                   
    inputs=[input_images, result, prompt],
    outputs=[output_video],
)


if __name__ == "__main__":
    demo.launch()