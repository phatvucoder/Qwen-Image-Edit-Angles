import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
from diffusers import QwenImageEditPlusPipeline
import math

import os
import base64
import json

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
- **Edits to makeup or facial features / expression must be subtle, not exaggerated, and must preserve the subject’s identity consistency.**
    > Original: "Add eyebrows to the face"  
    > Rewritten: "Slightly thicken the person’s eyebrows with little change, look natural."

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
- Rewritten prompts must clearly point out which image’s element is being modified. For example:  
    > Original: "Replace the subject of picture 1 with the subject of picture 2"  
    > Rewritten: "Replace the girl of picture 1 with the boy of picture 2, keeping picture 2’s background unchanged"  
- For stylization tasks, describe the reference image’s style in the rewritten prompt, while preserving the visual content of the source image.  

## 3. Rationale and Logic Check
- Resolve contradictory instructions: e.g., “Remove all trees but keep all trees” requires logical correction.
- Supplement missing critical information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, blank space, center/edge, etc.).

# Output Format Example
```json
{
   "Rewritten": "..."
}
'''
# --- Prompt Enhancement using Hugging Face InferenceClient ---
def polish_prompt_hf(prompt, img):
    """
    Rewrites the prompt using a Hugging Face InferenceClient.
    """
    # Ensure HF_TOKEN is set
    api_key = os.environ.get("HF_TOKEN")
    prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
    if not api_key:
        print("Warning: HF_TOKEN not set. Falling back to original prompt.")
        return original_prompt

    try:
        # Initialize the client
        client = InferenceClient(
            provider="cerebras",
            api_key=api_key,
        )

        # Format the messages for the chat completions API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

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
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            messages=messages,
        )
        
        # Parse the response
        result = completion.choices[0].message.content
        
        # Try to extract JSON if present
        if '{"Rewritten"' in result:
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
        
# def polish_prompt(prompt, img):
#     prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
#     success=False
#     while not success:
#         try:
#             result = api(prompt, [img])
#             # print(f"Result: {result}")
#             # print(f"Polished Prompt: {polished_prompt}")
#             if isinstance(result, str):
#                 result = result.replace('```json','')
#                 result = result.replace('```','')
#                 result = json.loads(result)
#             else:
#                 result = json.loads(result)

#             polished_prompt = result['Rewritten']
#             polished_prompt = polished_prompt.strip()
#             polished_prompt = polished_prompt.replace("\n", " ")
#             success = True
#         except Exception as e:
#             print(f"[Warning] Error during API call: {e}")
#     return polished_prompt


def encode_image(pil_image):
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# def api(prompt, img_list, model="qwen-vl-max-latest", kwargs={}):
#     import dashscope
#     api_key = os.environ.get('DASH_API_KEY')
#     if not api_key:
#         raise EnvironmentError("DASH_API_KEY is not set")
#     assert model in ["qwen-vl-max-latest"], f"Not implemented model {model}"
#     sys_promot = "you are a helpful assistant, you should provide useful answers to users."
#     messages = [
#         {"role": "system", "content": sys_promot},
#         {"role": "user", "content": []}]
#     for img in img_list:
#         messages[1]["content"].append(
#             {"image": f"data:image/png;base64,{encode_image(img)}"})
#     messages[1]["content"].append({"text": f"{prompt}"})

#     response_format = kwargs.get('response_format', None)

#     response = dashscope.MultiModalConversation.call(
#         api_key=api_key,
#         model=model, # For example, use qwen-plus here. You can change the model name as needed. Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
#         messages=messages,
#         result_format='message',
#         response_format=response_format,
#         )

#     if response.status_code == 200:
#         return response.output.choices[0].message.content[0]['text']
#     else:
#         raise Exception(f'Failed to post: {response}')

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Scheduler configuration for Lightning
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Initialize scheduler with Lightning config
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# Load the model pipeline
pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", 
                                                 scheduler=scheduler,
                                                 torch_dtype=dtype).to(device)
pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", 
        weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
    )
pipe.fuse_lora()

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

# --- Main Inference Function (with hardcoded negative prompt) ---
@spaces.GPU(duration=300)
def infer(
    images,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=8,
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

    return image, seed

# --- Examples and UI Layout ---
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML('<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">')
        gr.Markdown("[Learn more](https://github.com/QwenLM/Qwen-Image) about the Qwen-Image series. Try on [Qwen Chat](https://chat.qwen.ai/), or [download model](https://huggingface.co/Qwen/Qwen-Image-Edit) to run locally with ComfyUI or diffusers.")
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input Images", 
                                          show_label=False, 
                                          type="pil", 
                                          interactive=True)

            # result = gr.Image(label="Result", show_label=False, type="pil")
            result = gr.Gallery(label="Result", show_label=False, type="pil")
        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="describe the edit instruction",
                    container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            # Negative prompt UI element is removed here

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
                    value=8,
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

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

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
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch()