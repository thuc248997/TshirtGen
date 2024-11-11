import torch
from diffusers import FluxPipeline
from PIL import Image
import os
from datetime import datetime
import peft

# Constants
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
GUIDANCE_SCALE = 0.8
NUM_INFERENCE_STEPS = 30
MAX_SEQUENCE_LENGTH = 512
SEED = 42

# # Initialize the FLUX pipeline
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev", # Base model
#     torch_dtype=torch.bfloat16
# )
# # Load LoRA weights
# pipe.load_lora_weights("jakedahn/flux-latentpop")
# pipe.enable_model_cpu_offload()

def save_image(image: Image.Image, output_dir: str = "outputs") -> str:
    """
    Save the generated image to a file.

    Args:
        image (Image.Image): The PIL Image to save
        output_dir (str): Directory to save the image in

    Returns:
        str: Path to the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flux_generated_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the image
    image.save(filepath)
    print(f"Image saved to: {filepath}")
    return filepath

def generate_fashion_outfit(content_print: str, lora_path: str) -> Image.Image:
    """
    Generate a image sample T-shirt with specified color and print content.

    Args:
        color (str): Color of the T-shirt
        content_print (str): Content to print on the T-shirt
        lora_path (str): Path to LoRA weights

    Returns:
        Image.Image: Generated PIL Image
    """
    # Initialize the pipeline for each generation to ensure clean state
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    
    # Load selected LoRA weights
    pipe.load_lora_weights(lora_path)
    pipe.enable_model_cpu_offload()

    prompt = f"{content_print}"

    image = pipe(
        prompt,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        generator=torch.Generator(device="cuda").manual_seed(SEED),
    ).images[0]
    
    # Save image but return the PIL Image object
    save_image(image)
    return image

if __name__ == "__main__":
    # Example usage
    generate_fashion_outfit("Two person hugging with text 'ST LOUIS CITY SC'")
