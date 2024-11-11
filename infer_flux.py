import torch
from diffusers import FluxPipeline
from PIL import Image
import os
from datetime import datetime
import peft

# Constants
MODEL_NAME = "tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator"
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
GUIDANCE_SCALE = 3.5
NUM_INFERENCE_STEPS = 30
MAX_SEQUENCE_LENGTH = 512
SEED = 42

# Initialize the FLUX pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", # Base model
    torch_dtype=torch.bfloat16
)
# Load LoRA weights
pipe.load_lora_weights("tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator")
pipe.enable_model_cpu_offload()

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

def generate_fashion_outfit(color: str, content_print: str) -> str:
    """
    Generate a image sample T-shirt with specified color and print content.

    Args:
        color (str): Color of the T-shirt (e.g., "Black", "White", "Red")
        content_print (str): Content to print on the T-shirt (e.g., "a tiger in the center")

    Returns:
        str: Path to the saved image
    """
    prompt = f"A T-shirt with Color: {color}, Department: T-shirt, Detail: Basic, " \
             f"Fabric-Elasticity: Medium Stretch, Fit: Regular, Hemline: Straight, " \
             f"Material: Cotton, Neckline: Crew Neck, Pattern: Solid with Print, " \
             f"Sleeve-Length: Short Sleeve, Style: Casual, Type: Regular, " \
             f"Waistline: Natural, Content Print on Garment: {content_print}."

    image = pipe(
        prompt,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        generator=torch.Generator(device="cuda").manual_seed(SEED),
    ).images[0]

    return save_image(image)

if __name__ == "__main__":
    # Example usage
    generate_fashion_outfit("Black", "a tiger in the center")