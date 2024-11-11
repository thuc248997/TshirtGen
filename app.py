import gradio as gr
from infer_flux import generate_fashion_outfit
import os

# Define available LoRA models with their trigger words
LORA_MODELS = {
    "Latent Pop Style": {
        "path": "jakedahn/flux-latentpop",
        "trigger": "LNTP"
    },
    "FLUX.1-dev-LoRA-Text-Poster": {
        "path": "Shakker-Labs/FLUX.1-dev-LoRA-Text-Poster",
        "trigger": "Text poster"
    },
    "FLUX.1-dev-LoRA-Logo-Design": {
        "path": "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
        "trigger": "wablogo, logo, Minimalist"
    }
}

def generate_outfit_gradio(content_print: str, lora_model: str):
    """
    Wrapper function for Gradio interface
    """
    # Get the model info from the selected model name
    model_info = LORA_MODELS[lora_model]
    lora_path = model_info["path"]
    
    # Add trigger words to the content
    enhanced_prompt = f"{content_print}, {model_info['trigger']}"
    print(f"Enhanced prompt: {enhanced_prompt}")  # Debug print
    
    image = generate_fashion_outfit(enhanced_prompt, lora_path)
    return image

# Create Gradio interface
def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# T-shirt Generator Demo")
        
        with gr.Row():
            with gr.Column():
                # Input components
                content_input = gr.Textbox(
                    label="Content Print on Garment",
                    placeholder="Describe what to print on the t-shirt",
                )
                
                # Add LoRA model selection dropdown
                lora_dropdown = gr.Dropdown(
                    choices=list(LORA_MODELS.keys()),
                    value=list(LORA_MODELS.keys())[0],
                    label="Select LoRA Model",
                    info="Choose the style model to use"
                )
                
                # Add info text to show current trigger words
                trigger_info = gr.Markdown("")
                
                def update_trigger_info(lora_model):
                    if lora_model:
                        trigger = LORA_MODELS[lora_model]["trigger"]
                        return f"*Current model trigger words: {trigger}*"
                    return ""
                
                # Update trigger info when model selection changes
                lora_dropdown.change(
                    fn=update_trigger_info,
                    inputs=[lora_dropdown],
                    outputs=[trigger_info]
                )
                
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                # Output component
                output_image = gr.Image(
                    label="Generated T-shirt",
                    type="pil",
                    show_download_button=True,
                    format="png"
                )
        
        # Connect the button click to the generate function
        generate_btn.click(
            fn=generate_outfit_gradio,
            inputs=[content_input, lora_dropdown],
            outputs=output_image
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)