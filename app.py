import gradio as gr
from infer_flux import generate_fashion_outfit
def generate_outfit_gradio(color: str, content_print: str):
    """
    Wrapper function for Gradio interface
    """
    image_path = generate_fashion_outfit(color, content_print)
    return image_path

# Create Gradio interface
def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# T-shirt Generator Demo")
        
        with gr.Row():
            with gr.Column():
                # Input components
                color_input = gr.Textbox(
                    label="Color",
                    placeholder="Enter t-shirt color (e.g., Black, White, Red)",
                )
                content_input = gr.Textbox(
                    label="Content Print on Garment",
                    placeholder="Describe what to print on the t-shirt",
                )
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                # Output component
                output_image = gr.Image(
                    label="Generated T-shirt",
                    type="filepath"
                )
        
        # Connect the button click to the generate function
        generate_btn.click(
            fn=generate_outfit_gradio,
            inputs=[color_input, content_input],
            outputs=output_image
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)