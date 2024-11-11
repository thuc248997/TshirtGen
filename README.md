# AI T-shirt Design Generator

An AI-powered T-shirt design generator using the FLUX model with various LoRA adaptations to create custom designs based on text descriptions.

## Features

- Generate custom T-shirt designs using different AI style models.
- Support for multiple LoRA models:
  - **Latent Pop Style**
  - **Text Poster Style**
  - **Logo Design Style**
- User-friendly web interface powered by Gradio.
- High-resolution output (1024x1024).
- PNG download support.
- Real-time style preview.

## Installation

### Prerequisites

- Python 3.11 or higher.
- CUDA-compatible GPU (recommended).
- 8GB+ GPU VRAM.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/t-shirt-generator.git
   cd t-shirt-generator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   conda create -n TshirtGen python=3.11
   conda activate TshirtGen
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

1. **Start the Gradio web interface:**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:7860`.

3. **Using the interface:**
   - Enter your design description in the "Content Print on Garment" field.
   - Select a LoRA model style from the dropdown menu.
   - Click "Generate" to create your design.
   - Download the generated image using the download button.

### Available LoRA Models

1. **Latent Pop Style**
   - **Trigger:** `LNTP`
   - **Best for:** Artistic and pop-art style designs.

2. **Text Poster Style**
   - **Trigger:** `Text poster`
   - **Best for:** Typography and text-based designs.

3. **Logo Design Style**
   - **Trigger:** `wablogo`, `logo`, `Minimalist`
   - **Best for:** Minimalist logo designs.

## Project Structure

```plaintext
├── app.py                # Gradio web interface
├── infer_flux.py         # FLUX model inference code
├── requirements.txt      # Python dependencies
├── outputs/              # Generated images directory
└── README.md             # Project documentation
```

## Configuration

Key parameters can be modified in `infer_flux.py`:

- **IMAGE_HEIGHT**: Output image height (default: 1024).
- **IMAGE_WIDTH**: Output image width (default: 1024).
- **GUIDANCE_SCALE**: Generation guidance scale (default: 0.8).
- **NUM_INFERENCE_STEPS**: Number of inference steps (default: 30).

## Contributing

Contributions are welcome! Feel free to submit a pull request.

## License

[Specify your license here]

## Acknowledgments

- [Black Forest Labs](https://github.com/black-forest-labs) for the FLUX model.
- [Jake Dahn](https://huggingface.co/jakedahn) for the Latent Pop LoRA.
- [Shakker Labs](https://huggingface.co/Shakker-Labs) for Text Poster and Logo Design LoRAs.