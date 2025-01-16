# Stable-Diffusion-Pytorch

## Overview
This is an educational repository on fine-tuning the publicly available Stable-Diffusion model at HuggingFace. The repository includes a custom Stable-Diffusion Model class configurable for both conditional and unconditional image training. Compared to other impressive PyTorch implementations of Stable-Diffusion, of which there are many, this repository aims to provide an easy framework to work with the Stable-Diffusion model on fine-tuning tasks rather than implementing from scratch, out of consideration for computational resources and redundancy.

The current repository is available to fine-tune on conditional or unconditional fine-tuning. For future developments, this repository will aim to expand the fine-tuning framework to Image-to-Image translation, Image inpainting, and other areas within reason. Results are, or will be, shown below along with instructions on usage and deployment.

## Visual Results
<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; max-width: 90%; margin: 0 auto;">
    <div style="display: flex; justify-content: center; flex-basis: 100%; flex-wrap: wrap;">
        <img src="resources/result_0.png" alt="Stable-Diffusion Image 0" style="width: 30%; margin: 10px;">
        <img src="resources/result_1.png" alt="Stable-Diffusion Image 1" style="width: 30%; margin: 10px;">
        <img src="resources/result_2.png" alt="Stable-Diffusion Image 2" style="width: 30%; margin: 10px;">
    </div>
    <div style="display: flex; justify-content: center; flex-basis: 100%; flex-wrap: wrap;">
        <img src="resources/result_3.png" alt="Stable-Diffusion Image 3" style="width: 30%; margin: 10px;">
        <img src="resources/result_4.png" alt="Stable-Diffusion Image 4" style="width: 30%; margin: 10px;">
        <img src="resources/result_5.png" alt="Stable-Diffusion Image 5" style="width: 30%; margin: 10px;">
    </div>
</div>

## Dependencies
This repository requires the following:
- Python >= 3.8
- PyTorch >= 1.10
- HuggingFace Transformers
- Diffusers
- CUDA-capable GPU and drivers
- Other dependencies listed in `requirements.txt`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Planned Features
- **Image-to-Image Translation**
  - Enable the model to generate variations of a given input image by fine-tuning it for image-to-image translation tasks.
- **Image Inpainting**
  - Extend the framework to support inpainting tasks, allowing users to repair or complete partially damaged or missing sections of images.
- **Textual Inversion Integration**
  - Integrate support for textual inversion, enabling users to fine-tune the model to learn and generate specific styles or concepts based on new text prompts.
- **Multi-GPU Training**
  - Add support for distributed data parallel (DDP) to leverage multiple GPUs efficiently.
- **Better Dataset for Cyberpunk Style Finetuning**
  - Add supporting utils and scraping functions to create quality datasets from unlabeled data.

## Usage
### Training
To fine-tune the model using the `train.py` script, the following command-line arguments are required:
- `--root` (type: str, required): Specifies the root directory containing the dataset for fine-tuning.
- `--csv` (type: str, required): Path to the CSV file containing image captions paired with their respective images.
- `--epoch` (type: int, optional, default: 10): Number of epochs to run for fine-tuning.
- `--save` (type: str, required): Path where the model checkpoints will be saved.
- `--config` (type: str, required): Path to the configuration file specifying model and training parameters.

Example Usage:
```bash
python train.py \
    --root /path/to/dataset \
    --csv /path/to/captions.csv \
    --epoch 20 \
    --save /path/to/save/checkpoints \
    --config /path/to/config.json
```

### Inference (Direct Call)
The `inference.py` script allows generating images using a trained Stable Diffusion model. Below are the required and optional command-line arguments:
- `--weight` (type: str, optional): Path to the .pth file containing the model weights.
- `--config` (type: str, required): Path to the configuration file (config.yaml) specifying model and scheduler parameters.
- `--prompt` (type: str, optional, default: None): Text prompt to guide the image generation process.
- `--num_images` (type: int, optional, default: 1): Number of images to generate.
- `--output` (type: str, optional, default: "output/"): Directory to save the generated images.

Example Usage:
```bash
python inference.py \
    --weight /path/to/SD_full.pth \
    --config /path/to/config.yaml \
    --prompt "A futuristic cyberpunk cityscape" \
    --num_images 3 \
    --output /path/to/output/directory
```

## Inference (Terminal GUI)
The `cli_gui.py` is meant to provide a command-line interface for generating pictures with less coding required.
- `--weights` (type: str, optional): Path to the .pth file containing the model weights.
- `--config` (type: str, required): Path to Path to configuration file (default: config.yaml).

Example Usage:
```bash
python cli_gen.py \
    --weight /path/to/SD_full.pth \
    --config /path/to/config.yaml \
```
All remaining action, including generating pictures and changing prompt, can be done via the Terminal GUI displayed for convenience. 
