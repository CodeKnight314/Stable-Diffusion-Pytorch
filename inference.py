import torch
from diffusers import DDPMScheduler
from PIL import Image
import argparse
from diffusion import StableDiffusion
from tqdm import tqdm
import os

def inference(
    model : StableDiffusion,
    prompt: str = None,
    num_train_steps: int = 1000,
    num_images: int = 1,
):
    """
    Perform inference (image generation) using a StableDiffusion model and the DDPMScheduler.
    
    Args:
        model (StableDiffusion): A StableDiffusion instance (defined above).
        prompt (str, optional): Text prompt for conditional generation.
        num_train_steps (int, optional): Number of diffusion steps (DDPM steps). 
                                         Higher values generally mean better fidelity but slower inference.
        num_images (int, optional): Number of images to generate.

    Returns:
        List[PIL.Image.Image]: A list of generated images as PIL images.
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_steps,
        beta_start=0.00085,
        beta_end=0.012,
    )
    
    if model.use_conditioning and prompt is not None:
        text_embedding = model.prepare_text_embeddings(prompt)
    else:
        text_embedding = None
    
    batch_size = num_images
    latents = torch.randn(
        (
            batch_size,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size,
        ),
        device=model.device,
    )
    latents = latents * scheduler.init_noise_sigma
    
    for step in scheduler.timesteps:
        time_tensor = torch.tensor([step] * batch_size, device=model.device)

        with torch.no_grad():
            noise_pred = model.predict_noise(latents, time_tensor, text_embedding)

        latents = scheduler.step(
            model_output=noise_pred,
            timestep=step,
            sample=latents,
        ).prev_sample

    latents = latents / 0.18215

    with torch.no_grad():
        images = model.vae.decode(latents).sample
    
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def save_images(images, save_pth: str):
    os.makedirs(save_pth, exist_ok=True)
    for i, img in enumerate(tqdm(images, total=len(images), desc=f"Saving images to {save_pth}")):
        img.save(os.path.join(save_pth, f"result_{i}.png"))
    print("Finished saving images")
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=False, help="Path to Stable Diffusion Weights")
    parser.add_argument("--condition", action="store_true", help="Toggle option for turning on/off text prompting")
    parser.add_argument("--size", type=int, default=512, help="Patch size for Stable Diffusion Output")
    parser.add_argument("--prompt", type=str, help="Conditional Prompt for the Stable Diffusion Model")
    parser.add_argument("--steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="output", help="Directory to save generated images")
    
    args = parser.parse_args()
    
    model = StableDiffusion(
        model_id="CompVis/stable-diffusion-v1-4",
        use_conditioning=args.condition,
        train_text_encoder=False,
        sample_size=args.size,
    )
    
    if args.weight is not None:
        model.load_from_pretrained(args.weight)
    
    images = inference(
        model,
        prompt=args.prompt,
        num_train_steps=args.steps,
        num_images=args.num_images,
    )
    
    save_images(images, args.output)