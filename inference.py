import torch
from PIL import Image
import argparse
from diffusion import load_diffusion, StableDiffusion
from tqdm import tqdm
from utils import save_images, load_module, load_config

def inference(model: StableDiffusion, prompt: str, scheduler: torch.nn.Module, guidance_scale: float = 7.5):
    """
    Function to generate images using the Stable Diffusion Model with the given prompt and scheduler. 
    Optionally, guidance scale can be set to control the influence of the prompt on the generated image.
    
    Args:
        model (StableDiffusion): Stable Diffusion Model
        prompt (str): Text Prompt for the model
        scheduler (DDIMScheduler): DDIM Scheduler for the diffusion process
        guidance_scale (float): Guidance Scale for the prompt
    
    Returns:
        Image: Generated Image from the Stable Diffusion Model
    """
    if prompt is not None:
        uncond_embeddings = model.prepare_text_embeddings("")
        text_embeddings = model.prepare_text_embeddings(prompt)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    else:
        text_embeddings = None
    
    latents = torch.randn(
        (1, model.unet.config.in_channels, 
         model.unet.sample_size, model.unet.sample_size),
        device=model.device,
    )
    
    latents = latents * scheduler.init_noise_sigma
    
    for step in tqdm(scheduler.timesteps, desc="[INFO] Generating Image"):
        if prompt is not None:
            latent_model_input = torch.cat([latents] * 2)
            time_tensor = torch.tensor([step] * latent_model_input.shape[0], device=model.device)
        else:
            latent_model_input = latents
            time_tensor = torch.tensor([step] * latents.shape[0], device=model.device)

        with torch.no_grad():
            noise_pred = model.predict_noise(latent_model_input, time_tensor, text_embeddings)

        if prompt is not None:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(
            model_output=noise_pred,
            timestep=step,
            sample=latents,
        ).prev_sample

    latents = latents / 0.18215
    with torch.no_grad():
        image = model.vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    image = (image * 255).round().astype("uint8")
    
    return Image.fromarray(image)
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=False, help="Directory containing SD_full.pth")
    parser.add_argument("--config", type=str, required=True, help="Directory containing config.yaml")
    parser.add_argument("--prompt", type=str, default=None, help="Conditional Prompt for the Stable Diffusion Model")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="output/", help="Directory to save generated images")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    model = load_diffusion(config["model_params"])
    
    scheduler = load_module(config["noise_scheduler"])
    scheduler.set_timesteps(config["noise_scheduler"]["num_steps"])
    if args.weight is not None:
        model.load_model(args.weight)
    
    images = []
    for i in range(args.num_images):
        sample = inference(model, prompt=args.prompt, scheduler=scheduler)
        images.append(sample)
    
    save_images(images, args.output)