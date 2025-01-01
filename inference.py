import torch
from diffusers import DDPMScheduler, DDIMScheduler
from PIL import Image
import argparse
from diffusion import StableDiffusion
from tqdm import tqdm
from utils import save_images

def inference(model: StableDiffusion, prompt: str, scheduler: DDIMScheduler, guidance_scale: float = 7.5):
    if model.use_conditioning and prompt is not None:
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
    
    for step in tqdm(scheduler.timesteps):
        if model.use_conditioning and prompt is not None:
            latent_model_input = torch.cat([latents] * 2)
            time_tensor = torch.tensor([step] * latent_model_input.shape[0], device=model.device)
        else:
            latent_model_input = latents
            time_tensor = torch.tensor([step] * latents.shape[0], device=model.device)

        with torch.no_grad():
            noise_pred = model.predict_noise(latent_model_input, time_tensor, text_embeddings)

        if model.use_conditioning and prompt is not None:
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
    parser.add_argument("--weight", type=str, required=False, help="Path to Stable Diffusion Weights")
    parser.add_argument("--condition", action="store_true", help="Toggle option for turning on/off text prompting")
    parser.add_argument("--size", type=int, default=512, help="Image size for Stable Diffusion Output")
    parser.add_argument("--prompt", type=str, default=None, help="Conditional Prompt for the Stable Diffusion Model")
    parser.add_argument("--steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="output", help="Directory to save generated images")
    
    args = parser.parse_args()
    
    model = StableDiffusion(
        model_id="CompVis/stable-diffusion-v1-4",
        use_conditioning=args.condition,
        train_text_encoder=False,
        sample_size=args.size // 8,
    )
    
    if args.weight is not None:
        model.load_from_pretrained(args.weight)
    
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    scheduler.set_timesteps(args.steps)
    
    images = []
    for i in range(args.num_images):
        sample = inference(model, prompt=args.prompt, num_train_steps=args.steps, scheduler=scheduler)
        images.append(sample)
    
    save_images(images, args.output)