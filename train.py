import torch 
from dataset import load_FFHQ_dataset, load_Flickr30k, DataLoader
import argparse 
import os
from diffusion import StableDiffusion
from torch.optim import AdamW 
from diffusers import DDPMScheduler
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(
    model: StableDiffusion,
    train_dl: DataLoader,
    save_path: str,
    epochs: int,
    lr: float,
    mixed_precision: bool,
    gradient_accum_step: int,
    save_step: int
):
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
    )
    
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=lr
    )
    
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    os.makedirs(save_path, exist_ok=True)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs * len(train_dl)
    )
    
    global_step = 0
    for epoch in range(epochs):
        model.unet.train()
        if model.use_conditioning and model.text_encoder.parameters():
            model.text_encoder.train()
            
        epoch_loss = 0.0
        for step, data in tqdm(enumerate(train_dl), total=len(train_dl), 
                             desc=f"[Epoch {epoch}] Fine-tuning {'Conditional' if model.use_conditioning else 'Unconditional'} StableDiffusion"):
            if model.use_conditioning:
                img, prompt = data
                text_embeddings = model.prepare_text_embeddings(prompt).to(device)
            else:
                img = data
                text_embeddings = None
            
            img = img.to(device)
            latents = model.encode_images(img)
            
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                device=latents.device
            )
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    noisy_pred = model.predict_noise(noisy_latents, timesteps, text_embeddings)
                    loss = torch.nn.functional.l1_loss(noisy_pred, noise)
                    loss = loss / gradient_accum_step
                    
                scaler.scale(loss).backward()
                if (step + 1) % gradient_accum_step == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                noisy_pred = model.predict_noise(noisy_latents, timesteps, text_embeddings)
                loss = torch.nn.functional.l1_loss(noisy_pred, noise)
                loss = loss / gradient_accum_step
                
                loss.backward()
                if (step + 1) % gradient_accum_step == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
            epoch_loss += loss.item()            
            global_step += 1
            
            if global_step % save_step == 0:
                checkpoint_path = os.path.join(save_path, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_path)
        
        print(f"Epoch {epoch} average loss: {epoch_loss / len(train_dl):.4f}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Root directory to dataset")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for fine tuning")
    parser.add_argument("--condition", action="store_true", help="Toggle option for conditioning Diffusion Model")
    parser.add_argument("--mp", action="store_true", help="Toggle option for mixed precision")
    parser.add_argument("--g_step", type=int, default=5, help="Configurable gradient accumulation step")
    parser.add_argument("--s_step", type=int, default=100, help="Configurable save step for checkpointing")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--save", type=str, help="Save Path for checkpoints")
    parser.add_argument("--id", type=str, default="CompVis/stable-diffusion-v1-4", help="Optional ID for diffuser to load")
    parser.add_argument("--size", type=int, default=512, help="Patch size for Diffusion Model and dataset")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for dataset")
    
    args = parser.parse_args() 
    
    if args.condition: 
        train_dl = load_Flickr30k(root=args.root, csv_path=os.path.join(args.root, "results.csv"), size=(args.size, args.size), batch_size=args.batch)
    else: 
        train_dl = load_FFHQ_dataset(root=args.root, size=(args.size, args.size), batch_size=args.batch)
        
    model = StableDiffusion(
      model_id=args.id,
      use_conditioning=args.condition, device = device,
      train_text_encoder=False,
      sample_size=args.size
    )
    
    train(model, train_dl, args.save, args.epoch, args.lr, args.mp, args.g_step, args.s_step)
    
    