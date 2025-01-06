import torch
import argparse
import os
from dataset import DataLoader, load_dataset
from diffusion import load_diffusion, StableDiffusion
from tqdm import tqdm
from utils import EarlyStopping, save_images, load_config, load_module, load_optimizer
from inference import inference

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model: StableDiffusion, train_dl: DataLoader, config: dict, save_path: str, epochs: int):
    """
    Train the StableDiffusion model with the given DataLoader and configuration.
    
    Args:
        model (StableDiffusion): The StableDiffusion model to be trained.
        train_dl (DataLoader): The DataLoader containing the training data.
        config (dict): The configuration dictionary.
        save_path (str): The path to save the model checkpoints.
    """
    optimizer = load_optimizer(config["optimizer"], model.get_trainable_params())
    noise_scheduler = load_module(config["noise_scheduler"])
    scaler = torch.cuda.amp.GradScaler() if config["mixed_precision"] else None
    early_stopping = EarlyStopping(patience=5)

    os.makedirs(save_path, exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_dl)
    )

    try:
        global_step = 0
        for epoch in range(epochs):
            model.unet.train()
            if config["model_params"]["train_text_encoder"]:
                model.text_encoder.train()

            epoch_loss = 0.0
            for step, data in tqdm(enumerate(train_dl), total=len(train_dl),
                                   desc=f"[Epoch {epoch}] Fine-tuning Conditional StableDiffusion"):
                img, prompt = data
                text_embeddings = model.prepare_text_embeddings(prompt).to(device)

                img = img.to(device)
                latents = model.encode_images(img)

                noise = torch.randn_like(latents).to(device)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if config["mixed_precision"]:
                    with torch.cuda.amp.autocast():
                        noisy_pred = model.predict_noise(noisy_latents, timesteps, text_embeddings)
                        loss = torch.nn.functional.mse_loss(noisy_pred, noise) / config["g_step"]

                    scaler.scale(loss).backward()
                    if (step + 1) % config["g_step"] == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                else:
                    noisy_pred = model.predict_noise(noisy_latents, timesteps, text_embeddings)
                    loss = torch.nn.functional.mse_loss(noisy_pred, noise) / config["g_step"]

                    loss.backward()
                    if (step + 1) % config["g_step"] == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % config["s_step"] == 0:
                    checkpoint_path = os.path.join(save_path, f"checkpoint-{global_step}")
                    model.save_pretrained(checkpoint_path)

                    noise_scheduler.set_timesteps(100)
                    data = next(iter(train_dl))
                    pil_images = []
    
                    img, text = data
                    for prompt in text:
                        pil_images.append(
                            inference(
                                model,
                                prompt=prompt,
                                scheduler=noise_scheduler,
                                guidance_scale=7.5,
                            )
                        )

                    save_images(pil_images, os.path.join(save_path, f"checkpoint-{global_step}-samples/"))

                    early_stopping(epoch_loss)
                    if early_stopping.early_stop:
                        print("[INFO] Early Stopping Mechanism triggered.")
                        return

            print(f"Epoch {epoch + 1} average loss: {epoch_loss / len(train_dl):.4f}")

    except KeyboardInterrupt:
        checkpoint_path = os.path.join(save_path, "Interrupted_checkpoint")
        model.save_pretrained(checkpoint_path)
        print("[INFO] Saved model after interrupted fine-tuning")
    finally:
        torch.cuda.empty_cache()
        print("[INFO] Cleared CUDA memory cache")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Root directory to dataset")
    parser.add_argument("--csv", type=str, help="Path to CSV file containing image captions")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs for fine-tuning")
    parser.add_argument("--save", type=str, help="Save Path for checkpoints")
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()
    config = load_config(args.config)

    model = load_diffusion(config["model_params"])
    train_dl = load_dataset(args.root, args.csv, config["image_size"], config["batch"])

    train(model, train_dl, config, args.save, args.epoch)