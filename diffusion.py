import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from utils import load_config
from torchsummary import summary
import math
import os

class LoRa(nn.Module):
    def __init__(self, base_layer: nn.Module, rank: int = 4):
        super().__init__()
        self.base_layer = base_layer 
        self.rank = rank 
        
        if isinstance(base_layer, nn.Linear):
            self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)
        elif isinstance(base_layer, nn.Conv2d):
            self.lora_down = nn.Conv2d(base_layer.in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False)
            self.lora_up = nn.Conv2d(rank, base_layer.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            raise ValueError(f"[ERROR] LoRA Layer only supports nn.Linear and nn.Conv2d layers but received {type(base_layer)}.")
            
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, x):
        return self.base_layer(x) + self.lora_up(self.lora_down(x)) * (1.0/self.rank)
    
def inject_lora(model: nn.Module, rank: int = 4, device: str = "cuda"):
    require_grad_params = []
    lora_layers = {}
    
    for name, module in model.named_modules():
        if any(x in name for x in ['attn']):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                lora_layer = LoRa(module, rank).to(device)
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, lora_layer)
            
                module.requires_grad_(False)
                
                require_grad_params.extend([
                    lora_layer.lora_down.weight,
                    lora_layer.lora_up.weight,
                ])
            
                lora_layers[name] = lora_layer
    
    return lora_layers

class StableDiffusion: 
    def __init__(self, model_id: str, device: str = "cuda", sample_size: int = 64, rank: int = 4, lora_finetuning: bool = False):     
        self.vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae"
        ).to(device)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet", 
            sample_size=sample_size
        ).to(device)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, 
            subfolder="text_encoder"
        ).to(device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, 
            subfolder="tokenizer"
        )
        
        self.lora_layers = inject_lora(self.unet, rank, device)

        self.embedding_dim = self.unet.config.cross_attention_dim
        self.device = device 
        self.sample_size = sample_size
        self.lora_training = lora_finetuning
    
    def freeze_parameter(self, train_text_encoder : bool, lora: bool = True):
        """
        Freeze the parameters of the model
        
        Args:
            train_text_encoder (bool): Flag to freeze or unfreeze the text encoder
            lora (bool): Flag to freeze or unfreeze the LoRA layers
        """ 
        
        for params in self.unet.parameters():
            params.requires_grad = False
            
        for params in self.vae.parameters(): 
            params.requires_grad = False 
            
        if not train_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        if lora:
            for layer in self.lora_layers.values():
                layer.lora_down.weight.requires_grad = True
                layer.lora_up.weight.requires_grad = True
        else: 
            for params in self.unet.parameters():
                params.requires_grad = True
    
    def count_parameters(self):
        """
        Count total and trainable parameters in the model
        
        Returns:
            tuple: (total_params, trainable_params)
        """
        total_params = 0
        trainable_params = 0
        
        for model in [self.unet, self.vae, self.text_encoder]:
            for param in model.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    
        return total_params, trainable_params
                
    def prepare_text_embeddings(self, prompt : str):
        """
        Prepare text embeddings for the model
        
        Args:
            prompt (str): Text prompt to encode
        """ 
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]
        return text_embeddings
    
    def encode_images(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images.to(self.device)).latent_dist.sample()
            latents = latents * 0.18215
        return latents
    
    def predict_noise(self, latents, timesteps, text_embeddings=None):
        if text_embeddings is not None:
            return self.unet(latents, timesteps, encoder_hidden_states=text_embeddings).sample
        else:
            batch_size = latents.shape[0]
            dummy_embeddings = torch.zeros(
                (batch_size, 77, self.embedding_dim),
                device=self.device
            )
            return self.unet(latents, timesteps, encoder_hidden_states=dummy_embeddings).sample

    def get_trainable_params(self):
        params = []
        params.extend(self.unet.parameters())
        if self.text_encoder.parameters():
            params.extend(self.text_encoder.parameters())
        return params
    
    def get_trainable_lora_params(self):
        if not self.lora_layers:
            return []
        
        params = []
        for layer in self.lora_layers.values():
            params.extend([
                layer.lora_down.weight,
                layer.lora_up.weight
            ])
            
        if self.text_encoder.parameters(): 
            params.extend(self.text_encoder.parameters())
            
        return params
    
    def save_model(self, save_path: str):
        os.makedirs(save_path)
        if self.lora_training: 
            lora_state_dict = {}
            for name, layer in self.lora_layers.items():
                lora_state_dict[f"{name}.lora_down.weight"] = layer.lora_down.weight
                lora_state_dict[f"{name}.lora_up.weight"] = layer.lora_up.weight
            
            torch.save(lora_state_dict, os.path.join(save_path, "lora.pth"))
            print(f"Saved LoRA weights to {os.path.join(save_path, 'lora.pth')}")
            
            if any(param.requires_grad for param in self.text_encoder.parameters()):
                torch.save(
                    self.text_encoder.state_dict(),
                    os.path.join(save_path, "text_encoder.pth")
                )
                print(f"Saved text encoder weights to {os.path.join(save_path, 'text_encoder.pth')}")
        else: 
            full_state_dict = {
                "unet": self.unet.state_dict(),
                "text_encoder": self.text_encoder.state_dict(),
                "vae": self.vae.state_dict()
            }
            
            torch.save(full_state_dict, os.path.join(save_path, "SD_full.pth"))
            print(f"Saved full model weights to {os.path.join(save_path, 'SD_full.pth')}")
            
    def load_model(self, save_path: str):
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Weights file not found at {save_path}")
            
        if os.path.exists(os.path.join(save_path, "lora.pth")):
            lora_state_dict = torch.load(save_path, map_location=self.device)
            
            try: 
                for name, layer in self.lora_layers.items():
                    if f"{name}.lora_down.weight" in lora_state_dict:
                        layer.lora_down.weight.data = lora_state_dict[f"{name}.lora_down.weight"].to(self.device)
                        layer.lora_up.weight.data = lora_state_dict[f"{name}.lora_up.weight"].to(self.device)
                print("[INFO] Loaded LoRA weights")
            except Exception as e: 
                print(f"[ERROR] Error loading LoRA weights. Encountered {e}")
            
            try:
                if(os.path.join("text_encoder.pth")):
                    self.text_encoder.load_state_dict(torch.load(os.path.join(save_path, "text_encoder.pth"), map_location=self.device))
            except Exception as e: 
                print("[ERROR] Error loading text encoder weights. Encountered {e}")
            
        else:
            full_state_dict = torch.load(os.path.join(save_path, "SD_full.pth"), map_location=self.device)
            try:
                self.unet.load_state_dict(full_state_dict["unet"])
                print("[INFO] Loaded UNet weights successfully")
            except Exception as e:
                print(f"[ERROR] Error loading UNet weights. Encountered {e}")
            
            try:  
                self.vae.load_state_dict(full_state_dict["vae"])
                print("[INFO] Loaded VAE weights successfully")
            except Exception as e:
                print(f"[ERROR] Error loading VAE weights. Encountered {e}")
            
            if self.text_encoder and "text_encoder" in full_state_dict:
                try:
                    self.text_encoder.load_state_dict(full_state_dict["text_encoder"])
                    print("[INFO] Loaded text encoder weights successfully")
                except Exception as e:
                    print(f"[ERROR] Error loading text encoder weights. Encountered {e}")
      
def load_diffusion(config, display: bool = True):
    model = StableDiffusion(
        model_id=config["model_id"],
        sample_size=config["sample_size"],
        rank=config["rank"], 
        lora_finetuning=config["lora_finetuning"]
    )
    
    model.freeze_parameter(config["train_text_encoder"], config["lora_finetuning"])
    total_params, trainable_params = model.count_parameters()
    if display:
        print(f"----------------------------------------")
        print(f"Loading StableDiffusion model with the following parameters:")
        print(f"Model ID:           {config['model_id']}")
        print(f"Sample Size:        {config['sample_size']}")
        print(f"Train Text Encoder: {config['train_text_encoder']}")
        print(f"Lora Finetuning:    {config['lora_finetuning']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {(trainable_params/total_params)*100:.2f}%\n")
        print(f"----------------------------------------")
    return model

if __name__ == "__main__": 
    config = load_config("config.yaml")
    model = load_diffusion(config["model_params"])
    unet = model.unet
    summary(unet, input_size=(4, model.sample_size, model.sample_size))