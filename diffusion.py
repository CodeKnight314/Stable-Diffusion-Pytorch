import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

class LoRa(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 4):
        super().__init__()
        self.base_layer = base_layer 
        self.rank = rank 

        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)
        
    def forward(self, x):
        return self.base_layer(x) + self.lora_up(self.lora_down(x)) * (1.0/self.rank)
    
def inject_lora(model: nn.Module, rank: int = 4, device: str = "cuda"):
    require_grad_params = []
    lora_layers = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(x in name for x in ['attn']):
            lora_layer = LoRa(module, rank).to(device)
            
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
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
    def __init__(self, model_id: str, device: str = "cuda", sample_size: int = 64, train_text_encoder=False):     
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
        
        self.lora_layers = inject_lora(self.unet, 4, device)

        self.embedding_dim = self.unet.config.cross_attention_dim
        self.device = device 
        self.sample_size = sample_size
    
    def freeze_parameter(self, train_text_encoder : bool, lora: bool = True):
        """
        Freeze the parameters of the model
        
        Args:
            train_text_encoder (bool): Flag to freeze or unfreeze the text encoder
        """ 
        
        if lora:
            for layer in self.lora_layers.values():
                layer.lora_down.weight.requires_grad = True
                layer.lora_up.weight.requires_grad = True
        else:
            for params in self.unet.parameters():
                params.requires_grad = False
            
        for params in self.vae.parameters(): 
            params.requires_grad = False 
            
        if not train_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False 
                
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

    def save_pretrained(self, save_path):
        self.unet.save_pretrained(f"{save_path}/unet")
        self.vae.save_pretrained(f"{save_path}/vae")
        if self.tokenizer and self.text_encoder:
            self.text_encoder.save_pretrained(f"{save_path}/text_encoder")
            self.tokenizer.save_pretrained(f"{save_path}/tokenizer")
    
    def load_from_pretrained(self, save_path: str, device: str = "cuda", sample_size: int = 64):
        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                f"{save_path}/unet",
                sample_size=sample_size
            ).to(device)
        except Exception as e: 
            print("[ERROR] Loading unet weights encountered error. See below for details: ")
            print(e)
        
        try:
            self.vae = AutoencoderKL.from_pretrained(
                f"{save_path}/vae"
            ).to(device)
        except Exception as e: 
            print("[ERROR] Loading vae weights encountered error. See below for details: ")
            print(e)
        
        try:
            self.text_encoder = CLIPTextModel.from_pretrained(
                f"{save_path}/text_encoder"
            ).to(device)
        except Exception as e: 
            print("[ERROR] Loading text encoder weights encountered error. See below for details: ")
            print(e)
        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                f"{save_path}/tokenizer"
            )
        except Exception as e: 
            print("[ERROR] Loading tokenizer weights encountered error. See below for details: ")
            print(e)
        
        print(f"[INFO] Loaded weights from {save_path} successfully.")
      
def load_diffusion(config):
    model = StableDiffusion(
        model_id=config["model_id"],
        sample_size=config["sample_size"],
        train_text_encoder=config["train_text_encoder"]
    )
    
    model.freeze_parameter(config["train_text_encoder"], config["lora_finetuning"])
    
    print(f"----------------------------------------")
    print(f"Loading StableDiffusion model with the following parameters:")
    print(f"Model ID:           {config['model_id']}")
    print(f"Sample Size:        {config['sample_size']}")
    print(f"Train Text Encoder: {config['train_text_encoder']}")
    print(f"Lora Finetuning:    {config['lora_finetuning']}")
    print(f"----------------------------------------")
    return model

if __name__ == "__main__": 
    model = StableDiffusion(
      model_id="CompVis/stable-diffusion-v1-4",
      train_text_encoder=True,
      sample_size=64
    )