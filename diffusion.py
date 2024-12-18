import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

class StableDiffusion: 
    def __init__(self, model_id: str, use_conditioning: bool = True, device: str = "cuda", sample_size: int = 64, train_text_encoder=False): 
        self.device = device 
        self.use_conditioning = use_conditioning
        
        self.vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae"
        ).to(device)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet", 
            sample_size=sample_size
        ).to(device)
        
        if use_conditioning: 
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_id, 
                subfolder="text_encoder"
            ).to(device)
            
            self.tokenzier = CLIPTokenizer.from_pretrained(
                model_id, 
                subfolder="tokenizer"
            )
        else: 
            for name, module in self.unet.named_modules():
                if "attn2" in name:
                    module.register_forward_hook(lambda m, i, o: (None,))
    
    def freeze_parameter(self, train_text_encoder: bool): 
        for params in self.vae.parameters(): 
            params.requires_grad = False 
            
        if self.use_conditioning and not train_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False 
                
    def prepare_text_embeddings(self, prompt):
        if not self.use_conditioning:
            return None
            
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
        if self.use_conditioning:
            return self.unet(latents, timesteps, encoder_hidden_states=text_embeddings).sample
        else:
            return self.unet(latents, timesteps).sample

    def get_trainable_params(self):
        params = []
        params.extend(self.unet.parameters())
        if self.use_conditioning and self.text_encoder.parameters():
            params.extend(self.text_encoder.parameters())
        return params

    def save_pretrained(self, save_path):
        self.unet.save_pretrained(f"{save_path}/unet")
        self.vae.save_pretrained(f"{save_path}/vae")
        if self.use_conditioning:
            self.text_encoder.save_pretrained(f"{save_path}/text_encoder")
            self.tokenizer.save_pretrained(f"{save_path}/tokenizer")
            
if __name__ == "__main__": 
    model = StableDiffusion(
      model_id="CompVis/stable-diffusion-v1-4",
      use_conditioning=True,
      train_text_encoder=True,
      sample_size=64
    )