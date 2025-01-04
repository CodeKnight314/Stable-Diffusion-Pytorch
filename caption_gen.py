import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from glob import glob
import os 
from PIL import Image
from tqdm import tqdm
import csv
import argparse

class ImageDataset(Dataset):
    def __init__(self, root_dir: str, clip_preprocesser: CLIPProcessor):
        self.root_dir = glob(os.path.join(root_dir, "*.png")) + glob(os.path.join(root_dir, "*.jpg"))
        self.clip_processor = clip_preprocesser
        
    def __len__(self):
        return len(self.root_dir)
    
    def __getitem__(self, index):
        img_path = self.root_dir[index]
        img = Image.open(img_path).convert("RGB")
        processed = self.clip_processor(img, return_tensors="pt")
        
        return {
            'image': processed['pixel_values'].squeeze(),
            'path': self.root_dir[index]
        }
        
class CaptioningPipeline: 
    def __init__(self, style: str = "anime", batch_size: int = 4, device: str = "cuda" if torch.cuda.is_available() else "cpu"): 
        self.style = style
        self.batch_size = batch_size
        self.device = device
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
        self.llm = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
        
        self.style_config = {
            "anime": {
                "prefix": "This anime-style illustration shows",
                "elements": ["anime character", "manga style", "cel shading", 
                           "2D animation", "vibrant colors", "expressive eyes"],
                "prompt_template": "Describe this anime illustration in detail:"
            },
            "cyberpunk": {
                "prefix": "This cyberpunk scene depicts",
                "elements": ["neon lights", "dystopian city", "cybernetic augmentations",
                           "high tech", "urban environment", "futuristic elements"],
                "prompt_template": "Describe this cyberpunk scene in detail:"
            }
        }
        
    def detect_style(self, img_features: torch.Tensor):
        """
        Detect the style of the image based on the CLIP features
        
        Args:
            img_features (torch.Tensor): The CLIP features of the image
        
        Returns:
            dict: A dictionary containing the style elements and their similarity scores
        """
        style = self.style_config[self.style]["elements"]
        
        text_input = self.tokenizer(
            text=style, 
            padding=True,
            return_tensors="pt"
        )
        
        with torch.no_grad(): 
            text_features = self.clip.get_text_features(**text_input).to(self.device)
            similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
        
        return {element: score.item() for element, score in zip(style, similarity[0])}
    
    def generate_captions(self, styleDict: dict[str, float]): 
        top_k = sorted(styleDict.items(), lambda x: x[1], reverse=True)[0]
        style_str = ",".join(f"{elem}" for elem, _ in top_k)
        
        prompt = self.style_config[self.style]["prefix"] + " " + style_str
        prompt += self.style_config[self.style]["prompt_template"]
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad(): 
            outputs = self.llm.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
    
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def process_batch(self, images: torch.Tensor): 
        with torch.no_grad(): 
            img_features = self.clip.get_image_features(images.to(self.device))
            
            captions = []
            for img in img_features:
                styleDict = self.detect_style(img.unsqueeze(0))
                
                caption = self.generate_captions(styleDict)
                captions.append(caption)
            
            return captions            
    
    def process_directory(self, root_dir: str, output_path: str): 
        dataset = ImageDataset(root_dir, self.clip_processor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        results = {}
        for batch in tqdm(loader):
            img = batch["image"]
            path = batch["path"]
            
            captions = self.process_batch(img)
            
            for img_path, caption in zip(path, captions):
                results[img_path] = caption
        
        os.makedirs(output_path, exist_ok=True)
        file_name = os.path.join(output_path, f"{self.style}_captions.csv")
        with open(file_name, 'w') as f: 
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["image_path", "caption"])
            for path, capiton in results.items(): 
                writer.writerow([os.path.basename(path), caption])
        
        print(f"[INFO] Captions saved to {output_path}")
        
        return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Root directory containing all images")
    parser.add_argument("--style", type=str, default="cyberpunk", choices=["anime", "cyberpunk"], help="Desired style for caption system")
    parser.add_argument("--output", type=str, help="Output directory for caption csv")
    parser.add_argument("--batch", type=int, default=4, help="Number of images processed per batch")
    
    args = parser.parse_args() 
    
    pipeline = CaptioningPipeline(style=args.style, batch_size=args.batch)
    
    results = pipeline.process_directory(args.root, args.output)
            
        
            
        
        