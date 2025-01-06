import torch 
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from glob import glob
import os 
from PIL import Image
from tqdm import tqdm
import csv
import argparse
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

class ImageClass(Dataset):
    def __init__(self, img_dir: str, patch_size: Tuple[int, int] = (224, 224)):
        self.img_dir = img_dir
        self.img_paths = glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpg"))
        
        self.height, self.wdith = patch_size
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        width, height = img.size
        if width > self.wdith or height > self.height:
            ratio = min(self.wdith/width, self.height/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio) 
            img = img.resize((new_width, new_height))
        
        return img, img_path

def custom_collate_fn(batch):
    images, paths = zip(*batch)
    return list(images), list(paths)
    
class End2End_CaptioningPipeline:
    def __init__(self, model_id: str = "Salesforce/blip2-opt-2.7b", prompt: str = None):
        self.processor = Blip2Processor.from_pretrained(model_id=model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id=model_id, dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        if hasattr(self.model.config, "use_memory_efficient_attention"):
            self.model.config.use_memory_efficient_attention = True
        
        if prompt is not None:
            self.prompt = prompt
        else: 
            self.prompt = "Generate a detailed caption, focusing on the key objects and actions in the image."
        
    def caption_image(self, inputs: torch.Tensor):
        with torch.no_grad(): 
            outputs = self.model.generate(**inputs, 
                                          max_length=150, 
                                          num_beams=5, 
                                          do_sample=True, 
                                          temperature=0.7, 
                                          top_p=0.90,
                                          prompt=self.prompt)
        
        captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
        return captions
    
    def process_directory(self, root_dir: str, output_path: str, batch_size: int = 4):
        img_dir = ImageClass(root_dir, self.processor)  
        img_loader = DataLoader(img_dir, 
                                batch_size=batch_size, 
                                collate_fn=custom_collate_fn, 
                                shuffle=False, 
                                num_workers=os.cpu_count())           
        
        img2caption_pairs = {}
        for batch_images, batch_paths in tqdm(img_loader, total=len(img_loader), desc=f"[INFO] Captioning images in {root_dir}: "):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            inputs = self.processor(images=list(batch_images), return_tensors="pt", padding=True).to(self.device)

            caption = self.caption_image(inputs)

            for path, caption in zip(batch_paths, caption):
                img2caption_pairs[path] = caption
            
        os.makedirs(output_path, exist_ok=True)
        file_name = os.path.join(output_path, f"captions.csv")
        with open(file_name, 'w', newline='', encoding="utf-8") as f: 
            writer = csv.writer(f)
            writer.writerow(["image_path", "caption"])
            for path, caption in img2caption_pairs.items(): 
                writer.writerow([os.path.basename(path), caption])
        
        print(f"[INFO] Captions saved to {output_path}")
        
        return img2caption_pairs
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Root directory containing all images")
    parser.add_argument("--output", type=str, help="Output directory for caption csv")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for captioning")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for captioning")
    parser.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b", help="Model name or path")

    args = parser.parse_args() 
    
    pipeline = End2End_CaptioningPipeline(model_id=args.model)
    
    results = pipeline.process_directory(args.root, args.output, args.batch_size)