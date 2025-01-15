import torch 
from glob import glob
import os 
from PIL import Image
from tqdm import tqdm
import csv
import argparse
from typing import Union
import math

def process_LlavaCaptions(caption: Union[list, str]): 
    if isinstance(caption, list):
        captions = []
        for cap in caption:
            if "Positive Prompt: " not in cap:
                captions.append(None)
                continue
            positive = cap[cap.find("Positive Prompt: ") + len("Positive Prompt: "):cap.find("Negative Prompt: ")-1].replace('"', '')
            captions.append(positive)
        return captions
    else: 
        if "Positive Prompt: " not in caption:
            return None
        positive = caption[caption.find("Positive Prompt: ") + len("Positive Prompt: "):caption.find("Negative Prompt: ")-1].replace('"', '')
        return positive

def load_pil_images(image_paths: list, size: int = 336):
    images = []
    for img_path in image_paths:
        image = Image.open(img_path)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            image = image.convert('RGBA')
            background = Image.new('RGBA', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background.convert('RGB')
        else:
            image = image.convert('RGB')
        image = image.resize((size, size))
        images.append(image)
    return images

def load_llava():
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
    )
    
    model.config.eos_token_id = 2
    model.config.pad_token_id = 2
    
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = 2
        model.generation_config.eos_token_id = 2
    
    processor.image_processor.size = {"height": 336, "width": 336}
    processor.tokenizer.padding_side = "left"
    
    return processor, model
    
class End2End_CaptioningPipeline:
    def __init__(self, prompt: str = None, batch_size: int = 1):

        self.processor, self.model = load_llava()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        self.prompt = prompt if prompt is not None else "Generate prompt and negative prompt for this image."
                
        self.model.to(self.device)
        
    def caption_image(self, image: list):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image"}
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        if len(image) > 1: 
            prompt = prompt * len(image)
        else: 
            image = image[0]
            
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(self.device, dtype=torch.float16)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.5
            )
        
        if len(image) > 1:
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)
        else: 
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return process_LlavaCaptions(caption)
    
    def process_directory(self, root_dir: str, output_path: str, image_size: int):
        image_paths = glob(os.path.join(root_dir, "*.png")) + glob(os.path.join(root_dir, "*.jpg"))
        
        img2caption_pairs = {}
        for index in tqdm(range(0, len(image_paths), self.batch_size), desc=f"[INFO] Captioning images in {root_dir}", total=math.ceil(len(image_paths)/self.batch_size)):
            batched_img_paths = image_paths[index:index+self.batch_size]
            img_batch = load_pil_images(batched_img_paths, image_size)
            try:
                captions = self.caption_image(img_batch)

                for idx, img in enumerate(batched_img_paths):
                    if captions[idx] is None: 
                        continue
                    img2caption_pairs[img] = captions[idx]
                    
            except Exception as e:
                print(f"Error processing the batch of images {image_paths[index: index + self.batch_size]}: {str(e)}")
                continue
            
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
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for the model if needed")
    parser.add_argument("--image_size", type=int, default=336, help="Size of the image to be resized (height, width)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for captioning")
    
    args = parser.parse_args()
    
    pipeline = End2End_CaptioningPipeline(args.prompt, args.batch_size)
    results = pipeline.process_directory(args.root, args.output, args.image_size)