import torch 
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, LlavaNextForConditionalGeneration, AutoModelForCausalLM, LlamaTokenizer, LlavaNextProcessor
from glob import glob
import os 
from PIL import Image
from tqdm import tqdm
import csv
import argparse

def load_llava():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    
    model.config.eos_token_id = 2
    model.config.pad_token_id = 2
    
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = 2
        model.generation_config.eos_token_id = 2
    
    processor.image_processor.size = {"height": 336, "width": 336}
    
    return processor, model

def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        load_in_8bit=True
    )
    return processor, model

def load_cogvlm():
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    
    return tokenizer, model
    
class End2End_CaptioningPipeline:
    def __init__(self, model_id: str = "llava", prompt: str = None):
        model_loaders = {
            "llava": load_llava,
            "blip2": load_blip2,
            "cogvlm": load_cogvlm
        }
        
        if model_id not in model_loaders:
            raise ValueError(f"Model {model_id} not supported. Choose from {list(model_loaders.keys())}")
        
        self.processor, self.model = model_loaders[model_id]()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        
        self.prompt = prompt if prompt is not None else "Generate prompt and negative prompt for this image."
                
        self.model.to(self.device)
        
    def caption_image(self, image: Image.Image):
        if self.model_id == "llava":
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
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=10,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.5
                )
                
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)

            positive = caption[caption.find("Positive Prompt: ") + len("Positive Prompt: "):caption.find("Negative Prompt: ")-1].replace('"', '')
            negative = caption[caption.find("Negative Prompt: ") + len("Negative Prompt: "):].replace('"','')
            caption = positive + negative
            
            return caption
            
        elif self.model_id == "cogvlm":
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=self.prompt, history=[], images=[image])
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
                'images': [[inputs['images'][0].to(self.device).to(torch.bfloat16)]],
            }
            gen_kwargs = {"max_length": 2048, "do_sample": False}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                
            return self.output_decoder.decode(outputs[0], skip_special_tokens=True)
        
        else: 
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
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
                
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption
    
    def process_directory(self, root_dir: str, output_path: str, image_size: tuple):
        image_paths = glob(os.path.join(root_dir, "*.png")) + glob(os.path.join(root_dir, "*.jpg"))
        
        img2caption_pairs = {}
        for img_path in tqdm(image_paths, desc=f"[INFO] Captioning images in {root_dir}"):
            try:
                image = Image.open(img_path)
                if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                    image = image.convert('RGBA')
                    background = Image.new('RGBA', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background.convert('RGB')
                else:
                    image = image.convert('RGB')
                image = image.resize(image_size)
                
                caption = self.caption_image(image)
                img2caption_pairs[img_path] = caption
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
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
    parser.add_argument("--model", type=str, default="llava", choices=["llava", "blip2", "cogvlm"], help="Model to use for captioning")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for the model if needed")
    parser.add_argument("--image_size", type=int, nargs=2, default=[336, 336], 
                        help="Size of the image to be resized (height, width)")
    
    args = parser.parse_args()
    
    pipeline = End2End_CaptioningPipeline(args.model, args.prompt)
    results = pipeline.process_directory(args.root, args.output, tuple(args.image_size))