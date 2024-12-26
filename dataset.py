import os
import torch 
from torch.utils.data import Dataset, DataLoader 
from typing import Tuple 
from torchvision import transforms as T
from glob import glob 
from PIL import Image
import pandas as pd
import random
import argparse

# https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
class FFHQ(Dataset): 
    def __init__(self, root: str, size: Tuple[int] = (512, 512)): 
        super().__init__()
        
        self.root = root
        self.dir = glob(os.path.join(root, "/*.png"))
        self.transform = T.Compose([T.Resize(size), T.ToTensor()])
        
    def __len__(self): 
        return len(self.dir)
    
    def __getitem__(self, index):
        img = Image.open(self.dir[index]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor
    
# https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
class Flickr30K(Dataset):
    def __init__(self, root: str, csv_path: str, size: Tuple[int] = (512, 512)): 
        super().__init__() 
        
        self.dir = glob(os.path.join(root, "*.jpg"))
        self.csv = pd.read_csv(csv_path, sep='|', encoding='utf-8')
        self.csv.columns = self.csv.columns.str.strip()
        self.csv['comment'] = self.csv['comment'].str.replace(r'\s+\.', '.', regex=True)        
        
        self.transform = T.Compose([T.Resize((size)), T.ToTensor()])
        
    def __len__(self): 
        return len(self.dir)
    
    def __getitem__(self, index):
        file_name = os.path.basename(self.dir[index])
        img = Image.open(self.dir[index]).convert("RGB")
        img_tensor = self.transform(img)
        
        comments = self.csv[self.csv['image_name'] == file_name]
        comment_index = random.randint(0, len(comments['comment']) - 1)
        prompt = comments['comment'].iloc[comment_index]
        
        return img_tensor, prompt
    
def load_FFHQ_dataset(root: str, size: Tuple[int], batch_size: int): 
    dataset = FFHQ(root, size)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

def load_Flickr30k(root: str, csv_path: str, size: Tuple[int], batch_size: int): 
    dataset = Flickr30K(root, csv_path, size)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--condition", action="store_true", help="Toggle option for condition/unconditioned image datasets")
    parser.add_argument("--size", type=int, default=512, help="Patch size for image samples")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for image dataset")
    args = parser.parse_args() 
    
    if args.condition: 
        dataset = load_Flickr30k(args.root, os.path.join(args.root, "results.csv"), (args.size, args.size), args.batch)
    else: 
        dataset = load_FFHQ_dataset(args.root, (args.size, args.size), args.batch)
        
    data = next(iter(dataset))
    if args.condition: 
        img, prompt = data 
        print(f"Image shape from Conditional Image dataset: ", img.shape)
        print(f"Prompt: {prompt}")
    else: 
        img = data
        print(f"Image shape from Unconditional Image dataset: ", img.shape)