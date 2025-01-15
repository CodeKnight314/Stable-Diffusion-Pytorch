import os
import torch 
from torch.utils.data import Dataset, DataLoader 
from typing import Tuple 
from torchvision import transforms as T
from glob import glob 
from PIL import Image
import pandas as pd
import argparse

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str, size: Tuple[int] = (512, 512)):
        super().__init__()
        
        self.root_dir = root_dir 
        self.csv = pd.read_csv(csv_path)
        self.dir = glob(os.path.join(root_dir, "*.jpg")) + glob(os.path.join(root_dir, "*.png"))
        
        self.transform = T.Compose([T.RandomVerticalFlip(0.25),
                                    T.RandomHorizontalFlip(0.25),
                                    T.Resize(size), 
                                    T.ToTensor(), 
                                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    def __len__(self): 
        return len(self.dir)
    
    def __getitem__(self, index):
        path = self.dir[index]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        
        caption = self.csv[self.csv['image_path'] == os.path.basename(path)]['caption'].values[0]
        return img_tensor, caption

def text_image_collate_fn(batch): 
    """
    Collate function for text-image datasets
    
    Args:
        batch (list): List of tuples containing image and prompt
    """
    images = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    return torch.stack(images), prompts

def load_dataset(root: str, csv_path: str, size: Tuple[int], batch_size: int): 
    """
    Load image-caption dataset
    
    Args:
        root (str): Root directory of the dataset containing images  
        csv_path (str): Path to the CSV file containing the prompts for the associated images
        size (Tuple[int]): Size of the image samples for resizing
        batch_size (int): Batch size for the dataset
    """
    dataset = ImageCaptionDataset(root, csv_path, size)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=os.cpu_count(), collate_fn=text_image_collate_fn, pin_memory=True)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--condition", action="store_true", help="Toggle option for condition/unconditioned image datasets")
    parser.add_argument("--size", type=int, default=512, help="Patch size for image samples")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for image dataset")
    args = parser.parse_args() 
    
    dataset = load_dataset(args.root, "data/dataset.csv", (args.size, args.size), args.batch)
    data = next(iter(dataset))
    img, prompt = data 
    print(f"Image shape from Conditional Image dataset: ", img.shape)
    print(f"Prompt: {prompt}")