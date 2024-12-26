import os
import torch 
from torch.utils.data import Dataset, DataLoader 
from typing import Tuple 
from torchvision import transforms as T
from glob import glob 
from PIL import Image
import pandas as pd
import random

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
        
        self.dir = glob(os.path.join(root, "/*.jpg"))
        self.csv = pd.read_csv(csv_path)
        self.transform = T.Compose([T.Resize((size)), T.ToTensor()])
        
    def __len__(self): 
        return len(self.dir)
    
    def __getitem__(self, index):
        file_name = os.path.basename(self.dir[index])
        img = Image.open(self.dir[index]).convert("RGB")
        img_tensor = self.transform(img)
        
        comments = self.csv[self.csv['image_name'] == file_name]
        prompt = comments[comments['comment_number'] == random.randint(0, 5)]['caption'].iloc[0]
        
        return img_tensor, prompt
    
def load_FFHQ_dataset(root: str, size: Tuple[int], batch_size: int): 
    dataset = FFHQ(root, size)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

def load_Flickr30k(root: str, csv_path: str, size: Tuple[int], batch_size: int): 
    dataset = Flickr30K(root, csv_path, size, batch_size)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)