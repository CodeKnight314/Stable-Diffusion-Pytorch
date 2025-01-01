import os
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
def save_images(images, save_pth: str):
    os.makedirs(save_pth, exist_ok=True)
    for i, img in enumerate(tqdm(images, total=len(images), desc=f"Saving images to {save_pth}")):
        img.save(os.path.join(save_pth, f"result_{i}.png"))
    print("Finished saving images")