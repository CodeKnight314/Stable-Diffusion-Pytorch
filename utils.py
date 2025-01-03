import os
import importlib
from tqdm import tqdm
import yaml

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Early stopping to stop the training when the loss does not improve after
        certain epochs.
        
        Args:
            patience (int): Number of epochs to wait before stopping the training.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        """
        Check if the model should stop training.
        
        Args:
            val_loss (float): The validation loss to check.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
def save_images(images: list, save_pth: str):
    """
    Save a list of images to a directory.
    
    Args:
        images (list): List of PIL images to save.
        save_pth (str): Path to save the images.
    """
    os.makedirs(save_pth, exist_ok=True)
    for i, img in enumerate(tqdm(images, total=len(images), desc=f"Saving images to {save_pth}")):
        img.save(os.path.join(save_pth, f"result_{i}.png"))
    print("Finished saving images")

def load_config(config_path: str):
    """
    Load configuration from a yaml file.
    
    Args:
        config_path (str): Path to the configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_module(module_config):
    """
    Load a module from a configuration dictionary.
    
    Args:
        module_config (dict): The module configuration dictionary.
        
    Returns:
        module_class: The module class
    """
    print(f"----------------------------------------")
    print(f"Loading module: {module_config['class']}")
    print("With parameters: ")
    for k, v in module_config.get("params", {}).items():
        print(f"    {k}: {v}")
    print(f"----------------------------------------")
    class_path = module_config["class"]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    module_class = getattr(module, class_name)
    params = module_config.get("params", {})
    return module_class(**params)

def load_optimizer(optimizer_config, model_parameters):
    """
    Load an optimizer from a configuration dictionary.
    
    Args:
        optimizer_config (dict): The optimizer configuration dictionary.
        model_parameters: The model parameters to optimize.
    
    Returns:
        optimizer: The optimizer object.
    """
    print(f"----------------------------------------")
    print(f"Loading optimizer: {optimizer_config['class']}")
    print("With parameters: ")
    for k, v in optimizer_config.get("params", {}).items():
        print(f"    {k}: {v}")
    print(f"----------------------------------------")
    class_path = optimizer_config["class"]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    module_class = getattr(module, class_name)
    params = optimizer_config.get("params", {})

    params["lr"] = float(params["lr"])
    params["eps"] = float(params["eps"])
    
    return module_class(model_parameters, **params)