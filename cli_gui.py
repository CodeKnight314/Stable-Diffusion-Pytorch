import os
from inference import inference
from diffusion import load_diffusion, StableDiffusion
from utils import load_config, load_module, save_images
import argparse

def display_screen(): 
    print(" =========================================================================")
    
    ascii_art = r''' |                                                                       | 
 |       .-"-.            .-"-.            .-"-.            .-"-.        |
 |     _/_-.-_\_        _/.-.-.\_        _/.-.-.\_        _/.\./.\_.     |
 |    / __} {__ \      /|( o o )|\      ( ( o o ) )      ( ( o o ) )     |
 |   / //  "  \\ \    | //  "  \\ |      |/  "  \|        |/  "  \|      |
 |  / / \'---'/ \ \  / / \'---'/ \ \      \'/^\'/          \ \_/ /       |
 |  \ \_/`"""`\_/ /  \ \_/`"""`\_/ /      /`\ /`\          /`"""`\       |
 |   \           /    \           /      /  /|\  \        /       \      |
 |-={ see no evil }={ hear no evil }={ speak no evil }={ do all evil }=- |'''
    print(ascii_art)
    
def cli_interface(model: StableDiffusion, config: dict):
    scheduler = load_module(config["noise_scheduler"])
    scheduler.set_timesteps(config["noise_scheduler"]["num_steps"])
    model.unet.eval()
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        display_screen()
        
        print(" |                                                                       |")
        print(" | Loading StableDiffusion model with the following parameters:          |")
        print(f" | Model ID:                   {str(config['model_params']['model_id'])[:38]:<38}    |")
        print(f" | Sample Size:                {str(config['model_params']['sample_size'])[:38]:<38}    |")
        if config['model_params'].get('weights_path'):
            weights_path = config['model_params']['weights_path']
            if len(weights_path) > 38:
                weights_path = "..." + weights_path[-35:]
            print(f" | Weights Path:              {weights_path:<38}     |")
        print(" |                                                                       |")
        print(" =========================================================================")
        print("\n[Options]")
        print("1. Generate Image")
        print("2. Exit")
        choice = input("\nEnter your choice (1/2): ").strip()
        
        if choice == '2':
            print("\nExiting... Have a great day!")
            break
        elif choice == '1':
            prompt = input("\nEnter the text prompt for image generation: ").strip()
            guidance_scale = float(input("Enter guidance scale (default 7.5): ").strip() or "7.5")
            num_images = int(input("Enter number of images to generate: ").strip() or "1")
            output_dir = "SD_results_save"
            os.makedirs(output_dir, exist_ok=True)
            
            print("\n[INFO] Generating images...")
            images = []
            for i in range(num_images):
                sample = inference(model, prompt=prompt, scheduler=scheduler, guidance_scale=guidance_scale)
                images.append(sample)
            save_images(images, output_dir)
            print(f"\n[INFO] Images saved to '{output_dir}'")
            input("\nPress Enter to return to the main menu...")
        else:
            print("\nInvalid choice! Please try again.")
            input("\nPress Enter to return to the main menu...")

def setup_model(config: dict, weights_path: str = None) -> StableDiffusion:
    """
    Set up the model with optional custom weights loading
    """
    if weights_path:
        config['model_params']['weights_path'] = weights_path
        
    model = load_diffusion(config["model_params"], False)
    
    if weights_path:
        print(f"\n[INFO] Loading custom weights from: {weights_path}")
        try:
            model.load_model(weights_path)
            print("[INFO] Custom weights loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load weights from {weights_path}: {str(e)}")
            print("[INFO] Falling back to default weights")
            config['model_params'].pop('weights_path', None)
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Stable Diffusion CLI with custom weights loading')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--weights', type=str, help='Path to custom model weights (can be absolute or relative)')
    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stable Diffusion CLI with custom weights loading')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--weights', type=str, help='Path to custom model weights (can be absolute or relative)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    weights_path = None
    if args.weights:
        weights_path = os.path.abspath(args.weights)
    
    model = setup_model(config, weights_path)
    
    cli_interface(model, config)