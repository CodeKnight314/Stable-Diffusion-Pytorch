import os
from inference import inference
from diffusion import load_diffusion, StableDiffusion
from utils import load_config, load_module, save_images
import time

def display_screen(): 
    print(" ======================================================================")
    
    ascii_art = r''' |                                                                    | 
 |       .-"-.            .-"-.            .-"-.           .-"-.      |
 |     _/_-.-_\_        _/.-.-.\_        _/.-.-.\_       _/.-.-.\_.   |
 |    / __} {__ \      /|( o o )|\      ( ( o o ) )     ( ( o o ) )   |
 |   / //  "  \\ \    | //  "  \\ |      |/  "  \|       |/  "  \|    |
 |  / / \'---'/ \ \  / / \'---'/ \ \      \'/^\'/         \ \_/ /     |
 |  \ \_/`"""`\_/ /  \ \_/`"""`\_/ /      /`\ /`\         /`"""`\     |
 |   \           /    \           /      /  /|\  \       /       \    |
 |-={ see no evil }={ hear no evil }={ speak no evil }={ have fun }=-|       
    '''
    print(ascii_art)
    
def cli_interface(model: StableDiffusion, config: dict):
    scheduler = load_module(config["noise_scheduler"])
    scheduler.set_timesteps(config["noise_scheduler"]["num_steps"])

    model.eval()

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        display_screen()
        
        print(f"Loading StableDiffusion model with the following parameters:")
        print(f"Model ID:           {config['model_id']}")
        print(f"Sample Size:        {config['sample_size']}")
        print(f"Custom Weights loaded: {config['custom_weights']}")
        print(" ======================================================================")

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
            time.sleep(5)
            
if __name__ == "__main__":
    config = load_config("config.yaml")
    model = load_diffusion(config["model_params"], False)
    cli_interface(model, config["inference_params"])
