# sample.py

import os
import yaml
import torch

from models.unet import UNet
from diffusion.scheduler import DiffusionScheduler
from diffusion.diffusion import Diffusion
from utils.visualizer import save_images


def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def sample():
    config = load_config()
    image_size = config["image_size"]
    timesteps = config["timesteps"]
    save_dir = config.get("save_dir", "checkpoints")
    sample_dir = "samples"
    os.makedirs(sample_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(in_channels=3, out_channels=3).to(device)
    model_path = os.path.join(save_dir, "model_last.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Diffusion sampler
    scheduler = DiffusionScheduler(timesteps=timesteps)
    diffusion = Diffusion(model, scheduler, device)

    # Generate images
    with torch.no_grad():
        images = diffusion.sample(batch_size=16, image_size=image_size, channels=3)

    # Save to disk
    save_images(images, output_dir=sample_dir, prefix="sample")
    print("✅ Saved generated images to 'samples/' directory.")


if __name__ == "__main__":
    sample()
