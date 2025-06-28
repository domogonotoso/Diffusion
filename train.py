# train.py

import yaml
import os
import torch
import torch.optim as optim
from tqdm import tqdm

from data.prepare_dataset import get_dataloader
from models.unet import UNet
from diffusion.scheduler import DiffusionScheduler
from diffusion.diffusion import Diffusion


def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train():
    config = load_config()

    # Config variables
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = float(config["learning_rate"])
    image_size = config["image_size"]
    timesteps = config["timesteps"]
    save_every = config.get("save_every", 5)
    save_dir = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataloader = get_dataloader(
    batch_size=batch_size,
    image_size=image_size,
    target_class="cat" 
)


    # Initialize model, scheduler, diffusion
    model = UNet(in_channels=3, out_channels=3).to(device)
    scheduler = DiffusionScheduler(timesteps=timesteps)
    diffusion = Diffusion(model, scheduler, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for step, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(device)
            loss = diffusion.loss(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
            print(f"✅ Saved best model at epoch {epoch+1}")

        # Save every N epochs
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))
            print(f"📦 Saved checkpoint at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "model_last.pt"))
    print("💾 Saved final model.")


if __name__ == "__main__":
    train()
