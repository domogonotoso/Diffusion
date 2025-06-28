# utils/visualizer.py

import os
import torchvision.utils as vutils


def save_images(images, output_dir="samples", prefix="sample"):
    """
    Save a batch of images to disk.
    Args:
        images (Tensor): shape (B, C, H, W)
        output_dir (str): directory to save images
        prefix (str): file name prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
        vutils.save_image(img, path, normalize=True)
