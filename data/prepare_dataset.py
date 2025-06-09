# data/prepare_dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64, image_size=32, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
    ])

    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers # Use multiple CPU cores for loading data
    )

    return dataloader
