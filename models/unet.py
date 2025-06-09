# models/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """
    Downsampling block: Conv -> ReLU -> Conv -> ReLU -> Downsample
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block: Upsample -> Conv -> ReLU -> Conv -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1) # Concatenate along channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """
    A minimal UNet for 32x32 images.
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        # Encoder
        self.down1 = DownBlock(in_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bot_conv1 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1)
        self.bot_conv2 = nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1)

        # Decoder
        self.up3 = UpBlock(base_channels * 8, base_channels * 2)
        self.up2 = UpBlock(base_channels * 4, base_channels)
        self.up1 = UpBlock(base_channels * 2, base_channels)

        # Final output
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1, skip1 = self.down1(x)  # 32 → 16
        x2, skip2 = self.down2(x1) # 16 → 8
        x3, skip3 = self.down3(x2) # 8 → 4

        # Bottleneck
        x = F.relu(self.bot_conv1(x3))
        x = F.relu(self.bot_conv2(x))

        # Decoder
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        # Output
        return self.final_conv(x)
