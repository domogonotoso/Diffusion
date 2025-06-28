import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb  # (B, dim)


class DownBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, base_channels * 8)
        )

        # Encoder
        self.down1 = DownBlock(in_channels, base_channels)               # 128 → 64
        self.down2 = DownBlock(base_channels, base_channels * 2)         # 64 → 32
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)     # 32 → 16
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)     # 16 → 8
        self.down5 = DownBlock(base_channels * 8, base_channels * 8)     # 8 → 4

        # Bottleneck
        self.bot_conv1 = nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1)
        self.bot_conv2 = nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1)

        # Decoder
        self.up5 = UpBlock(base_channels * 16, base_channels * 8)        # 4 → 8
        self.up4 = UpBlock(base_channels * 16, base_channels * 4)        # 8 → 16
        self.up3 = UpBlock(base_channels * 8, base_channels * 2)         # 16 → 32
        self.up2 = UpBlock(base_channels * 4, base_channels)             # 32 → 64
        self.up1 = UpBlock(base_channels * 2, base_channels)             # 64 → 128

        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

        self.time_emb_dim = time_emb_dim

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)[:, :, None, None]  # (B, C, 1, 1)

        # Encoder
        x1, skip1 = self.down1(x)  # 128 → 64
        x2, skip2 = self.down2(x1) # 64 → 32
        x3, skip3 = self.down3(x2) # 32 → 16
        x4, skip4 = self.down4(x3) # 16 → 8
        x5, skip5 = self.down5(x4) # 8 → 4

        # Bottleneck
        x = F.relu(self.bot_conv1(x5 + t_emb))
        x = F.relu(self.bot_conv2(x))

        # Decoder
        x = self.up5(x, skip5)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        return self.final_conv(x)
