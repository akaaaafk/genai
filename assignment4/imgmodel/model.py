import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyModel(nn.Module):
    def __init__(self, in_channels=3):
        super(EnergyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)   # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)           # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)          # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)          # 8x8 -> 4x4

        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.act(self.fc1(x))
        energy = self.fc2(x)

        return energy.squeeze(-1)


def sinusoidal_time_embedding(timesteps, dim, max_period=10000):
    half = dim // 2

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )

    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))

    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim, time_dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, t):
        embedding = sinusoidal_time_embedding(t, self.dim)
        return self.mlp(embedding)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(ResidualBlock, self).__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))

        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_dim=256):
        super(UNet, self).__init__()

        self.time_embedding = TimeEmbedding(dim=base_channels, time_dim=time_dim)

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.down1_pool = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.down2_pool = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down3_pool = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)

        # Bottleneck at 8x8
        self.bottleneck1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.bottleneck2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        # Upsampling path: 8x8 -> 16x16 -> 32x32 -> 64x64 (with skip connections)
        self.up3_upsample = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.up3 = ResidualBlock(base_channels * 4 + base_channels * 4, base_channels * 2, time_dim)

        self.up2_upsample = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_dim)

        self.up1_upsample = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(base_channels + base_channels, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x = self.in_conv(x)

        skip1 = self.down1(x, t_emb)      # (B, C,   64, 64)
        x = self.down1_pool(skip1)        # (B, C,   32, 32)

        skip2 = self.down2(x, t_emb)      # (B, 2C,  32, 32)
        x = self.down2_pool(skip2)        # (B, 2C,  16, 16)

        skip3 = self.down3(x, t_emb)      # (B, 4C,  16, 16)
        x = self.down3_pool(skip3)        # (B, 4C,  8,  8)

        x = self.bottleneck1(x, t_emb)
        x = self.bottleneck2(x, t_emb)

        x = self.up3_upsample(x)
        x = self.up3(torch.cat([x, skip3], dim=1), t_emb)

        x = self.up2_upsample(x)
        x = self.up2(torch.cat([x, skip2], dim=1), t_emb)

        x = self.up1_upsample(x)
        x = self.up1(torch.cat([x, skip1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


def get_model(model_name):
    if model_name == "EBM":
        return EnergyModel(in_channels=3)

    elif model_name == "UNet":
        return UNet(in_channels=3, base_channels=64, time_dim=256)

    else:
        raise ValueError(f"Model {model_name} not recognized.")
