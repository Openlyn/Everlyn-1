import math

import torch
import torch.nn as nn

from .downsamplers import BlurPooling2D, BlurPooling3D


class DiscriminatorBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.BatchNorm2d(in_channels)

        self.nonlinearity = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.downsampler = BlurPooling2D(out_channels, out_channels)
        else:
            self.downsampler = nn.Identity()

        self.norm2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.shortcut = nn.Sequential(
                BlurPooling2D(in_channels, in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = nn.Identity()

        self.spatial_downsample_factor = 2
        self.temporal_downsample_factor = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.downsampler(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class Discriminator2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels = (64,),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = out_channels
            is_final_block = i == len(block_out_channels) - 1

            self.blocks.append(
                DiscriminatorBlock2D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    output_scale_factor=math.sqrt(2),
                    add_downsample=not is_final_block,
                )
            )

        self.conv_norm_out = nn.BatchNorm2d(block_out_channels[-1])
        self.conv_act = nn.LeakyReLU(0.2)

        self.conv_out = nn.Conv2d(block_out_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)

        return x


class DiscriminatorBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.GroupNorm(32, in_channels)

        self.nonlinearity = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.downsampler = BlurPooling3D(out_channels, out_channels)
        else:
            self.downsampler = nn.Identity()

        self.norm2 = nn.GroupNorm(32, out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.shortcut = nn.Sequential(
                BlurPooling3D(in_channels, in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )

        self.spatial_downsample_factor = 2
        self.temporal_downsample_factor = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.downsampler(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class Discriminator3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels = (64,),
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1, stride=2)

        self.blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = out_channels
            is_final_block = i == len(block_out_channels) - 1

            self.blocks.append(
                DiscriminatorBlock3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    output_scale_factor=math.sqrt(2),
                    add_downsample=not is_final_block,
                )
            )

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1])
        self.conv_act = nn.LeakyReLU(0.2)

        self.conv_out = nn.Conv3d(block_out_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)

        return x
