import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CausalConv3d


class Downsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_downsample_factor: int = 1,
        temporal_downsample_factor: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor


class SpatialDownsampler3D(Downsampler):
    def __init__(self, in_channels: int, out_channels):
        if out_channels is None:
            out_channels = in_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_downsample_factor=2,
            temporal_downsample_factor=1,
        )

        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class TemporalDownsampler3D(Downsampler):
    def __init__(self, in_channels: int, out_channels):
        if out_channels is None:
            out_channels = in_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_downsample_factor=1,
            temporal_downsample_factor=2,
        )

        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialTemporalDownsampler3D(Downsampler):
    def __init__(self, in_channels: int, out_channels):
        if out_channels is None:
            out_channels = in_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_downsample_factor=2,
            temporal_downsample_factor=2,
        )

        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class BlurPooling2D(Downsampler):
    def __init__(self, in_channels: int, out_channels):
        if out_channels is None:
            out_channels = in_channels

        assert in_channels == out_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_downsample_factor=2,
            temporal_downsample_factor=1,
        )

        filt = torch.tensor([1, 2, 1], dtype=torch.float32)
        filt = torch.einsum("i,j -> ij", filt, filt)
        filt = filt / filt.sum()
        filt = filt[None, None].repeat(out_channels, 1, 1, 1)

        self.register_buffer("filt", filt)
        self.filt: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return F.conv2d(x, self.filt, stride=2, padding=1, groups=self.in_channels)


class BlurPooling3D(Downsampler):
    def __init__(self, in_channels: int, out_channels):
        if out_channels is None:
            out_channels = in_channels

        assert in_channels == out_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_downsample_factor=2,
            temporal_downsample_factor=2,
        )

        filt = torch.tensor([1, 2, 1], dtype=torch.float32)
        filt = torch.einsum("i,j,k -> ijk", filt, filt, filt)
        filt = filt / filt.sum()
        filt = filt[None, None].repeat(out_channels, 1, 1, 1, 1)

        self.register_buffer("filt", filt)
        self.filt: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        return F.conv3d(x, self.filt, stride=2, padding=1, groups=self.in_channels)
