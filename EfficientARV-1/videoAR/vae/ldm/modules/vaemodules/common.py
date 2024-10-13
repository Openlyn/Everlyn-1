import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .activations import get_activation


def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

class CausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3, # : int | tuple[int, int, int], 
        stride=1, # : int | tuple[int, int, int] = 1,
        padding=1, # : int | tuple[int, int, int],  # TODO: change it to 0.
        dilation=1, # :  int | tuple[int, int, int] = 1,
        **kwargs,
    ):
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        assert len(kernel_size) == 3, f"Kernel size must be a 3-tuple, got {kernel_size} instead."

        stride = stride if isinstance(stride, tuple) else (stride,) * 3
        assert len(stride) == 3, f"Stride must be a 3-tuple, got {stride} instead."

        dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        assert len(dilation) == 3, f"Dilation must be a 3-tuple, got {dilation} instead."

        t_ks, h_ks, w_ks = kernel_size
        _, h_stride, w_stride = stride
        t_dilation, h_dilation, w_dilation = dilation

        t_pad = (t_ks - 1) * t_dilation
        # TODO: align with SD
        if padding is None:
            h_pad = math.ceil(((h_ks - 1) * h_dilation + (1 - h_stride)) / 2)
            w_pad = math.ceil(((w_ks - 1) * w_dilation + (1 - w_stride)) / 2)
        elif isinstance(padding, int):
            h_pad = w_pad = padding
        else:
            assert NotImplementedError

        self.temporal_padding = t_pad
        self.temporal_padding_origin = math.ceil(((t_ks - 1) * w_dilation + (1 - w_stride)) / 2)
        self.padding_flag = 0

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(0, h_pad, w_pad),
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if self.padding_flag == 0:
            x = F.pad(
                x,
                pad=(0, 0, 0, 0, self.temporal_padding, 0),
                mode="replicate",     # TODO: check if this is necessary
            )
        else:
            x = F.pad(
                x,
                pad=(0, 0, 0, 0, self.temporal_padding_origin, self.temporal_padding_origin),
            )
        return super().forward(x)
    
    def set_padding_one_frame(self):
        def _set_padding_one_frame(name, module):
            if hasattr(module, 'padding_flag'):
                print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 1
            for sub_name, sub_mod in module.named_children():
                _set_padding_one_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_padding_one_frame(name, module)

    def set_padding_more_frame(self):
        def _set_padding_more_frame(name, module):
            if hasattr(module, 'padding_flag'):
                print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 2
            for sub_name, sub_mod in module.named_children():
                _set_padding_more_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_padding_more_frame(name, module)

class ResidualBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )

        self.nonlinearity = get_activation(non_linearity)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=out_channels,
            eps=norm_eps,
            affine=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )

        self.nonlinearity = get_activation(non_linearity)

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=out_channels,
            eps=norm_eps,
            affine=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.conv2(x)
        return (x + shortcut) / self.output_scale_factor


class SpatialNorm2D(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.FloatTensor, zq: torch.FloatTensor) -> torch.FloatTensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class SpatialNorm3D(SpatialNorm2D):
    def forward(self, f: torch.FloatTensor, zq: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = f.shape[0]
        f = rearrange(f, "b c t h w -> (b t) c h w")
        zq = rearrange(zq, "b c t h w -> (b t) c h w")

        x = super().forward(f, zq)

        x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)

        return x
