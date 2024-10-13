import torch
import torch.nn as nn

from .attention import SpatialAttention, TemporalAttention
from .common import ResidualBlock3D
from .gc_block import GlobalContextBlock
from .upsamplers import (SpatialTemporalUpsampler3D, SpatialUpsampler3D,
                         TemporalUpsampler3D)


def get_up_block(
    up_block_type: str,
    in_channels: int,
    out_channels: int,
    num_layers: int,
    act_fn: str,
    norm_num_groups: int = 32,
    norm_eps: float = 1e-6,
    dropout: float = 0.0,
    num_attention_heads: int = 1,
    output_scale_factor: float = 1.0,
    add_gc_block: bool = False,
    add_upsample: bool = True,
) -> nn.Module:
    if up_block_type == "SpatialUpBlock3D":
        return SpatialUpBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
            add_upsample=add_upsample,
        )
    elif up_block_type == "SpatialAttnUpBlock3D":
        return SpatialAttnUpBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            attention_head_dim=out_channels // num_attention_heads,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
            add_upsample=add_upsample,
        )
    elif up_block_type == "TemporalUpBlock3D":
        return TemporalUpBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
            add_upsample=add_upsample,
        )
    elif up_block_type == "TemporalAttnUpBlock3D":
        return TemporalAttnUpBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            attention_head_dim=out_channels // num_attention_heads,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
            add_upsample=add_upsample,
        )
    elif up_block_type == "SpatialTemporalUpBlock3D":
        return SpatialTemporalUpBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
            add_upsample=add_upsample,
        )
    else:
        raise ValueError(f"Unknown up block type: {up_block_type}")


class SpatialUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_gc_block: bool = False,
        add_upsample: bool = True,
    ):
        super().__init__()

        if add_upsample:
            self.upsampler = SpatialUpsampler3D(in_channels, in_channels)
        else:
            self.upsampler = None

        if add_gc_block:
            self.gc_block = GlobalContextBlock(in_channels, in_channels, fusion_type="mul")
        else:
            self.gc_block = None

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x


class SpatialAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_gc_block: bool = False,
        add_upsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )
            self.attentions.append(
                SpatialAttention(
                    out_channels,
                    nheads=out_channels // attention_head_dim,
                    head_dim=attention_head_dim,
                    bias=True,
                    upcast_softmax=True,
                    norm_num_groups=norm_num_groups,
                    eps=norm_eps,
                    rescale_output_factor=output_scale_factor,
                    residual_connection=True,
                )
            )

        if add_gc_block:
            self.gc_block = GlobalContextBlock(out_channels, out_channels, fusion_type="mul")
        else:
            self.gc_block = None

        if add_upsample:
            self.upsampler = SpatialUpsampler3D(out_channels, out_channels)
        else:
            self.upsampler = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv, attn in zip(self.convs, self.attentions):
            x = conv(x)
            x = attn(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x


class TemporalUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_gc_block: bool = False,
        add_upsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

        if add_gc_block:
            self.gc_block = GlobalContextBlock(out_channels, out_channels, fusion_type="mul")
        else:
            self.gc_block = None

        if add_upsample:
            self.upsampler = TemporalUpsampler3D(out_channels, out_channels)
        else:
            self.upsampler = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x


class TemporalAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_gc_block: bool = False,
        add_upsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )
            self.attentions.append(
                TemporalAttention(
                    out_channels,
                    nheads=out_channels // attention_head_dim,
                    head_dim=attention_head_dim,
                    bias=True,
                    upcast_softmax=True,
                    norm_num_groups=norm_num_groups,
                    eps=norm_eps,
                    rescale_output_factor=output_scale_factor,
                    residual_connection=True,
                )
            )

        if add_gc_block:
            self.gc_block = GlobalContextBlock(out_channels, out_channels, fusion_type="mul")
        else:
            self.gc_block = None

        if add_upsample:
            self.upsampler = TemporalUpsampler3D(out_channels, out_channels)
        else:
            self.upsampler = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv, attn in zip(self.convs, self.attentions):
            x = conv(x)
            x = attn(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x


class SpatialTemporalUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_gc_block: bool = False,
        add_upsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

        if add_gc_block:
            self.gc_block = GlobalContextBlock(out_channels, out_channels, fusion_type="mul")
        else:
            self.gc_block = None

        if add_upsample:
            self.upsampler = SpatialTemporalUpsampler3D(out_channels, out_channels)
        else:
            self.upsampler = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x
