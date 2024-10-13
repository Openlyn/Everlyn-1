import torch
import torch.nn as nn

from .attention import SpatialAttention, TemporalAttention
from .common import ResidualBlock3D
from .downsamplers import (SpatialDownsampler3D, SpatialTemporalDownsampler3D,
                           TemporalDownsampler3D)
from .gc_block import GlobalContextBlock


def get_down_block(
    down_block_type: str,
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
    add_downsample: bool = True,
) -> nn.Module:
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
        )
    elif down_block_type == "SpatialDownBlock3D":
        return SpatialDownBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
            add_downsample=add_downsample,
        )
    elif down_block_type == "SpatialAttnDownBlock3D":
        return SpatialAttnDownBlock3D(
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
            add_downsample=add_downsample,
        )
    elif down_block_type == "TemporalDownBlock3D":
        return TemporalDownBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
            add_downsample=add_downsample,
        )
    elif down_block_type == "TemporalAttnDownBlock3D":
        return TemporalAttnDownBlock3D(
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
            add_downsample=add_downsample,
        )
    elif down_block_type == "SpatialTemporalDownBlock3D":
        return SpatialTemporalDownBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_gc_block=add_gc_block,
            add_downsample=add_downsample,
        )
    else:
        raise ValueError(f"Unknown down block type: {down_block_type}")


class DownBlock3D(nn.Module):
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

        self.spatial_downsample_factor = 1
        self.temporal_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        return x


class SpatialDownBlock3D(nn.Module):
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
        add_downsample: bool = True,
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

        if add_downsample:
            self.downsampler = SpatialDownsampler3D(out_channels, out_channels)
            self.spatial_downsample_factor = 2
        else:
            self.downsampler = None
            self.spatial_downsample_factor = 1

        self.temporal_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x


class TemporalDownBlock3D(nn.Module):
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
        add_downsample: bool = True,
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

        if add_downsample:
            self.downsampler = TemporalDownsampler3D(out_channels, out_channels)
            self.temporal_downsample_factor = 2
        else:
            self.downsampler = None
            self.temporal_downsample_factor = 1

        self.spatial_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x


class SpatialTemporalDownBlock3D(nn.Module):
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
        add_downsample: bool = True,
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

        if add_downsample:
            self.downsampler = SpatialTemporalDownsampler3D(out_channels, out_channels)
            self.spatial_downsample_factor = 2
            self.temporal_downsample_factor = 2
        else:
            self.downsampler = None
            self.spatial_downsample_factor = 1
            self.temporal_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x


class SpatialAttnDownBlock3D(nn.Module):
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
        add_downsample: bool = True,
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

        if add_downsample:
            self.downsampler = SpatialDownsampler3D(out_channels, out_channels)
            self.spatial_downsample_factor = 2
        else:
            self.downsampler = None
            self.spatial_downsample_factor = 1

        self.temporal_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv, attn in zip(self.convs, self.attentions):
            x = conv(x)
            x = attn(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x


class TemporalDownBlock3D(nn.Module):
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
        add_downsample: bool = True,
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

        if add_downsample:
            self.downsampler = TemporalDownsampler3D(out_channels, out_channels)
            self.temporal_downsample_factor = 2
        else:
            self.downsampler = None
            self.temporal_downsample_factor = 1

        self.spatial_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x


class TemporalAttnDownBlock3D(nn.Module):
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
        add_downsample: bool = True,
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

        if add_downsample:
            self.downsampler = TemporalDownsampler3D(out_channels, out_channels)
            self.temporal_downsample_factor = 2
        else:
            self.downsampler = None
            self.temporal_downsample_factor = 1

        self.spatial_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv, attn in zip(self.convs, self.attentions):
            x = conv(x)
            x = attn(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x
