import torch
import torch.nn as nn

from .attention import Attention3D, SpatialAttention, TemporalAttention
from .common import ResidualBlock3D


def get_mid_block(
    mid_block_type: str,
    in_channels: int,
    num_layers: int,
    act_fn: str,
    norm_num_groups: int = 32,
    norm_eps: float = 1e-6,
    dropout: float = 0.0,
    add_attention: bool = True,
    attention_type: str = "3d",
    num_attention_heads: int = 1,
    output_scale_factor: float = 1.0,
) -> nn.Module:
    if mid_block_type == "MidBlock3D":
        return MidBlock3D(
            in_channels=in_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            add_attention=add_attention,
            attention_type=attention_type,
            attention_head_dim=in_channels // num_attention_heads,
            output_scale_factor=output_scale_factor,
        )
    else:
        raise ValueError(f"Unknown mid block type: {mid_block_type}")


class MidBlock3D(nn.Module):
    """
    A 3D UNet mid-block [`MidBlock3D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        norm_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_type: (`str`, *optional*, defaults to `3d`): The type of attention to use. Defaults to `3d`.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, temporal_length, height, width)`.

    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        add_attention: bool = True,
        attention_type: str = "3d",
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.attention_type = attention_type

        norm_num_groups = norm_num_groups if norm_num_groups is not None else min(in_channels // 4, 32)

        self.convs = nn.ModuleList([
            ResidualBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                non_linearity=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                dropout=dropout,
                output_scale_factor=output_scale_factor,
            )
        ])

        self.attentions = nn.ModuleList([])
        for _ in range(num_layers - 1):
            if add_attention:
                if attention_type == "3d":
                    self.attentions.append(
                        Attention3D(
                            in_channels,
                            nheads=in_channels // attention_head_dim,
                            head_dim=attention_head_dim,
                            bias=True,
                            upcast_softmax=True,
                            norm_num_groups=norm_num_groups,
                            eps=norm_eps,
                            rescale_output_factor=output_scale_factor,
                            residual_connection=True,
                        )
                    )
                elif attention_type == "spatial_temporal":
                    self.attentions.append(
                        nn.ModuleList([
                            SpatialAttention(
                                in_channels,
                                nheads=in_channels // attention_head_dim,
                                head_dim=attention_head_dim,
                                bias=True,
                                upcast_softmax=True,
                                norm_num_groups=norm_num_groups,
                                eps=norm_eps,
                                rescale_output_factor=output_scale_factor,
                                residual_connection=True,
                            ),
                            TemporalAttention(
                                in_channels,
                                nheads=in_channels // attention_head_dim,
                                head_dim=attention_head_dim,
                                bias=True,
                                upcast_softmax=True,
                                norm_num_groups=norm_num_groups,
                                eps=norm_eps,
                                rescale_output_factor=output_scale_factor,
                                residual_connection=True,
                            ),
                        ])
                    )
                elif attention_type == "spatial":
                    self.attentions.append(
                        SpatialAttention(
                            in_channels,
                            nheads=in_channels // attention_head_dim,
                            head_dim=attention_head_dim,
                            bias=True,
                            upcast_softmax=True,
                            norm_num_groups=norm_num_groups,
                            eps=norm_eps,
                            rescale_output_factor=output_scale_factor,
                            residual_connection=True,
                        )
                    )
                elif attention_type == "temporal":
                    self.attentions.append(
                        TemporalAttention(
                            in_channels,
                            nheads=in_channels // attention_head_dim,
                            head_dim=attention_head_dim,
                            bias=True,
                            upcast_softmax=True,
                            norm_num_groups=norm_num_groups,
                            eps=norm_eps,
                            rescale_output_factor=output_scale_factor,
                            residual_connection=True,
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attention_type}")
            else:
                self.attentions.append(None)

            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.convs[0](hidden_states)

        for attn, resnet in zip(self.attentions, self.convs[1:]):
            if attn is not None:
                if self.attention_type == "spatial_temporal":
                    spatial_attn, temporal_attn = attn
                    hidden_states = spatial_attn(hidden_states)
                    hidden_states = temporal_attn(hidden_states)
                else:
                    hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)

        return hidden_states
