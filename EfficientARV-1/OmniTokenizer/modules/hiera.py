from typing import List, Tuple, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, Mlp


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # Refer to `Unroll` to see how this performs a maxpool-Nd
    # B, N, C
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values

class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = x.shape
        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            # B, Ws, Nw, 3, H, C -> 3, B, H, Nw, Ws, C -> B, H, Nw, Ws, C
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = (attn @ v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.softmax(dim=-1),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.projection = nn.Linear(dim, num_classes)
        # act_fun for eval and testing only
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x

