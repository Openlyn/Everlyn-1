import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GlobalContextBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        min_channels: int = 16,
        init_bias: float = -10.,
        fusion_type: str = "mul",
    ):
        super().__init__()

        assert fusion_type in ("mul", "add"), f"Unsupported fusion type: {fusion_type}"
        self.fusion_type = fusion_type

        self.conv_ctx = nn.Conv2d(in_channels, 1, kernel_size=1)

        num_channels = max(min_channels, out_channels // 2)

        if fusion_type == "mul":
            self.conv_mul = nn.Sequential(
                nn.Conv2d(in_channels, num_channels, kernel_size=1),
                nn.LayerNorm([num_channels, 1, 1]), # TODO: LayerNorm or GroupNorm?
                nn.LeakyReLU(0.1),
                nn.Conv2d(num_channels, out_channels, kernel_size=1),
                nn.Sigmoid(),
            )

            nn.init.zeros_(self.conv_mul[-2].weight)
            nn.init.constant_(self.conv_mul[-2].bias, init_bias)
        else:
            self.conv_add = nn.Sequential(
                nn.Conv2d(in_channels, num_channels, kernel_size=1),
                nn.LayerNorm([num_channels, 1, 1]), # TODO: LayerNorm or GroupNorm?
                nn.LeakyReLU(0.1),
                nn.Conv2d(num_channels, out_channels, kernel_size=1),
            )

            nn.init.zeros_(self.conv_add[-1].weight)
            nn.init.constant_(self.conv_add[-1].bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_image = x.ndim == 4
        if is_image:
            x = rearrange(x, "b c h w -> b c 1 h w")

        # x: (B, C, T, H, W)
        orig_x = x
        batch_size = x.shape[0]

        x = rearrange(x, "b c t h w -> (b t) c h w")

        ctx = self.conv_ctx(x)
        ctx = rearrange(ctx, "b c h w -> b c (h w)")
        ctx = F.softmax(ctx, dim=-1)

        flattened_x = rearrange(x, "b c h w -> b c (h w)")

        x = torch.einsum("b c1 n, b c2 n -> b c2 c1", ctx, flattened_x)
        x = rearrange(x, "... -> ... 1")

        if self.fusion_type == "mul":
            mul_term = self.conv_mul(x)
            mul_term = rearrange(mul_term, "(b t) c h w -> b c t h w", b=batch_size)
            x = orig_x * mul_term
        else:
            add_term = self.conv_add(x)
            add_term = rearrange(add_term, "(b t) c h w -> b c t h w", b=batch_size)
            x = orig_x + add_term

        if is_image:
            x = rearrange(x, "b c 1 h w -> b c h w")

        return x
