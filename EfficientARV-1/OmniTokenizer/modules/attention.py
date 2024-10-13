import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from beartype import beartype
from typing import Tuple

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from fairscale.nn import checkpoint_wrapper

def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # Refer to `Unroll` to see how this performs a maxpool-Nd
    # B, N, C
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def l2norm(t):
    return F.normalize(t, dim=-1)

def precompute_freqs_cis_2d(dim: int, end: int, theta: float = 10000.0, scale=1.0, use_cls=False):
    H = int( end**0.5 )
    # assert  H * H == end
    flat_patch_pos = torch.arange(0 if not use_cls else -1, end) # N = end
    x_pos = flat_patch_pos % H # N
    y_pos = flat_patch_pos // H # N
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(end if not use_cls else end + 1, -1)
    # we need to think how to implement this for multi heads.
    # freqs_cis = torch.cat([x_cis, y_cis], dim=-1) # N, Hc/2
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x: B N H Hc/2
    # freqs_cis:  N, H*Hc/2 or  N Hc/2
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape[-1] == x.shape[-1]:
        shape = [1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)]  # 1, N, 1, Hc/2
    else:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)] # 1, N, H, Hc/2
        # B, N, Hc/2
    return freqs_cis.view(*shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class Pooling(nn.Module):
    def __init__(self, pool_type, dim):
        super().__init__()
        if pool_type == "a":
            self.pool = nn.AvgPool2d(kernel_size=2)
        
        elif pool_type == "m":
            self.pool = nn.MaxPool2d(kernel_size=2)
        
        elif pool_type == "l":
            self.pool = nn.Linear(4 * dim, dim)

        else:
            raise NotImplementedError
        
        self.pool_type = pool_type

    def forward(self, x):
        # B N C
        B, N, C= x.shape
        if self.pool_type in ["a", "m"]:
            H, W = int(math.sqrt(N)), int(math.sqrt(N))
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = self.pool(x)
            x = x.view(B, C, -1).transpose(1, 2).contiguous()
        
        else:
            x = x.view(B, N//4, -1)
            x = self.pool(x)

        return x


class Up(nn.Module):
    def __init__(self, up_type, dim):
        super().__init__()
        if up_type == "n":
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        elif up_type == "r":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Rearrange('b c h w -> b (h w) c'),
                nn.Linear(dim, dim)
            )
            
        else:
            raise NotImplementedError
        
        self.up_type = up_type

    def forward(self, x):
        # B N C
        B, N, C= x.shape
        if self.up_type == "n":
            H, W = int(math.sqrt(N)), int(math.sqrt(N))
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = self.up(x)
            x = x.view(B, C, -1).transpose(1, 2).contiguous()
        
        else:
            #x = self.up(x) # B, N, 4c
            #x = x.view(B, N * 4, -1)
            H, W = int(math.sqrt(N)), int(math.sqrt(N))
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # B, C, H, W
            x = self.up(x) # B, (2H 2W), C
        
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.):
    """ Check this paper to understand the computation: https://arxiv.org/pdf/2002.05202.pdf"""
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        
        self.norm = LayerNorm(dim)
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        x = self.norm(x)

        x = x.view(B_, H, W, -1)
        # partition windows
        x_windows = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        BW, NW = x_windows.shape[:2]

        qkv = self.qkv(x_windows).reshape(BW, NW, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_windows = (attn @ v).transpose(1, 2).reshape(BW, NW, C)
        x_windows = self.proj(x_windows)
        x_windows = self.proj_drop(x_windows)

        x = window_reverse(x_windows, self.window_size[0], H, W)  # B H' W' C
        x = x.view(B_, H * W, C)

        return x




class PEG(nn.Module):
    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape
        if needs_shape:
            x = x.reshape(*shape, -1)
        
        x = rearrange(x, 'b ... d -> b d ...')

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.)
        x = self.dsconv(x)

        x = rearrange(x, 'b d ... -> b ... d')

        if needs_shape:
            x = rearrange(x, 'b ... d -> b (...) d')

        return x.reshape(orig_shape)

# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=64,
        heads=8,
        causal=False,
        num_null_kv=0,
        norm_context=True,
        dropout=0.,
        scale=8,
        spatial_pos="rel"
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if spatial_pos == "rel":
            self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads) # HACK this: whether shared pos encoding is better or on the contrary
        
        self.spatial_pos = spatial_pos
        self.freqs_cis = None

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)

        self.p_dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(
            dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        if self.num_null_kv > 0:
            self.null_kv = nn.Parameter(
                torch.randn(heads, 2 * num_null_kv, dim_head))
        else:
            self.null_kv = None

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.dim = inner_dim

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x,
        mask=None,
        context=None,
        is_spatial=True,
        q_stride=1,
    ):
        batch, device, dtype = x.shape[0], x.device, x.dtype

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)
        N = x.shape[1]

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b n h d', h=self.heads), (q, k, v))


        if self.spatial_pos == "rope" and is_spatial:
            if self.freqs_cis is None or self.freqs_cis.shape[0] != N:
                self.freqs_cis = precompute_freqs_cis_2d(self.dim // self.heads, N).to(x.device)

            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)

        q, k, v = map(lambda t: rearrange(
            t, 'b n h d -> b h n d', h=self.heads), (q, k, v))

        B, H, _, D = q.shape    
        if q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, H, q_stride, -1, D)
                .max(dim=2)
                .values
            )

        if self.num_null_kv > 0:
            nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d',
                            b=batch, r=2).unbind(dim=-2)

            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        if hasattr(F, "scaled_dot_product_attention") and torch.__version__ >= "2.1.0":
            # Note: the original paper did *not* use SDPA, it's a free boost!
            if exists(mask):
                mask = F.pad(mask, (self.num_null_kv, 0), value=True)
                mask = rearrange(mask, 'b j -> b 1 1 j')
            
            if self.spatial_pos == "rel" and is_spatial:
                h, w = int(math.sqrt(N)), int(math.sqrt(N))
                attn_bias = self.spatial_rel_pos_bias(h, w, device=x.device)
                attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)
            
            # query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p_dropout, is_causal=self.causal, scale=self.scale)
        
        else:
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            i, j = sim.shape[-2:]
            if self.spatial_pos == "rel" and is_spatial:
                h, w = int(math.sqrt(N)), int(math.sqrt(N))
                attn_bias = self.spatial_rel_pos_bias(h, w, device=x.device)
                attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)

                if sim.shape[2] != attn_bias.shape[1]:
                    # handle q_pooling here
                    q_len = sim.shape[2]
                    kv_len = sim.shape[3]
                    q_stride = kv_len // q_len
                    attn_bias = attn_bias[:, ::q_stride]
                    
                sim = sim + attn_bias

            if exists(mask):
                mask = F.pad(mask, (self.num_null_kv, 0), value=True)
                mask = rearrange(mask, 'b j -> b 1 1 j')
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

            if self.causal:
                sim = sim + self.rel_pos_bias(sim)

                causal_mask = torch.ones(
                    (i, j), device=device, dtype=torch.bool).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim=-1)
            attn = self.attn_dropout(attn)

            out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# alibi positional bias for extrapolation
class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') -
                          rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return self.bias


class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims=2,  # 2 for images, 3 for video
        layers=2,
        log_dist=True,
        cache_rel_pos=False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(
            nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent=False)

    def forward(self, *dimensions, device=torch.device('cpu')):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing='ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - \
                rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent=False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

# transformer


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        block,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        peg=False,
        peg_causal=False,
        attn_num_null_kv=2,
        has_cross_attn=False,
        attn_dropout=0.,
        ff_dropout=0.,
        window_size=4,
        spatial_pos="rel"
    ):
        super().__init__()
        assert len(block) == depth
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if block[i] == 't':
                self.layers.append(nn.ModuleList([
                    PEG(dim=dim, causal=peg_causal) if peg else None,
                    Attention(dim=dim, dim_head=dim_head, heads=heads,
                            causal=causal, dropout=attn_dropout, spatial_pos=spatial_pos),
                    Attention(dim=dim, dim_head=dim_head, dim_context=dim_context, heads=heads, causal=False,
                            num_null_kv=attn_num_null_kv, dropout=attn_dropout) if has_cross_attn else None,
                    FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
                ]))
            
            elif block[i] == 'w':
                self.layers.append(nn.ModuleList([
                    None,
                    WindowAttention(dim=dim, window_size=window_size, num_heads=heads, attn_drop=attn_dropout),
                    None,
                    FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
                ]))
            
            # various pooling methods: B, N, C
            elif block[i] in ['a', 'm', 'l']:
                self.layers.append(nn.ModuleList([
                    None,
                    Pooling(block[i], dim),
                    None,
                    FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
                ]))

            elif block[i] in ['n', 'r']:
                self.layers.append(nn.ModuleList([
                    None,
                    Up(block[i], dim),
                    None,
                    FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
                ]))

            else:
                raise NotImplementedError

        self.block = block
        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        context=None,
        self_attn_mask=None,
        cross_attn_context_mask=None,
        q_strides=None,
        is_spatial=True
    ):

        if q_strides is None:
            q_strides = '1' * len(self.layers)
        
        for blk, q_stride, (peg, self_attn, cross_attn, ff) in zip(self.block, q_strides, self.layers):
            if exists(peg):
                x = peg(x, shape=video_shape) + x

            if isinstance(self_attn, Attention):
                x = self_attn(x, mask=self_attn_mask, q_stride=int(q_stride), is_spatial=is_spatial) + do_pool(x, int(q_stride))
                # x = checkpoint.checkpoint(self_attn, x, self_attn_mask, None, attn_bias, int(q_stride))

            elif isinstance(self_attn, WindowAttention):
                x = self_attn(x) + x
            else:
                x = self_attn(x)
            
            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context,
                               mask=cross_attn_context_mask) + x

            x = ff(x) + x

            # deal with downsampling:
            if blk in ['a', 'm', 'l']:
                video_shape = (video_shape[0], video_shape[1], video_shape[2]//2, video_shape[3]//2) # video_shape: B, T, H, W
            
            elif blk in ['n', 'r']:
                video_shape = (video_shape[0], video_shape[1], int(video_shape[2]*2), int(video_shape[3]*2))
            

            if q_stride != '1':
                down_ratio = int(math.sqrt(int(q_stride)))
                video_shape = (video_shape[0], video_shape[1], video_shape[2]//down_ratio, video_shape[3]//down_ratio)

        return self.norm_out(x)
