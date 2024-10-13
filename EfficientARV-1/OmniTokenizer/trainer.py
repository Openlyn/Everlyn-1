import os
import json
import tempfile
from random import random
import math
from math import log2, floor
from pathlib import Path
from functools import partial
from contextlib import contextmanager, ExitStack
from pathlib import Path
from shutil import rmtree

import torch
from torch.optim import Adam
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad as torch_grad

from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from kornia.filters import filter2d

from huggan.pytorch.lightweight_gan.diff_augment import DiffAugment

from tqdm import tqdm
from einops import rearrange, reduce, repeat

from datasets import load_dataset

from accelerate import Accelerator, DistributedDataParallelKwargs
from huggingface_hub import hf_hub_download, create_repo

from huggan.pytorch.huggan_mixin import HugGANModelHubMixin
from huggan.utils.hub import get_full_repo_name

# constants

# NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']
PYTORCH_WEIGHTS_NAME = 'model.pt'


# helpers

def exists(val):
    return val is not None


@contextmanager
def null_context():
    yield


def is_power_of_two(val):
    return log2(val).is_integer()


def default(val, d):
    return val if exists(val) else d


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = '' if int(n >= 0) else '-'
        res = float(f'{prefix}inf')
    return res


# loss functions

def gen_hinge_loss(fake, real):
    return fake.mean()


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i=t1.shape[0])
        t = torch.cat((t1, t2), dim=-1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device=device, dtype=torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)


# helper classes

class NanException(Exception):
    pass


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class ChanNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))


class Fuzziness(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


Blur = nn.Identity


# attention

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=h), (q, k, v))

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


# dataset

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class identity(object):
    def __call__(self, tensor):
        return tensor


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0., types=[], detach=False, **kwargs):
        context = torch.no_grad if detach else null_context

        with context():
            if random() < prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=types)

        return self.D(images, **kwargs)


# modifiable global variables

norm_class = nn.BatchNorm2d


def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor)


# squeeze excitation classes

# global context network
# https://arxiv.org/abs/2012.13375
# similar to squeeze-excite, but with a simplified attention pooling and a subsequent layer norm

class GlobalContext(nn.Module):
    def __init__(
            self,
            *,
            chan_in,
            chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim=-1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)


# frequency channel attention
# https://arxiv.org/abs/2012.11879

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))


def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part: (i + 1) * c_part, x, y] = coor_value

    return dct_weights


class FCANet(nn.Module):
    def __init__(
            self,
            *,
            chan_in,
            chan_out,
            reduction=4,
            width
    ):
        super().__init__()

        freq_w, freq_h = ([0] * 8), list(range(8))  # in paper, it seems 16 frequencies was ideal
        dct_weights = get_dct_weights(width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w])
        self.register_buffer('dct_weights', dct_weights)

        chan_intermediate = max(3, chan_out // reduction)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = reduce(x * self.dct_weights, 'b c (h h1) (w w1) -> b c h1 w1', 'sum', h1=1, w1=1)
        return self.net(x)


# generative adversarial network

class Generator(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            latent_dim=256,
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            attn_res_layers=[],
            freq_chan_attn=False
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            norm_class(latent_dim * 2),
            nn.GLU(dim=1)
        )

        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in=chan_out,
                        chan_out=sle_chan_out,
                        width=2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(
                        chan_in=chan_out,
                        chan_out=sle_chan_out
                    )

            layer = nn.ModuleList([
                nn.Sequential(
                    upsample(),
                    Blur(),
                    nn.Conv2d(chan_in, chan_out * 2, 3, padding=1),
                    norm_class(chan_out * 2),
                    nn.GLU(dim=1)
                ),
                sle,
                attn
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)
        x = F.normalize(x, dim=1)

        residuals = dict()

        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if exists(attn):
                x = attn(x) + x

            x = up(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x)


class SimpleDecoder(nn.Module):
    def __init__(
            self,
            *,
            chan_in,
            chan_out=3,
            num_upsamples=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(
                upsample(),
                nn.Conv2d(chans, chan_out, 3, padding=1),
                nn.GLU(dim=1)
            )
            self.layers.append(layer)
            chans //= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            disc_output_size=5,
            attn_res_layers=[]
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        assert disc_output_size in {1, 5}, 'discriminator output dimensions can only be 5x5 or 1x1'

        resolution = int(resolution)

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        non_residual_resolutions = range(min(8, resolution), 2, -1)
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chan_in_out = list(zip(features[:-1], features[1:]))

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                Blur(),
                nn.Conv2d(init_channel, chan_out, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1)
            ))

        self.residual_layers = nn.ModuleList([])

        for (res, ((_, chan_in), (_, chan_out))) in zip(non_residual_resolutions, chan_in_out):
            image_width = 2 ** res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            self.residual_layers.append(nn.ModuleList([
                SumBranches([
                    nn.Sequential(
                        Blur(),
                        nn.Conv2d(chan_in, chan_out, 4, stride=2, padding=1),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding=1),
                        nn.LeakyReLU(0.1)
                    ),
                    nn.Sequential(
                        Blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chan_in, chan_out, 1),
                        nn.LeakyReLU(0.1),
                    )
                ]),
                attn
            ]))

        last_chan = features[-1][-1]
        if disc_output_size == 5:
            self.to_logits = nn.Sequential(
                nn.Conv2d(last_chan, last_chan, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )
        elif disc_output_size == 1:
            self.to_logits = nn.Sequential(
                Blur(),
                nn.Conv2d(last_chan, last_chan, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )

        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding=1),
            Residual(PreNorm(64, LinearAttention(64))),
            SumBranches([
                nn.Sequential(
                    Blur(),
                    nn.Conv2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    Blur(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 32, 1),
                    nn.LeakyReLU(0.1),
                )
            ]),
            Residual(PreNorm(32, LinearAttention(32))),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 1, 4)
        )

        self.decoder1 = SimpleDecoder(chan_in=last_chan, chan_out=init_channel)
        self.decoder2 = SimpleDecoder(chan_in=features[-2][-1], chan_out=init_channel) if resolution >= 9 else None

    def forward(self, x, calc_aux_loss=False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (net, attn) in self.residual_layers:
            if exists(attn):
                x = attn(x) + x

            x = net(x)
            layer_outputs.append(x)

        out = self.to_logits(x).flatten(1)

        img_32x32 = F.interpolate(orig_img, size=(32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)

        if not calc_aux_loss:
            return out, out_32x32, None

        # self-supervised auto-encoding loss

        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)

        aux_loss = F.mse_loss(
            recon_img_8x8,
            F.interpolate(orig_img, size=recon_img_8x8.shape[2:])
        )

        if exists(self.decoder2):
            select_random_quadrant = lambda rand_quadrant, img: \
                rearrange(img, 'b c (m h) (n w) -> (m n) b c h w', m=2, n=2)[rand_quadrant]
            crop_image_fn = partial(select_random_quadrant, floor(random() * 4))
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

            recon_img_16x16 = self.decoder2(layer_16x16_part)

            aux_loss_16x16 = F.mse_loss(
                recon_img_16x16,
                F.interpolate(img_part, size=recon_img_16x16.shape[2:])
            )

            aux_loss = aux_loss + aux_loss_16x16

        return out, out_32x32, aux_loss


class LightweightGAN(nn.Module, HugGANModelHubMixin):
    def __init__(
            self,
            *,
            latent_dim,
            image_size,
            optimizer="adam",
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            disc_output_size=5,
            attn_res_layers=[],
            freq_chan_attn=False,
            ttur_mult=1.,
            lr=2e-4,
    ):
        super().__init__()

        self.config = {
            'latent_dim': latent_dim,
            'image_size': image_size,
            'optimizer': optimizer,
            'fmap_max': fmap_max,
            'fmap_inverse_coef': fmap_inverse_coef,
            'transparent': transparent,
            'greyscale': greyscale,
            'disc_output_size': disc_output_size,
            'attn_res_layers': attn_res_layers,
            'freq_chan_attn': freq_chan_attn,
            'ttur_mult': ttur_mult,
            'lr': lr
        }

        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size=image_size,
            latent_dim=latent_dim,
            fmap_max=fmap_max,
            fmap_inverse_coef=fmap_inverse_coef,
            transparent=transparent,
            greyscale=greyscale,
            attn_res_layers=attn_res_layers,
            freq_chan_attn=freq_chan_attn
        )

        self.G = Generator(**G_kwargs)

        self.D = Discriminator(
            image_size=image_size,
            fmap_max=fmap_max,
            fmap_inverse_coef=fmap_inverse_coef,
            transparent=transparent,
            greyscale=greyscale,
            attn_res_layers=attn_res_layers,
            disc_output_size=disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = Generator(**G_kwargs)
        set_requires_grad(self.GE, False)

        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr=lr * ttur_mult, betas=(0.5, 0.9))
        elif optimizer == "adabelief":
            from adabelief_pytorch import AdaBelief

            self.G_opt = AdaBelief(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
            self.D_opt = AdaBelief(self.D.parameters(), lr=lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented

    def _save_pretrained(self, save_directory):
        """
        Overwrite this method in case you don't want to save complete model,
        rather some specific layers
        """
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, "module") else self

        # We update this to be a dict containing 'GAN', as that's what is expected
        torch.save({'GAN': model_to_save.state_dict()}, path)

    @classmethod
    def _from_pretrained(
            cls,
            model_id,
            revision,
            cache_dir,
            force_download,
            proxies,
            resume_download,
            local_files_only,
            token,
            map_location="cpu",
            strict=False,
            **model_kwargs,
    ):
        """
        Overwrite this method in case you wish to initialize your model in a
        different way.
        """
        map_location = torch.device(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        # We update here to directly unpack config
        model = cls(**model_kwargs['config'])

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict["GAN"], strict=strict)
        model.eval()

        return model


# trainer

class Trainer():
    def __init__(
            self,
            dataset_name="huggan/CelebA-faces",
            name='default',
            results_dir='results',
            models_dir='models',
            base_dir='./',
            optimizer='adam',
            latent_dim=256,
            image_size=128,
            num_image_tiles=8,
            fmap_max=512,
            transparent=False,
            greyscale=False,
            batch_size=4,
            gp_weight=10,
            gradient_accumulate_every=1,
            attn_res_layers=[],
            freq_chan_attn=False,
            disc_output_size=5,
            dual_contrast_loss=False,
            antialias=False,
            lr=2e-4,
            lr_mlp=1.,
            ttur_mult=1.,
            save_every=10000,
            evaluate_every=1000,
            aug_prob=None,
            aug_types=['translation', 'cutout'],
            dataset_aug_prob=0.,
            calculate_fid_every=None,
            calculate_fid_num_images=12800,
            clear_fid_cache=False,
            log=False,
            cpu=False,
            mixed_precision="no",
            wandb=False,
            push_to_hub=False,
            organization_name=None,
            *args,
            **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.dataset_name = dataset_name

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name

        # Note - in original repo config is private - ".config.json", but here, we make it public
        self.config_path = self.models_dir / name / 'config.json'

        assert is_power_of_two(image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        assert all(map(is_power_of_two,
                       attn_res_layers)), 'resolution layers of attention must all be powers of 2 (16, 32, 64, 128, 256, 512)'

        assert not (
                dual_contrast_loss and disc_output_size > 1), 'discriminator output size cannot be greater than 1 if using dual contrastive loss'

        self.image_size = image_size
        self.num_image_tiles = num_image_tiles

        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.transparent = transparent
        self.greyscale = greyscale

        assert (int(self.transparent) + int(self.greyscale)) < 2, 'you can only set either transparency or greyscale'

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.optimizer = optimizer
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.gp_weight = gp_weight

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.attn_res_layers = attn_res_layers
        self.freq_chan_attn = freq_chan_attn

        self.disc_output_size = disc_output_size
        self.antialias = antialias

        self.dual_contrast_loss = dual_contrast_loss

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.syncbatchnorm = torch.cuda.device_count() > 1 and not cpu

        self.cpu = cpu
        self.mixed_precision = mixed_precision

        self.wandb = wandb

        self.push_to_hub = push_to_hub
        self.organization_name = organization_name
        self.repo_name = get_full_repo_name(self.name, self.organization_name)
        if self.push_to_hub:
            self.repo_url = create_repo(self.repo_name, exist_ok=True)

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    def init_GAN(self):
        args, kwargs = self.GAN_params

        # set some global variables before instantiating GAN

        global norm_class
        global Blur

        norm_class = nn.SyncBatchNorm if self.syncbatchnorm else nn.BatchNorm2d
        Blur = nn.Identity if not self.antialias else Fuzziness

        # instantiate GAN

        self.GAN = LightweightGAN(
            optimizer=self.optimizer,
            lr=self.lr,
            latent_dim=self.latent_dim,
            attn_res_layers=self.attn_res_layers,
            freq_chan_attn=self.freq_chan_attn,
            image_size=self.image_size,
            ttur_mult=self.ttur_mult,
            fmap_max=self.fmap_max,
            disc_output_size=self.disc_output_size,
            transparent=self.transparent,
            greyscale=self.greyscale,
            *args,
            **kwargs
        )

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        self.syncbatchnorm = config['syncbatchnorm']
        self.disc_output_size = config['disc_output_size']
        self.greyscale = config.pop('greyscale', False)
        self.attn_res_layers = config.pop('attn_res_layers', [])
        self.freq_chan_attn = config.pop('freq_chan_attn', False)
        self.optimizer = config.pop('optimizer', 'adam')
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
            'greyscale': self.greyscale,
            'syncbatchnorm': self.syncbatchnorm,
            'disc_output_size': self.disc_output_size,
            'optimizer': self.optimizer,
            'attn_res_layers': self.attn_res_layers,
            'freq_chan_attn': self.freq_chan_attn
        }

    def set_data_src(self):
        # start of using HuggingFace dataset
        dataset = load_dataset(self.dataset_name)

        if self.transparent:
            num_channels = 4
            pillow_mode = 'RGBA'
            expand_fn = expand_greyscale(self.transparent)
        elif self.greyscale:
            num_channels = 1
            pillow_mode = 'L'
            expand_fn = identity()
        else:
            num_channels = 3
            pillow_mode = 'RGB'
            expand_fn = expand_greyscale(self.transparent)

        convert_image_fn = partial(convert_image_to, pillow_mode)

        transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, self.image_size)),
            transforms.Resize(self.image_size),
            RandomApply(0., transforms.RandomResizedCrop(self.image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                        transforms.CenterCrop(self.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_fn)
        ])

        def transform_images(examples):
            transformed_images = [transform(image.convert("RGB")) for image in examples["image"]]

            examples["image"] = torch.stack(transformed_images)

            return examples

        transformed_dataset = dataset.with_transform(transform_images)

        per_device_batch_size = math.ceil(self.batch_size / self.accelerator.num_processes)
        dataloader = DataLoader(transformed_dataset["train"], per_device_batch_size, sampler=None, shuffle=False,
                                drop_last=True, pin_memory=True)
        num_samples = len(transformed_dataset)
        ## end of HuggingFace dataset

        # Note - in original repo, this is wrapped with cycle, but we will do that after accelerator prepares
        self.loader = dataloader

        # auto set augmentation prob for user if dataset is detected to be low
        # num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def init_accelerator(self):
        # Initialize the accelerator. We will let the accelerator handle device placement.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=self.mixed_precision, cpu=self.cpu)
        
        if self.accelerator.is_local_main_process:
            # set up Weights and Biases if requested
            if self.wandb:
                import wandb

                wandb.init(project=str(self.results_dir).split("/")[-1])

        if not exists(self.GAN):
            self.init_GAN()

        G = self.GAN.G
        D = self.GAN.D
        D_aug = self.GAN.D_aug

        # discriminator loss fn

        self.set_data_src()

        # prepare
        G, D, D_aug, self.GAN.D_opt, self.GAN.G_opt, self.loader = self.accelerator.prepare(G, D, D_aug, self.GAN.D_opt,
                                                                                            self.GAN.G_opt, self.loader)
        self.loader = cycle(self.loader)

        return G, D, D_aug

    def train(self, G, D, D_aug):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        self.GAN.train()
        total_disc_loss = torch.zeros([], device=self.accelerator.device)
        total_gen_loss = torch.zeros([], device=self.accelerator.device)

        batch_size = math.ceil(self.batch_size / self.accelerator.num_processes)

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        aug_prob = default(self.aug_prob, 0)
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0

        # discriminator loss fn

        if self.dual_contrast_loss:
            D_loss_fn = dual_contrastive_loss
        else:
            D_loss_fn = hinge_loss

        # train discriminator

        self.GAN.D_opt.zero_grad()
        for i in range(self.gradient_accumulate_every):
            latents = torch.randn(batch_size, latent_dim, device=self.accelerator.device)
            image_batch = next(self.loader)["image"]
            image_batch.requires_grad_()

            with torch.no_grad():
                generated_images = G(latents)

            fake_output, fake_output_32x32, _ = D_aug(generated_images, detach=True, **aug_kwargs)

            real_output, real_output_32x32, real_aux_loss = D_aug(image_batch, calc_aux_loss=True, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            divergence = D_loss_fn(real_output_loss, fake_output_loss)
            divergence_32x32 = D_loss_fn(real_output_32x32, fake_output_32x32)
            disc_loss = divergence + divergence_32x32

            aux_loss = real_aux_loss
            disc_loss = disc_loss + aux_loss

            if apply_gradient_penalty:
                outputs = [real_output, real_output_32x32]
                if self.accelerator.scaler is not None:
                    outputs = list(map(self.accelerator.scaler.scale, outputs))

                scaled_gradients = torch_grad(outputs=outputs, inputs=image_batch,
                                              grad_outputs=list(
                                                  map(lambda t: torch.ones(t.size(), device=self.accelerator.device),
                                                      outputs)),
                                              create_graph=True, retain_graph=True, only_inputs=True)[0]

                inv_scale = 1.
                if self.accelerator.scaler is not None:
                    inv_scale = safe_div(1., self.accelerator.scaler.get_scale())

                if inv_scale != float('inf'):
                    gradients = scaled_gradients * inv_scale

                    gradients = gradients.reshape(batch_size, -1)
                    gp = self.gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    if not torch.isnan(gp):
                        disc_loss = disc_loss + gp
                        self.last_gp_loss = gp.clone().detach().item()

            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            disc_loss = disc_loss / self.gradient_accumulate_every

            disc_loss.register_hook(raise_if_nan)
            self.accelerator.backward(disc_loss)
            total_disc_loss += divergence

        self.last_recon_loss = aux_loss.item()
        self.d_loss = float(total_disc_loss.item() / self.gradient_accumulate_every)
        self.GAN.D_opt.step()

        # generator loss fn

        if self.dual_contrast_loss:
            G_loss_fn = dual_contrastive_loss
            G_requires_calc_real = True
        else:
            G_loss_fn = gen_hinge_loss
            G_requires_calc_real = False

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in range(self.gradient_accumulate_every):
            latents = torch.randn(batch_size, latent_dim, device=self.accelerator.device)

            if G_requires_calc_real:
                image_batch = next(self.loader)["image"]
                image_batch.requires_grad_()

            generated_images = G(latents)

            fake_output, fake_output_32x32, _ = D_aug(generated_images, **aug_kwargs)
            real_output, real_output_32x32, _ = D_aug(image_batch, **aug_kwargs) if G_requires_calc_real else (
                None, None, None)

            loss = G_loss_fn(fake_output, real_output)
            loss_32x32 = G_loss_fn(fake_output_32x32, real_output_32x32)

            gen_loss = loss + loss_32x32

            gen_loss = gen_loss / self.gradient_accumulate_every

            gen_loss.register_hook(raise_if_nan)
            self.accelerator.backward(gen_loss)
            total_gen_loss += loss

            # divide loss by gradient accumulation steps since gradients
        # are accumulated for multiple backward passes in PyTorch
        self.g_loss = float(total_gen_loss.item() / self.gradient_accumulate_every)
        self.GAN.G_opt.step()

        # calculate moving averages
        if self.accelerator.is_main_process and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.accelerator.is_main_process and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        del total_disc_loss
        del total_gen_loss

        # periodically save results

        if self.accelerator.is_main_process:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

                if self.push_to_hub:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        self.GAN.push_to_hub(temp_dir, self.repo_url, config=self.GAN.config, skip_lfs_files=True)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 20000):
                self.evaluate(floor(self.steps / self.evaluate_every), num_image_tiles=self.num_image_tiles)

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1

    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=4):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents = torch.randn(num_rows ** 2, latent_dim, device=self.accelerator.device)

        # regular

        generated_images = self.generate_(self.GAN.G, latents)
        file_name = str(self.results_dir / self.name / f'{str(num)}.{ext}')
        save_image(generated_images, file_name, nrow=num_rows)

        # moving averages

        generated_images = self.generate_(self.GAN.GE.to(self.accelerator.device), latents)
        file_name_ema = str(self.results_dir / self.name / f'{str(num)}-ema.{ext}')
        save_image(generated_images, file_name_ema, nrow=num_rows)

        if self.accelerator.is_local_main_process and self.wandb:
            import wandb

            wandb.log({'generated_examples': wandb.Image(str(file_name))})
            wandb.log({'generated_examples_ema': wandb.Image(str(file_name_ema))})

    @torch.no_grad()
    def generate(self, num=0, num_image_tiles=4, checkpoint=None, types=['default', 'ema']):
        self.GAN.eval()

        latent_dim = self.GAN.latent_dim
        dir_name = self.name + str('-generated-') + str(checkpoint)
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension

        if not dir_full.exists():
            os.mkdir(dir_full)

        # regular
        if 'default' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated default images'):
                latents = torch.randn(1, latent_dim, device=self.accelerator.device)
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}.{ext}')
                save_image(generated_image[0], path, nrow=1)

        # moving averages
        if 'ema' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated EMA images'):
                latents = torch.randn(1, latent_dim, device=self.accelerator.device)
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-ema.{ext}')
                save_image(generated_image[0], path, nrow=1)

        return dir_full

    @torch.no_grad()
    def show_progress(self, num_images=4, types=['default', 'ema']):
        checkpoints = self.get_checkpoints()
        assert exists(checkpoints), 'cannot find any checkpoints to create a training progress video for'

        dir_name = self.name + str('-progress')
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension
        latents = None

        zfill_length = math.ceil(math.log10(len(checkpoints)))

        if not dir_full.exists():
            os.mkdir(dir_full)

        for checkpoint in tqdm(checkpoints, desc='Generating progress images'):
            self.load(checkpoint, print_version=False)
            self.GAN.eval()

            if checkpoint == 0:
                latents = torch.randn(num_images, self.GAN.latent_dim, self.accelerator.device)

            # regular
            if 'default' in types:
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}.{ext}')
                save_image(generated_image, path, nrow=num_images)

            # moving averages
            if 'ema' in types:
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}-ema.{ext}')
                save_image(generated_image, path, nrow=num_images)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories
        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)["image"]
                for k, image in enumerate(real_batch.unbind(0)):
                    ind = k + batch_num * self.batch_size
                    save_image(image, real_path / f'{ind}.png')

        # generate a bunch of fake images in results / name / fid_fake
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = torch.randn(self.batch_size, latent_dim, device=self.accelerator.device)

            # moving averages
            generated_images = self.generate_(self.GAN.GE, latents)

            for j, image in enumerate(generated_images.unbind(0)):
                ind = j + batch_num * self.batch_size
                save_image(image, str(fake_path / f'{str(ind)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, latents.device, 2048)

    @torch.no_grad()
    def generate_(self, G, style, num_image_tiles=8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, num_steps=100, save_frames=False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise
        latents_low = torch.randn(num_rows ** 2, latent_dim, device=self.accelerator.device)
        latents_high = torch.randn(num_rows ** 2, latent_dim, device=self.accelerator.device)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_(self.GAN.GE, interp_latents)
            images_grid = torchvision.utils.make_grid(generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            if self.transparent:
                background = Image.new('RGBA', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:],
                       duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

        if self.accelerator.is_local_main_process:
            log_dict = {v[0]: v[1] for v in data}
            if self.wandb:
                import wandb

                wandb.log(log_dict)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            checkpoints = self.get_checkpoints()

            if not exists(checkpoints):
                return

            name = checkpoints[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print(
                'unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e

    def get_checkpoints(self):
        file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
        saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))

        if len(saved_nums) == 0:
            return None

        return saved_nums