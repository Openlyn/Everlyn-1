import torch
import torch.nn.functional as F
from torch import nn


def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size,
        pad_mode = 'constant',
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode = 'replicate')
        return self.conv(x)

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)

class ResBlockX(nn.Module):
    def __init__(self, inchannel) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, inchannel),
            Swish(),
            CausalConv3d(inchannel, inchannel, 3),
            nn.GroupNorm(32, inchannel),
            Swish(),
            CausalConv3d(inchannel, inchannel, 3)
        )
    
    def forward(self, x):
        return x + self.conv(x)
    
class ResBlockXY(nn.Module):
    def __init__(self, inchannel, outchannel) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, inchannel),
            Swish(),
            CausalConv3d(inchannel, outchannel, 3),
            nn.GroupNorm(32, outchannel),
            Swish(),
            CausalConv3d(outchannel, outchannel, 3)
        )
        self.conv_1 = nn.Conv3d(inchannel, outchannel, 1)
    
    def forward(self, x):
        return self.conv_1(x) + self.conv(x)
    
class PoolDown222(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
    
    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 1, 0), 'replicate')
        return self.pool(x)
    
class PoolDown122(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))
    
    def forward(self, x):
        return self.pool(x)
    
class Unpool222(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.up(x)
        return x[:, :, 1:]
    
class Unpool122(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')

    def forward(self, x):
        x = self.up(x)
        return x

class ResBlockDown(nn.Module):
    def __init__(self, inchannel, outchannel) -> None:
        super().__init__()
        self.blcok = nn.Sequential(
            CausalConv3d(inchannel, outchannel, 3),
            nn.LeakyReLU(inplace=True),
            PoolDown222(),
            CausalConv3d(outchannel, outchannel, 3),
            nn.LeakyReLU(inplace=True)
        )
        self.res = nn.Sequential(
            PoolDown222(),
            nn.Conv3d(inchannel, outchannel, 1)
        )

    def forward(self, x):
        return self.res(x) + self.blcok(x)
    

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            CausalConv3d(3, 64, 3),
            nn.LeakyReLU(inplace=True),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 256),
            ResBlockDown(256, 256),
            ResBlockDown(256, 256),
            CausalConv3d(256, 256, 3),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        if x.ndim==4:
            x = x.unsqueeze(2)
        return self.block(x)



class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv3d(3, 64, 3),
            ResBlockX(64),
            ResBlockX(64),
            PoolDown222(),
            ResBlockXY(64, 128),
            ResBlockX(128),
            PoolDown222(),
            ResBlockX(128),
            ResBlockX(128),
            PoolDown122(),
            ResBlockXY(128, 256),
            ResBlockX(256),
            ResBlockX(256),
            ResBlockX(256),
            nn.GroupNorm(32, 256),
            Swish(),
            nn.Conv3d(256, 16, 1)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            CausalConv3d(8, 256, 3),
            ResBlockX(256),
            ResBlockX(256),
            ResBlockX(256),
            ResBlockX(256),
            Unpool122(),
            CausalConv3d(256, 256, 3),
            ResBlockXY(256, 128),
            ResBlockX(128),
            Unpool222(),
            CausalConv3d(128, 128, 3),
            ResBlockX(128),
            ResBlockX(128),
            Unpool222(),
            CausalConv3d(128, 128, 3),
            ResBlockXY(128, 64),
            ResBlockX(64),
            nn.GroupNorm(32, 64),
            Swish(),
            CausalConv3d(64, 64, 3)
        )
        self.conv_out = nn.Conv3d(64, 3, 1)
    
    def forward(self, x):
        return self.conv_out(self.decoder(x))
    

if __name__=='__main__':
    encoder = Encoder()
    decoder = Decoder()
    dis = Discriminator()
    x = torch.randn((1, 3, 1, 64, 64))
    embedding = encoder(x)
    y = decoder(embedding)
    tmp = torch.randn((1, 4, 1, 64, 64))
    print('something mmm')