import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if x.ndim == 4:
            # image: B C H W
            if noise is None:
                noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)
        
        else:
            # video: B C T H W
            if noise is None:
                noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), x.size(4), device=x.device, dtype=x.dtype)

            return x + self.weight.view(1, -1, 1, 1, 1) * noise.to(x.device)

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                # f = f[:, :, ::-1, ::-1]
                f = torch.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x


class Conv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        super(LayerEpilogue, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):
        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)

        return x


class GBlock(nn.Module):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,        # noise
                 dlatent_size=512,   # Disentangled latent (W) dimensionality.
                 use_style=True,     # Enable style inputs?
                 f=None,        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_base=8192,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,       # Maximum number of feature maps in any layer.
                 ):
        super(GBlock, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        # res
        self.res = res

        # blur2d
        self.blur = Blur2d(f)

        # noise
        self.noise_input = noise_input

        if res < 7:
            # upsample method 1
            self.up_sample = Upscale2d(factor)
        else:
            # upsample method 2
            self.up_sample = nn.ConvTranspose2d(self.nf(res-3), self.nf(res-2), 4, stride=2, padding=1)

        # A Composition of LayerEpilogue and Conv2d.
        self.adaIn1 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=self.nf(res-2), output_channels=self.nf(res-2),
                             kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)

    def forward(self, x, dlatent):
        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])
        return x

#model.apply(weights_init)


# =========================================================================
#   Define sub-network
#   2019.3.31
#   FC
# =========================================================================
class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=1024,
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                 gain=2**(0.5)            # original gain in tensorflow.
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size,                       # Disentangled latent (W) dimensionality.
                 resolution=1024,                    # Output resolution (1024 x 1024 by default).
                 fmap_base=8192,                     # Overall multiplier for the number of feature maps.
                 num_channels=3,                     # Number of output color channels.
                 structure='fixed',                  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 fmap_max=512,                       # Maximum number of feature maps in any layer.
                 fmap_decay=1.0,                     # log2 feature map reduction when doubling the resolution.
                 f=None,                        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
                 use_instance_norm   = True,        # Enable instance normalization?
                 use_wscale = True,                  # Enable equalized learning rate?
                 use_noise = True,                   # Enable noise inputs?
                 use_style = True                    # Enable style inputs?
                 ):                             # batch size.
        """
            2019.3.31
        :param dlatent_size: 512 Disentangled latent(W) dimensionality.
        :param resolution: 1024 x 1024.
        :param fmap_base:
        :param num_channels:
        :param structure: only support 'fixed' mode.
        :param fmap_max:
        """
        super(G_synthesis, self).__init__()

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.structure = structure
        self.resolution_log2 = int(np.log2(resolution))
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        num_layers = self.resolution_log2 * 2 - 2
        self.num_layers = num_layers

        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to("cuda"))

        # Blur2d
        self.blur = Blur2d(f)

        # torgb: fixed mode
        self.channel_shrinkage = Conv2d(input_channels=self.nf(self.resolution_log2-2),
                                        output_channels=self.nf(self.resolution_log2),
                                        kernel_size=3,
                                        use_wscale=use_wscale)
        self.torgb = Conv2d(self.nf(self.resolution_log2), num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        # Initial Input Block
        self.const_input = nn.Parameter(torch.ones(1, self.nf(1), 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf(1)))
        self.adaIn1 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=self.nf(1), output_channels=self.nf(1), kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)

        # Common Block
        # 4 x 4 -> 8 x 8
        res = 3
        self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 8 x 8 -> 16 x 16
        res = 4
        self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 16 x 16 -> 32 x 32
        res = 5
        self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 32 x 32 -> 64 x 64
        res = 6
        self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 64 x 64 -> 128 x 128
        res = 7
        self.GBlock5 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 128 x 128 -> 256 x 256
        res = 8
        self.GBlock6 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 256 x 256 -> 512 x 512
        res = 9
        self.GBlock7 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 512 x 512 -> 1024 x 1024
        res = 10
        self.GBlock8 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

    def forward(self, dlatent):
        """
           dlatent: Disentangled latents (W), shape为[minibatch, num_layers, dlatent_size].
        :param dlatent:
        :return:
        """
        images_out = None
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if self.structure == 'fixed':
            # initial block 0:
            x = self.const_input.expand(dlatent.size(0), -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])
            x = self.conv1(x)
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])

            # block 1:
            # 4 x 4 -> 8 x 8
            x = self.GBlock1(x, dlatent)

            # block 2:
            # 8 x 8 -> 16 x 16
            x = self.GBlock2(x, dlatent)

            # block 3:
            # 16 x 16 -> 32 x 32
            x = self.GBlock3(x, dlatent)

            # block 4:
            # 32 x 32 -> 64 x 64
            x = self.GBlock4(x, dlatent)

            # block 5:
            # 64 x 64 -> 128 x 128
            x = self.GBlock5(x, dlatent)

            # block 6:
            # 128 x 128 -> 256 x 256
            x = self.GBlock6(x, dlatent)

            # block 7:
            # 256 x 256 -> 512 x 512
            x = self.GBlock7(x, dlatent)

            # block 8:
            # 512 x 512 -> 1024 x 1024
            x = self.GBlock8(x, dlatent)

            x = self.channel_shrinkage(x)
            images_out = self.torgb(x)
            return images_out


class StyleGenerator(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
                 truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=8,          # Number of layers for which to apply the truncation trick. None = disable.
                 **kwargs
                 ):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, latents1):
        dlatents1, num_layers = self.mapping(latents1)
        # let [N, O] -> [N, num_layers, O]
        # 这里的unsqueeze不能使用inplace操作, 如果这样的话, 反向传播的链条会断掉的.
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)

        # Add mixing style mechanism.
        # with torch.no_grad():
        #     latents2 = torch.randn(latents1.shape).to(latents1.device)
        #     dlatents2, num_layers = self.mapping(latents2)
        #     dlatents2 = dlatents2.unsqueeze(1)
        #     dlatents2 = dlatents2.expand(-1, int(num_layers), -1)
        #
        #     # TODO: original NvLABs produce a placeholder "lod", this mechanism was not added here.
        #     cur_layers = num_layers
        #     mix_layers = num_layers
        #     if np.random.random() < self.style_mixing_prob:
        #         mix_layers = np.random.randint(1, cur_layers)
        #
        #     # NvLABs: dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
        #     for i in range(num_layers):
        #         if i >= mix_layers:
        #             dlatents1[:, i, :] = dlatents2[:, i, :]

        # Apply truncation trick.
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            """Linear interpolation.
               a + (b - a) * t (a = 0)
               reduce to
               b * t
            """

            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)

        img = self.synthesis(dlatents1)
        return img


class StyleDiscriminator(nn.Module):
    def __init__(self,
                 resolution=1024,
                 fmap_base=8192,
                 num_channels=3,
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, only support 'fixed' mode now.
                 fmap_max=512,
                 fmap_decay=1.0,
                 # f=[1, 2, 1]         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 f=None         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        """
            Noitce: we only support input pic with height == width.

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        # fromrgb: fixed mode
        self.fromrgb = nn.Conv2d(num_channels, self.nf(self.resolution_log2-1), kernel_size=1)
        self.structure = structure

        # blur2d
        self.blur2d = Blur2d(f)

        # down_sample
        self.down1 = nn.AvgPool2d(2)
        self.down21 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-5), kernel_size=2, stride=2)
        self.down22 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-6), kernel_size=2, stride=2)
        self.down23 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-7), kernel_size=2, stride=2)
        self.down24 = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(self.resolution_log2-8), kernel_size=2, stride=2)

        # conv1: padding=same
        self.conv1 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-1), kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-2), kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2-3), kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(self.nf(self.resolution_log2-3), self.nf(self.resolution_log2-4), kernel_size=3, padding=(1, 1))
        self.conv5 = nn.Conv2d(self.nf(self.resolution_log2-4), self.nf(self.resolution_log2-5), kernel_size=3, padding=(1, 1))
        self.conv6 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-6), kernel_size=3, padding=(1, 1))
        self.conv7 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-7), kernel_size=3, padding=(1, 1))
        self.conv8 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-8), kernel_size=3, padding=(1, 1))

        # calculate point:
        self.conv_last = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(1), kernel_size=3, padding=(1, 1))
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.structure == 'fixed':
            x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
            # 1. 1024 x 1024 x nf(9)(16) -> 512 x 512
            res = self.resolution_log2
            x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 2. 512 x 512 -> 256 x 256
            res -= 1
            x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 3. 256 x 256 -> 128 x 128
            res -= 1
            x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 4. 128 x 128 -> 64 x 64
            res -= 1
            x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 5. 64 x 64 -> 32 x 32
            res -= 1
            x = F.leaky_relu(self.conv5(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down21(self.blur2d(x)), 0.2, inplace=True)

            # 6. 32 x 32 -> 16 x 16
            res -= 1
            x = F.leaky_relu(self.conv6(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down22(self.blur2d(x)), 0.2, inplace=True)

            # 7. 16 x 16 -> 8 x 8
            res -= 1
            x = F.leaky_relu(self.conv7(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down23(self.blur2d(x)), 0.2, inplace=True)

            # 8. 8 x 8 -> 4 x 4
            res -= 1
            x = F.leaky_relu(self.conv8(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down24(self.blur2d(x)), 0.2, inplace=True)

            # 9. 4 x 4 -> point
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
            # N x 8192(4 x 4 x nf(1)).
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            # N x 1
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            return x
