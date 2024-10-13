from inspect import trace
import math
import argparse
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import shift_dim, adopt_weight
from .modules import LPIPS, Codebook, ApplyNoise, Blur2d
from .modules.diffaug import DiffAugment

def silu(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class VQGAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes

        if not hasattr(args, 'padding_type'):
            args.padding_type = 'replicate'
        self.encoder = Encoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type, args.padding_type)
        self.decoder = Decoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, args.embedding_dim, 1, padding_type=args.padding_type)
        self.post_vq_conv = SamePadConv3d(args.embedding_dim, self.enc_out_ch, 1)
        
        self.codebook = Codebook(args.n_codes, args.embedding_dim, no_random_restart=args.no_random_restart, restart_thres=args.restart_thres)

        self.apply_allframes = False # args.apply_allframes
        self.gan_feat_weight = args.gan_feat_weight
        self.image_discriminator = NLayerDiscriminator(args.image_channels, args.disc_channels, args.disc_layers)
        self.video_discriminator = NLayerDiscriminator3D(args.image_channels, args.disc_channels, args.disc_layers)
        
        if args.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif args.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = args.image_gan_weight
        self.video_gan_weight = args.video_gan_weight

        self.perceptual_weight = args.perceptual_weight

        self.l1_weight = args.l1_weight
        self.save_hyperparameters()

    @property
    def latent_shape(self):
        input_shape = (self.args.sequence_length//self.args.sample_every_n_frames, self.args.resolution,
                       self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape,
                                             self.args.downsample)])

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output['embeddings'], vq_output['encodings']
        else:
            return vq_output['encodings']

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, optimizer_idx=None, log_image=False):
        B, C, T, H, W = x.shape
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)
        
        all_frames = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, H, W)
        all_frames_recon = x_recon.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, H, W)

        if log_image:
            return frames, frames_recon, x, x_recon, vq_output["batch_usage"]

        if optimizer_idx == 0:
            # autoencoder
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                if self.apply_allframes:
                    perceptual_loss = self.perceptual_model(all_frames, all_frames_recon).mean() * self.perceptual_weight
                    logits_image_fake, pred_image_fake = self.image_discriminator(all_frames_recon)
                else:
                    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
                    logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
            
            logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
            
            disc_factor = adopt_weight(self.global_step, threshold=self.args.discriminator_iter_start)
            aeloss = disc_factor * g_loss


            # gan feature matching loss
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                if self.apply_allframes:
                    logits_image_real, pred_image_real = self.image_discriminator(all_frames)
                else:
                    logits_image_real, pred_image_real = self.image_discriminator(frames)
                for i in range(len(pred_image_fake)-1):
                    image_gan_feat_loss += feat_weights * F.l1_loss(pred_image_fake[i], pred_image_real[i].detach()) * (self.image_gan_weight>0)
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(x)
                for i in range(len(pred_video_fake)-1):
                    video_gan_feat_loss += feat_weights * F.l1_loss(pred_video_fake[i], pred_video_real[i].detach()) * (self.video_gan_weight>0)

            gan_feat_loss = disc_factor * self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)

            self.log("train/g_image_loss", g_image_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/g_video_loss", g_video_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/image_gan_feat_loss", image_gan_feat_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/video_gan_feat_loss", video_gan_feat_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/commitment_loss", vq_output['commitment_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log('train/perplexity', vq_output['perplexity'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

        if optimizer_idx == 1:
            # discriminator
            if self.apply_allframes:
                logits_image_real, _ = self.image_discriminator(all_frames.detach())
                logits_image_fake, _ = self.image_discriminator(all_frames_recon.detach())
            else:
                logits_image_real, _ = self.image_discriminator(frames.detach())
                logits_image_fake, _ = self.image_discriminator(frames_recon.detach())

            logits_video_real, _ = self.video_discriminator(x.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            disc_factor = adopt_weight(self.global_step, threshold=self.args.discriminator_iter_start)
            discloss = disc_factor * (self.image_gan_weight*d_image_loss + self.video_gan_weight*d_video_loss)

            self.log("train/logits_image_real", logits_image_real.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_image_fake", logits_image_fake.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_real", logits_video_real.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_fake", logits_video_fake.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/d_image_loss", d_image_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/d_video_loss", d_video_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return discloss

        if self.apply_allframes:
            perceptual_loss = self.perceptual_model(all_frames, all_frames_recon) * self.perceptual_weight
        else:
            perceptual_loss = self.perceptual_model(frames, frames_recon) * self.perceptual_weight
        return recon_loss, x_recon, vq_output, perceptual_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['video']
        if optimizer_idx == 0:
            recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(x, optimizer_idx)
            commitment_loss = vq_output['commitment_loss']
            loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        if optimizer_idx == 1:
            discloss = self.forward(x, optimizer_idx)
            loss = discloss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['video'] # TODO: batch['stft']
        recon_loss, _, vq_output, perceptual_loss = self.forward(x)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)

    def configure_optimizers(self):

        lr = self.args.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.pre_vq_conv.parameters())+
                                  list(self.post_vq_conv.parameters())+
                                  list(self.codebook.parameters()),
                                  lr=self.args.lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters())+
                                    list(self.video_discriminator.parameters()),
                                    lr=self.args.lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['video']
        x = x.to(self.device)
        frames, frames_rec, _, _, _ = self(x, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch['video']
        _, _, x, x_rec, _ = self(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--n_codes', type=int, default=2048)
        parser.add_argument('--n_hiddens', type=int, default=240)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        parser.add_argument('--disc_channels', type=int, default=64)
        parser.add_argument('--disc_layers', type=int, default=3)
        parser.add_argument('--discriminator_iter_start', type=int, default=50000)
        parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])
        parser.add_argument('--apply_allframes', action="store_true")
        parser.add_argument('--image_gan_weight', type=float, default=1.0)
        parser.add_argument('--video_gan_weight', type=float, default=1.0)
        parser.add_argument('--l1_weight', type=float, default=4.0)
        parser.add_argument('--gan_feat_weight', type=float, default=0.0)
        parser.add_argument('--perceptual_weight', type=float, default=0.0)
        parser.add_argument('--i3d_feat', action='store_true')
        parser.add_argument('--restart_thres', type=float, default=1.0)
        parser.add_argument('--no_random_restart', action='store_true')
        parser.add_argument('--norm_type', type=str, default='group', choices=['batch', 'group'])
        parser.add_argument('--padding_type', type=str, default='replicate', choices=['replicate', 'constant', 'reflect', 'circular'])

        return parser


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate'):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type), 
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group'):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        
        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i ==0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(out_channels, image_channel, kernel_size=3)
    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))

        
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type="batch", use_sigmoid=False, getIntermFeat=True, activation="leaky_relu", apply_blur=False, apply_noise=False):
    # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        if apply_noise:
            self.noise = ApplyNoise(channels=input_nc)
        else:
            self.noise = nn.Identity()
        

        activation_func = nn.LeakyReLU(0.2, True) if activation == "leaky_relu" else nn.Tanh() 
        
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), activation_func]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                Blur2d(f=None) if apply_blur else nn.Identity(),
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                Normalize(nf, norm_type), 
                activation_func
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            Normalize(nf, norm_type),
            activation_func
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input, apply_diffaug=False):
        if self.getIntermFeat:
            trans_input = self.noise(input)
            if apply_diffaug:
                trans_input = DiffAugment(trans_input, policy='color,translation,cutout')
            
            res = [trans_input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _




class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type="batch", use_sigmoid=False, getIntermFeat=True, activation="leaky_relu", apply_blur=False, apply_noise=False):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        if apply_noise:
            self.noise = ApplyNoise(channels=input_nc)
        else:
            self.noise = nn.Identity()

        activation_func = nn.LeakyReLU(0.2, True) if activation == "leaky_relu" else nn.Tanh() 

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), activation_func]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                Blur2d(f=None) if apply_blur else nn.Identity(),
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                Normalize(nf, norm_type), 
                activation_func
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            Normalize(nf, norm_type),
            activation_func
        ]]

        sequence += [[
            nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw),
            Normalize(1, norm_type),
            activation_func
        ]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input, apply_diffaug=False):
        if self.getIntermFeat:
            trans_input = self.noise(input)
            if apply_diffaug:
                B = input.shape[0]
                trans_input = rearrange(trans_input, "b c t h w -> (b t) c h w") # B C T H W
                trans_input = DiffAugment(trans_input, policy='color,translation,cutout')
                trans_input = rearrange(trans_input, "(b t) c h w -> b c t h w", b=B)
            
            res = [trans_input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
                    
            return res[-1], res[1:]
        else:
            return self.model(input), _

