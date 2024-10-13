import math
import argparse
import random
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from .quantizer.vector_quantize_pytorch import VectorQuantize
from .utils import shift_dim, adopt_weight
from .modules import LPIPS, Codebook
from .modules.attention import Transformer
from .base import Normalize, NLayerDiscriminator, NLayerDiscriminator3D

from OmniTokenizer.modules.vae import DiagonalGaussianDistribution


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors if x is not None]
        # ctx.input_tensors = [x.detach().requires_grad_(True) if x is not None else None for x in ctx.input_tensors]
        # Ensure all tensors have requires_grad set to True
        ctx.input_params = [p.requires_grad_(True) for p in ctx.input_params]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            # shallow_copies = [x.view_as(x) if x is not None else None for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def logits_laplace(x, x_recons, logit_laplace_eps=0.1):
    # [-0.5, 0.5] -> [0, 1]
    x += 0.5
    x_recons += 0.5
    # [0, 1] -> [eps, 1-eps]
    x_laplace = (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps
    x_recons_laplace = (1 - 2 * logit_laplace_eps) * x_recons + logit_laplace_eps
    return F.l1_loss(x_laplace, x_recons_laplace)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

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
        
        if not hasattr(args, 'enc_block'):
            args.enc_block = 't' * args.spatial_depth
        
        if not hasattr(args, 'dec_block'):
            args.dec_block = 't' * args.spatial_depth
        
        if not hasattr(args, 'twod_window_size'):
            args.twod_window_size = 4

        if not hasattr(args, 'defer_temporal_pool'):
            args.defer_temporal_pool = False
            args.defer_spatial_pool = False

        if not hasattr(args, 'q_strides'):
            args.q_strides = '1' * args.spatial_depth
        
        if not hasattr(args, "spatial_pos"):
            args.spatial_pos = "rel"

        if not hasattr(args, "logitslaplace_weight"):
            args.logitslaplace_weight = 0.

        self.use_checkpoint = args.use_checkpoint
        self.logitslaplace_weight = args.logitslaplace_weight

        if not hasattr(args, "gen_upscale"):
            args.gen_upscale = None

        self.gen_upscale = args.gen_upscale

        if not hasattr(args, "initialize_vit"):
            args.initialize_vit = False
        


        self.resolution = args.resolution
        self.patch_size = args.patch_size
        
        self.encoder = OmniTokenizer_Encoder(
            image_size = args.resolution, image_channel=args.image_channels, norm_type=args.norm_type, 
            block=args.enc_block, window_size=args.twod_window_size, q_strides=args.q_strides, spatial_pos=args.spatial_pos,
            patch_embed = args.patch_embed, patch_size = args.patch_size, temporal_patch_size= args.temporal_patch_size, defer_temporal_pool=args.defer_temporal_pool, defer_spatial_pool=args.defer_spatial_pool,
            spatial_depth=args.spatial_depth, temporal_depth=args.temporal_depth, casual_in_temporal_transformer=args.casual_in_temporal_transformer, casual_in_peg=args.casual_in_peg, 
            dim = args.embedding_dim, dim_head=args.dim_head, heads=args.heads, attn_dropout=args.attn_dropout, ff_dropout=args.ff_dropout, ff_mult=args.ff_mult,
            initialize=args.initialize_vit, sequence_length=args.sequence_length,
        )
        
        self.decoder = OmniTokenizer_Decoder(
            image_size = args.resolution, image_channel=args.image_channels, norm_type=args.norm_type, block=args.dec_block, window_size=args.twod_window_size, spatial_pos=args.spatial_pos,
            patch_embed = args.patch_embed, patch_size = args.patch_size, temporal_patch_size= args.temporal_patch_size, defer_temporal_pool=args.defer_temporal_pool, defer_spatial_pool=args.defer_spatial_pool,
            spatial_depth=len(args.dec_block), temporal_depth=args.temporal_depth, casual_in_temporal_transformer=args.casual_in_temporal_transformer, casual_in_peg=args.casual_in_peg, 
            dim = args.embedding_dim, dim_head=args.dim_head, heads=args.heads, attn_dropout=args.attn_dropout, ff_dropout=args.ff_dropout, ff_mult=args.ff_mult, gen_upscale=args.gen_upscale, 
            initialize=args.initialize_vit, sequence_length=args.sequence_length,
        )

        if not hasattr(args, "use_vae"):
            args.use_vae = False
        
        if not hasattr(args, "kl_weight"):
            args.kl_weight = 0.000001
        

        self.use_vae = args.use_vae
        self.kl_weight = args.kl_weight

        if args.use_external_codebook:
            if args.codebook_type == 'vq':
                self.codebook = VectorQuantize(
                    accept_image_fmap=True, dim=args.embedding_dim, codebook_size=args.n_codes, use_cosine_sim=args.l2_code, commitment_weight=args.commitment_weight, codebook_dim=args.codebook_dim
                )
                self.pre_vq_conv = nn.Identity()
                self.post_vq_conv = nn.Identity()

            else:
                raise NotImplementedError
        else:
            self.codebook = Codebook(args.n_codes, args.codebook_dim, no_random_restart=args.no_random_restart, restart_thres=args.restart_thres)
            if not self.use_vae:
                self.pre_vq_conv = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(args.embedding_dim, args.codebook_dim),
                    Rearrange("b t h w c -> b c t h w")
                )
            else:
                self.pre_vq_conv = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(args.embedding_dim, args.codebook_dim * 2),
                    Rearrange("b t h w c -> b c t h w")
                )

            self.post_vq_conv = nn.Sequential(
                Rearrange("b c t h w -> b t h w c"),
                nn.Linear(args.codebook_dim, args.embedding_dim),
                Rearrange("b t h w c -> b c t h w")
            )

        self.use_external_codebook = args.use_external_codebook
        self.l2_code = args.l2_code
        self.apply_allframes = args.apply_allframes

        self.gan_feat_weight = args.gan_feat_weight

        if not hasattr(args, "apply_diffaug"):
            args.apply_diffaug = False

        if not hasattr(args, "apply_noise"):
            args.apply_noise = False
        
        if not hasattr(args, "apply_blur"):
            args.apply_blur = False
        
        # input_nc, ndf=64, n_layers=3, norm_type="batch", use_sigmoid=False, getIntermFeat=True, activation="leaky_relu", apply_blur=False, apply_noise=False
        if not hasattr(args, "sigmoid_in_disc"):
            args.sigmoid_in_disc = False
        
        if not hasattr(args, "activation_in_disc"):
            args.activation_in_disc = "leaky_relu"
        
        self.image_discriminator = NLayerDiscriminator(args.image_channels, args.disc_channels, args.disc_layers, args.norm_type, use_sigmoid=args.sigmoid_in_disc, activation=args.activation_in_disc, apply_blur=args.apply_blur, apply_noise=args.apply_noise)
        self.video_discriminator = NLayerDiscriminator3D(args.image_channels, args.disc_channels, args.disc_layers, args.norm_type, use_sigmoid=args.sigmoid_in_disc, activation=args.activation_in_disc, apply_blur=args.apply_blur, apply_noise=args.apply_noise)
        

        if args.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif args.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()
        # self.video_perceptual_model = load_i3d_perceptual().eval()

        self.image_gan_weight = args.image_gan_weight
        self.video_gan_weight = args.video_gan_weight

        self.perceptual_weight = args.perceptual_weight

        if not hasattr(args, "video_perceptual_weight"):
            args.video_perceptual_weight = 0.

        self.video_perceptual_weight = args.video_perceptual_weight

        self.recon_loss_type = args.recon_loss_type
        self.l1_weight = args.l1_weight
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.grad_accumulates = args.grad_accumulates
        self.grad_clip_val = args.grad_clip_val

        if not hasattr(args, "grad_clip_val_disc"):
            args.grad_clip_val_disc = 1.0
        
        self.grad_clip_val_disc = args.grad_clip_val_disc
        
        if not hasattr(args, "disloss_check_thres"):
            args.disloss_check_thres = None
        
        if not hasattr(args, "perloss_check_thres"):
            args.perloss_check_thres = None
        
        if not hasattr(args, "recloss_check_thres"):
            args.recloss_check_thres = None

        self.apply_diffaug = args.apply_diffaug
        
        self.disloss_check_thres = args.disloss_check_thres
        self.perloss_check_thres = args.perloss_check_thres
        self.recloss_check_thres = args.recloss_check_thres


        if not hasattr(args, "resolution_scale"):
            args.resolution_scale = None
        
        self.resolution_scale = args.resolution_scale
        
    @property
    def latent_shape(self):
        input_shape = (self.args.sequence_length//self.args.sample_every_n_frames, self.args.resolution,
                       self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape,
                                             self.args.downsample)])

    def encode(self, x, is_image, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x, is_image))

        if not self.use_vae:
            if self.l2_code and not self.use_external_codebook:
                h = F.normalize(h, p=2, dim=1)
            
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        
        else:
            posterior = DiagonalGaussianDistribution(h)
            z = posterior.sample()
            if is_image:
                return z.squeeze(2) # b c t h w -> b c h w
            else:
                return z
            
    def features_to_idx_latents(self, h, include_embeddings=True):
        '''
        h: b c t h w
        '''
        if not self.use_vae:
            if self.l2_code and not self.use_external_codebook:
                h = F.normalize(h, p=2, dim=1)
            
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        else:
            raise NotImplementedError
    
    def decode(self, encodings, is_image):
        if not self.use_vae:
            z = F.embedding(encodings, self.codebook.embeddings)
            if is_image:
                if z.ndim == 3:
                    hw = z.shape[1]
                    h = int(math.sqrt(hw))
                    z = rearrange(z, "b (h w) c -> b c 1 h w", h=h)
                
                else:
                    z = rearrange(z, "b t h w c -> b c t h w")
                
                z = self.post_vq_conv(z)
            
            else:
                if z.ndim == 3:
                    h = self.resolution // self.patch_size
                    w = h
                    z = rearrange(z, "b (t h w) c -> b c t h w", h=h, w=w)
                else:
                    z = rearrange(z, "b t h w c -> b c t h w")
                
                z = self.post_vq_conv(z)
            
            return self.decoder(z, is_image)
        else:
            z = encodings

            if is_image:
                if z.ndim == 3:
                    hw = z.shape[1]
                    h = int(math.sqrt(hw))
                    z = rearrange(z, "b (h w) c -> b c 1 h w", h=h)
                
                else:
                    z = rearrange(z, "b c h w -> b c 1 h w")
                
                z = self.post_vq_conv(z)
            
            else:
                if z.ndim == 3:
                    h = self.resolution // self.patch_size
                    w = h
                    z = rearrange(z, "b (t h w) c -> b c t h w", h=h, w=w)
                else:
                    z = rearrange(z, "b t h w c -> b c t h w")
                
                z = self.post_vq_conv(z)
            
            return self.decoder(z, is_image)
        
        
    
    def prepare_video_4_log(self, video_recons):
        _, C, _, H, _ = video_recons.shape
        video_recons = video_recons[0].detach().cpu() # C, T, H, W
        video_recons = video_recons.permute(2, 1, 3, 0).contiguous().view(H, -1, C).numpy()
        video_recons = video_recons * 0.5 + 0.5
        video_pil = Image.fromarray(np.clip(video_recons * 255.0, 0, 255.0).astype('uint8'))
        return video_pil


    def forward(self, x, optimizer_idx=None, log_image=False):
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
            if self.resolution_scale is not None:
                x = rearrange(x, "b c t h w -> (b t) c h w")
                target_resolution_scale = random.choices(self.resolution_scale)[0]
                target_resolution = int(H * target_resolution_scale)
                x = F.interpolate(
                    x, size=(target_resolution, target_resolution), mode="bilinear", align_corners=True
                )

                x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
                H = W = target_resolution

        else:
            B, C, H, W = x.shape
            T = 1

            if self.resolution_scale is not None:
                target_resolution_scale = random.choices(self.resolution_scale)[0]
                target_resolution = int(H * target_resolution_scale)
                x = F.interpolate(
                    x, size=(target_resolution, target_resolution), mode="bilinear", align_corners=True
                )
                H = W = target_resolution
        
        z = self.pre_vq_conv(self.encoder(x, is_image, use_checkpoint=self.use_checkpoint))
        
        if not self.use_vae:
            if self.l2_code and not self.use_external_codebook:
                z = F.normalize(z, p=2, dim=1)

            vq_output = self.codebook(z)
            x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']), is_image, use_checkpoint=self.use_checkpoint)
        
        else:
            posterior = DiagonalGaussianDistribution(z)
            z = posterior.sample()
            z = self.post_vq_conv(z)
            x_recon = self.decoder(z, is_image)

            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] * self.kl_weight
            
            vq_output = {
                "commitment_loss": kl_loss
            }

        if x.shape != x_recon.shape:
            assert self.gen_upscale is not None
            if is_image:
                x = F.interpolate(x, scale_factor=self.gen_upscale, mode="bilinear", align_corners=True)
            else:
                x = rearrange(x, "b c t h w -> (b t) c h w")
                x = F.interpolate(x, scale_factor=self.gen_upscale, mode="bilinear", align_corners=True)
                x = rearrange(x, "(b t) c h w -> b c t h w", frames.shape[0])

        if self.recon_loss_type == 'l1':
            recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        else:
            recon_loss = F.mse_loss(x_recon, x) * self.l1_weight

        if self.recon_loss_type != 'l1':
            recon_loss += logits_laplace(x, x_recon) * self.logitslaplace_weight

        if is_image: # handle the cases with 4 dims
            frames = x
            frames_recon = x_recon
        
        else:
            frame_idx = torch.randint(0, T, [B]).cuda()
            frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
            frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
            frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)
        
            all_frames = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, H, W)
            all_frames_recon = x_recon.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, H, W)

        if log_image:
            if self.use_vae:
                return frames, frames_recon, x, x_recon, None
            else:
                return frames, frames_recon, x, x_recon, vq_output


        if optimizer_idx == 0:
            # autoencoder
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                if self.apply_allframes:
                    perceptual_loss = self.perceptual_model(all_frames, all_frames_recon).mean() * self.perceptual_weight
                    logits_image_fake, pred_image_fake = self.image_discriminator(all_frames_recon, False)
                else:
                    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
                    logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon, False)
            

            perceptual_video_loss = 0
            # if self.video_perceptual_weight > 0 and T > 1:
            #     perceptual_video_loss = self.video_perceptual_model.extract_perceptual(x, x_recon).mean() * self.video_perceptual_weight

            g_image_loss = -torch.mean(logits_image_fake)
            
            if T > 1:
                logits_video_fake, pred_video_fake = self.video_discriminator(x_recon, False)
                g_video_loss = -torch.mean(logits_video_fake)
            else:
                logits_video_fake, pred_video_fake = None, None
                g_video_loss = 0.
            
            g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
            
            disc_factor = adopt_weight(self.global_step, threshold=self.args.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            # gan feature matching loss
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                if self.apply_allframes:
                    logits_image_real, pred_image_real = self.image_discriminator(all_frames, False)
                else:
                    logits_image_real, pred_image_real = self.image_discriminator(frames, False)
                for i in range(len(pred_image_fake)-1):
                    image_gan_feat_loss += feat_weights * F.l1_loss(pred_image_fake[i], pred_image_real[i].detach()) * (self.image_gan_weight>0)
            
            if self.video_gan_weight > 0 and T > 1:
                logits_video_real, pred_video_real = self.video_discriminator(x, False)
                for i in range(len(pred_video_fake)-1):
                    video_gan_feat_loss += feat_weights * F.l1_loss(pred_video_fake[i], pred_video_real[i].detach()) * (self.video_gan_weight>0)

            gan_feat_loss = disc_factor * self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)
            
            if T > 1:
                self.log("train/video_gan_feat_loss", video_gan_feat_loss, logger=True, on_step=True, on_epoch=True)
                self.log("train/g_video_loss", g_video_loss, logger=True, on_step=True, on_epoch=True)
                self.log("train/video_perceptual_loss", perceptual_video_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            else:
                self.log("train/image_gan_feat_loss", image_gan_feat_loss, logger=True, on_step=True, on_epoch=True)
                self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/g_image_loss", g_image_loss, logger=True, on_step=True, on_epoch=True)
            

            perceptual_loss += perceptual_video_loss

            self.log("train/recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            if not self.use_vae:
                self.log("train/commitment_loss", vq_output['commitment_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log('train/perplexity', vq_output['perplexity'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log('train/usage', vq_output['avg_usage'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            else:
                self.log("train/kl_loss", kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

        if optimizer_idx == 1:
            # discriminator
            if self.apply_allframes:
                logits_image_real, _ = self.image_discriminator(all_frames.detach(), self.apply_diffaug)
                logits_image_fake, _ = self.image_discriminator(all_frames_recon.detach(), self.apply_diffaug)
            else:
                logits_image_real, _ = self.image_discriminator(frames.detach(), self.apply_diffaug)
                logits_image_fake, _ = self.image_discriminator(frames_recon.detach(), self.apply_diffaug)

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)

            if T > 1:
                logits_video_real, _ = self.video_discriminator(x.detach(), self.apply_diffaug)
                logits_video_fake, _ = self.video_discriminator(x_recon.detach(), self.apply_diffaug)
                d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            else:
                logits_video_real, logits_video_fake = None, None
                d_video_loss = 0.
            
            disc_factor = adopt_weight(self.global_step, threshold=self.args.discriminator_iter_start)
            discloss = disc_factor * (self.image_gan_weight*d_image_loss + self.video_gan_weight*d_video_loss)
            
            self.log("train/logits_image_real", logits_image_real.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_image_fake", logits_image_fake.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_real", logits_video_real.mean().detach() if logits_video_real is not None else 0., logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_fake", logits_video_fake.mean().detach() if logits_video_fake is not None else 0., logger=True, on_step=True, on_epoch=True)
            self.log("train/d_image_loss", d_image_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/d_video_loss", d_video_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return discloss

        if self.apply_allframes:
            perceptual_loss = self.perceptual_model(all_frames, all_frames_recon) * self.perceptual_weight
        else:
            perceptual_loss = self.perceptual_model(frames, frames_recon) * self.perceptual_weight
        
        return recon_loss, x_recon, vq_output, perceptual_loss

    def training_step(self, batch, batch_idx):
        if len(batch) == 1:
            batch = batch[0]
        else:
            assert len(batch) == len(self.args.sample_ratio)
            if not self.args.force_alternation:
                sample_ratios = self.args.sample_ratio
                sample_ratios = [r / sum(sample_ratios) for r in sample_ratios]
                batch = random.choices(batch, weights=sample_ratios, k=1)[0]
            
            else:
                num_datasets = len(batch)
                batch = batch[batch_idx % num_datasets]

        # print(batch_idx, batch["video"].shape)

        x = batch['video']
        sch1, sch2 = self.lr_schedulers()
        opt1, opt2 = self.optimizers()

        # if optimizer_idx == 0:
        recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(x, optimizer_idx=0)
        commitment_loss = vq_output['commitment_loss']
        loss_generator = (recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss) / self.grad_accumulates

        self.manual_backward(loss_generator)

        cur_global_step = self.global_step
        if (cur_global_step + 1) % self.grad_accumulates == 0:
            optim_gen = True
            """if self.grad_check_thres is not None:
                for p in opt1.param_groups[0]['params']:
                    if p.grad is not None and torch.any(p.grad > self.grad_check_thres):
                        optim = False""" 

            if cur_global_step > 100000:
                if self.recloss_check_thres is not None:
                    if recon_loss.item() > self.recloss_check_thres:
                        optim_gen = False
                
                if self.perloss_check_thres is not None:
                    if perceptual_loss.item() > self.perloss_check_thres:
                        optim_gen = False

            
            if optim_gen:
                if self.grad_clip_val is not None:
                    self.clip_gradients(opt1, gradient_clip_val=self.grad_clip_val)
                
                opt1.step()
            
            sch1.step(cur_global_step)
            opt1.zero_grad()

        # if optimizer_idx == 1:
        discloss = self.forward(x, optimizer_idx=1)
        loss_discriminator = discloss / self.grad_accumulates

        self.manual_backward(loss_discriminator)

        if (cur_global_step + 1) % self.grad_accumulates == 0:
            optim_disc = True
            
            """if self.grad_check_thres is not None:
                for p in opt2.param_groups[0]['params']:
                    if p.grad is not None and torch.any(p.grad > self.grad_check_thres):
                        optim = False"""

            if self.disloss_check_thres is not None:
                if discloss.item() < self.disloss_check_thres:
                    optim_disc = False

            if optim_disc and optim_gen:
                if self.grad_clip_val_disc is not None:
                    self.clip_gradients(opt2, gradient_clip_val=self.grad_clip_val_disc)
                opt2.step()
            
            sch2.step(cur_global_step)
            opt2.zero_grad()


    def validation_step(self, batch, batch_idx):
        x = batch['video'] # TODO: batch['stft']
        recon_loss, _, vq_output, perceptual_loss = self.forward(x)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)

        if not self.use_vae:
            self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
            self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)
        else:
            self.log("val/kl_loss", vq_output['commitment_loss'], prog_bar=True)

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters())+
                                    list(self.codebook.parameters()),
                                    lr=self.args.lr, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters())+
                                    list(self.video_discriminator.parameters()),
                                    lr=self.args.lr * self.args.dis_lr_multiplier, betas=(0.5, 0.9))
        
        lr_min = self.args.lr_min
        train_iters = self.args.max_steps
        warmup_steps = self.args.warmup_steps
        warmup_lr_init = self.args.warmup_lr_init

       
        sch_ae = CosineLRScheduler(
            opt_ae,
            lr_min = lr_min,
            t_initial = train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_steps,
            cycle_mul = 1.,
            cycle_limit=1,
            t_in_epochs=True,
        )

        if self.args.dis_warmup_steps == 0:
            self.args.dis_warmup_steps = warmup_steps

        sch_disc = CosineLRScheduler(
            opt_disc,
            lr_min = lr_min * self.args.dis_lr_multiplier if self.args.dis_minlr_multiplier else lr_min,
            t_initial = train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=self.args.dis_warmup_steps,
            cycle_mul = 1.,
            cycle_limit=1,
            t_in_epochs=True,
        )

        return [opt_ae, opt_disc], [{"scheduler": sch_ae, "interval": "step"}, {"scheduler": sch_disc, "interval": "step"}]

    """def lr_scheduler_step(self, scheduler):
        print(scheduler)
        print(self.current_epoch)
        print(self.global_step)
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value"""


    def log_images(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch['video']
        x = x.to(self.device)
        frames, frames_rec, _, _, _ = self(x, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        x = batch['video']
        _, _, x, x_rec, _ = self(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        # training configurations
        parser.add_argument('--lr_min', type=float, default=0.)
        parser.add_argument('--warmup_steps', type=int, default=0)
        parser.add_argument('--warmup_lr_init', type=float, default=0.)
        parser.add_argument('--grad_accumulates', type=int, default=1)
        parser.add_argument('--grad_clip_val', type=float, default=1.0)
        parser.add_argument('--grad_clip_val_disc', type=float, default=1.0)

        parser.add_argument('--disloss_check_thres', type=float, default=None)
        parser.add_argument('--perloss_check_thres', type=float, default=None)
        parser.add_argument('--recloss_check_thres', type=float, default=None)

        parser.add_argument('--force_alternation', action="store_true")
        

        parser.add_argument('--kl_weight', type=float, default=0.)
        parser.add_argument('--use_vae', action="store_true")

        parser.add_argument('--video_perceptual_weight', type=float, default=0.)
        parser.add_argument('--initialize_vit', action="store_true")


        # configuration for discriminator
        parser.add_argument('--sigmoid_in_disc', action="store_true")
        parser.add_argument('--activation_in_disc', type=str, default="leaky_relu")
        parser.add_argument('--apply_blur', action="store_true")
        parser.add_argument('--apply_noise', action="store_true")
        parser.add_argument('--apply_diffaug', action="store_true")

        parser.add_argument('--logitslaplace_weight', type=float, default=0.)
        

        parser.add_argument('--dis_warmup_steps', type=int, default=0)
        parser.add_argument('--dis_lr_multiplier', type=float, default=1.)
        parser.add_argument('--dis_minlr_multiplier', action="store_true")

        parser.add_argument("--recon_loss_type", type=str, default='l1', choices=['l1', 'l2'])
        parser.add_argument('--patch_size', type=int, default=16)
        parser.add_argument('--gen_upscale', type=int, default=None)
        parser.add_argument('--patch_embed', type=str, default='linear', choices=['linear', 'cnn', 'pixelshuffle'])
        
        parser.add_argument('--enc_block', type=str, default='tttt')
        parser.add_argument('--dec_block', type=str, default='tttt')
        parser.add_argument('--q_strides', type=str, default='1111')
        parser.add_argument('--use_checkpoint', action="store_true")

        parser.add_argument('--twod_window_size', type=int, default=4)
        parser.add_argument('--temporal_patch_size', type=int, default=2)
        parser.add_argument('--defer_temporal_pool', action="store_true")
        parser.add_argument('--defer_spatial_pool', action="store_true")
        parser.add_argument('--spatial_pos', type=str, default="rel", choices=["rel", "rope"])

        parser.add_argument('--spatial_depth', type=int, default=4)
        parser.add_argument('--temporal_depth', type=int, default=4)
        parser.add_argument('--casual_in_temporal_transformer', action="store_true") # tune the param
        parser.add_argument('--casual_in_peg', action="store_true")
        parser.add_argument('--dim_head', type=int, default=64)
        parser.add_argument('--heads', type=int, default=8)
        parser.add_argument('--attn_dropout', type=float, default=0.)
        parser.add_argument('--ff_dropout', type=float, default=0.)
        parser.add_argument('--ff_mult', type=float, default=4.)

        parser.add_argument('--use_external_codebook', action="store_true")
        parser.add_argument('--fp32_quant', action="store_true")
        parser.add_argument('--codebook_type', type=str, default='vq')
        parser.add_argument('--codebook_dim', type=int, default=None)
        parser.add_argument('--l2_code', action="store_true")
        parser.add_argument('--commitment_weight', type=float, default=0.25)

        parser.add_argument('--resolution_scale', default=None, nargs='+', type=float)


        return parser

  

class OmniTokenizer_Encoder(nn.Module):
    def __init__(self, image_size, patch_embed, norm_type, block='tttt', window_size=4, q_strides='1111', spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, casual_in_temporal_transformer=False, dim=512, 
                    casual_in_peg=True, dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., initialize=False, sequence_length=17):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.temporal_patch_size = temporal_patch_size
        self.block = block

        # self.spatial_rel_pos_bias = ContinuousPositionBias(
        #     dim=dim, heads=heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (
            image_width % patch_width) == 0

        if patch_embed == 'linear':
            if defer_temporal_pool:
                temporal_patch_size //= 2
                self.temporal_patch_size = temporal_patch_size
                self.temporal_pool = nn.AvgPool3d(kernel_size=(2, 1, 1))
            else:
                self.temporal_pool = nn.Identity()
            
            if defer_spatial_pool:
                self.patch_size =  pair(patch_size // 2)
                patch_height, patch_width = self.patch_size
                self.spatial_pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
            else:
                self.spatial_pool = nn.Identity()

            self.to_patch_emb_first_frame = nn.Sequential(
                Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)',
                        p1=patch_height, p2=patch_width),
                nn.LayerNorm(image_channel * patch_width * patch_height),
                nn.Linear(image_channel * patch_width * patch_height, dim),
                nn.LayerNorm(dim)
            )

            self.to_patch_emb = nn.Sequential(
                Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)',
                        p1=patch_height, p2=patch_width, pt=temporal_patch_size),
                nn.LayerNorm(image_channel * patch_width *
                            patch_height * temporal_patch_size),
                nn.Linear(image_channel * patch_width *
                        patch_height * temporal_patch_size, dim),
                nn.LayerNorm(dim)
            )
        elif patch_embed == 'cnn':
            self.to_patch_emb_first_frame = nn.Sequential(
                # SamePadConv3d(image_channel, dim, kernel_size=(1, patch_height, patch_width), stride=(1, patch_height, patch_width)),
                nn.Conv3d(image_channel, dim, kernel_size=(1, patch_height, patch_width), stride=(1, patch_height, patch_width)),
                Normalize(dim, norm_type),
                Rearrange('b c t h w -> b t h w c'),
            )

            self.to_patch_emb = nn.Sequential(
                # SamePadConv3d(image_channel, dim, kernel_size=(temporal_patch_size, patch_height, patch_width), stride=(temporal_patch_size, patch_height, patch_width)),
                nn.Conv3d(image_channel, dim, kernel_size=(temporal_patch_size, patch_height, patch_width), stride=(temporal_patch_size, patch_height, patch_width)),
                Normalize(dim, norm_type),
                Rearrange('b c t h w -> b t h w c'),
            )

            self.temporal_pool, self.spatial_pool = nn.Identity(), nn.Identity()

        else:
            raise NotImplementedError

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=casual_in_peg,
            ff_mult=ff_mult
        )

        self.q_strides = q_strides
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, block=block, window_size=window_size, spatial_pos=spatial_pos, **transformer_kwargs)

        
        if casual_in_temporal_transformer:
            transformer_kwargs["causal"] = True

        self.enc_temporal_transformer = Transformer(
            depth=temporal_depth, block='t' * temporal_depth, **transformer_kwargs)
        
        if initialize:
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
    

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]  # batch size
        # h, w = self.patch_height_width  # patch h,w
        is_image = tokens.shape[1] == 1

        # video shape, last dimension is the embedding size
        video_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        
        # encode - spatial
        # tokens = self.enc_spatial_transformer(
        #     tokens, attn_bias_func=self.spatial_rel_pos_bias, video_shape=video_shape, q_strides=self.q_strides)
        tokens = self.enc_spatial_transformer(tokens, video_shape=video_shape, q_strides=self.q_strides, is_spatial=True)

        hw = tokens.shape[1]
        new_h, new_w = int(math.sqrt(hw)), int(math.sqrt(hw))
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=new_h, w=new_w)

        # encode - temporal
        video_shape2 = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape2, is_spatial=False)
        # tokens = self.enc_temporal_transformer(tokens)

        # codebook expects:  [b, c, t, h, w]
        tokens = rearrange(tokens, '(b h w) t d -> b d t h w', b=b, h=new_h, w=new_w)
        tokens = self.spatial_pool(tokens)

        if tokens.shape[2] > 1:
            first_frame_tokens = tokens[:, :, 0:1]
            rest_frames_tokens = tokens[:, :, 1:]
            rest_frames_tokens = self.temporal_pool(rest_frames_tokens)
            tokens = torch.cat([first_frame_tokens, rest_frames_tokens], dim=2)

        return tokens

    
    def forward(self, video, is_image, mask=None, use_checkpoint=False):
        # 4 is BxCxHxW (for images), 5 is BxCxFxHxW
        assert video.ndim in {4, 5}

        if is_image:  # add temporal channel to 1 for images only
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert mask is None

        _, _, f, *image_dims = *video.shape, 

        # assert tuple(image_dims) == self.image_size
        assert mask is None or mask.shape[-1] == f
        assert divisible_by(
            f - 1, self.temporal_patch_size), f'number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})'

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # derive patches
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)

        if rest_frames.shape[2] != 0:
            rest_frames_tokens = self.to_patch_emb(rest_frames)
            # simple cat
            tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)

        else:
            tokens = first_frame_tokens

        if not use_checkpoint:
            return self.encode(tokens)
        else:
            return checkpoint.checkpoint(self.encode, tokens)


class OmniTokenizer_Decoder(nn.Module):
    def __init__(self, image_size, patch_embed, norm_type, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, casual_in_temporal_transformer=False, dim=512, 
                    casual_in_peg=True, dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., gen_upscale=None, initialize=False,
                    sequence_length=17):
        super().__init__()
        self.gen_upscale = gen_upscale
        if gen_upscale is not None:
            patch_size *= gen_upscale


        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.block = block

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=casual_in_peg,
            ff_mult=ff_mult
        )

        #self.spatial_rel_pos_bias = ContinuousPositionBias(
        #    dim=dim, heads=heads) # HACK this: whether shared pos encoding is better or on the contrary

        self.dec_spatial_transformer = Transformer(
            depth=spatial_depth, block=block, window_size=window_size, spatial_pos=spatial_pos, **transformer_kwargs)
        
        if casual_in_temporal_transformer:
            transformer_kwargs["causal"] = True

        self.dec_temporal_transformer = Transformer(
            depth=temporal_depth, block='t' * temporal_depth, **transformer_kwargs)

        if patch_embed == "linear":
            if defer_temporal_pool:
                temporal_patch_size //= 2
                self.temporal_patch_size = temporal_patch_size
                self.temporal_up = nn.Upsample(scale_factor=(2, 1, 1), mode="nearest") # AvgPool3d(kernel_size=(2, 1, 1))
            else:
                self.temporal_up = nn.Identity()
            
            if defer_spatial_pool:
                self.patch_size =  pair(patch_size // 2)
                patch_height, patch_width = self.patch_size
                self.spatial_up = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest") # nn.AvgPool3d(kernel_size=(1, 2, 2))
            else:
                self.spatial_up = nn.Identity()
            
            # b 1 nhnw dim -> b 1 phpw 3phpw
            self.to_pixels_first_frame = nn.Sequential(
                nn.Linear(dim, image_channel * patch_width * patch_height),
                Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)',
                        p1=patch_height, p2=patch_width)
            )

            self.to_pixels = nn.Sequential(
                nn.Linear(dim, image_channel * patch_width *
                        patch_height * temporal_patch_size),
                Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)',
                        p1=patch_height, p2=patch_width, pt=temporal_patch_size),
            )

        elif patch_embed == "cnn":
            # torch.Size([1, 1, 8, 8, 512])
            self.to_pixels_first_frame = nn.Sequential(
                Rearrange('b 1 h w dim -> b dim 1 h w', h=image_size//patch_size),
                # SamePadConvTranspose3d(dim, image_channel, kernel_size=(1, patch_height, patch_width), stride=(1, patch_height, patch_width)),
                nn.ConvTranspose3d(dim, image_channel, kernel_size=(1, patch_height, patch_width), stride=(1, patch_height, patch_width)),
                Normalize(image_channel, norm_type)
            )

            self.to_pixels = nn.Sequential(
                Rearrange('b t h w dim -> b dim t h w', h=image_size//patch_size),
                # SamePadConvTranspose3d(dim, image_channel, kernel_size=(temporal_patch_size, patch_height, patch_width), stride=(temporal_patch_size, patch_height, patch_width)),
                nn.ConvTranspose3d(dim, image_channel, kernel_size=(temporal_patch_size, patch_height, patch_width), stride=(temporal_patch_size, patch_height, patch_width)),
                Normalize(image_channel, norm_type)
            )
            self.temporal_up = nn.Identity()
            self.spatial_up = nn.Identity()
        
        else:
            raise NotImplementedError
    
        if initialize:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def patch_height_width(self):
        if self.gen_upscale is None:
            return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
        else:
            return int(self.image_size[0] // self.patch_size[0] * self.gen_upscale), int(self.image_size[1] // self.patch_size[1] * self.gen_upscale)

    def decode(
        self,
        tokens,
    ):
        b = tokens.shape[0]
        # h, w = self.patch_height_width
        is_image = tokens.shape[1] == 1
        video_shape = tuple(tokens.shape[:-1]) # b t h' w' d
        h = tokens.shape[2]
        w = tokens.shape[3]


        # decode - temporal
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.dec_temporal_transformer(tokens, video_shape=video_shape, is_spatial=False)
        # tokens = self.dec_temporal_transformer(tokens)

        # might spatial downsampling here
        down_op = self.block.count('n') + self.block.count('r')
        down_ratio = int(2 ** down_op)

        # decode - spatial
        tokens = rearrange(tokens, '(b h w) t d -> (b t) (h w) d', b=b, h=h//down_ratio, w=w//down_ratio)
        #tokens = self.dec_spatial_transformer(
        #    tokens, attn_bias_func=self.spatial_rel_pos_bias, video_shape=video_shape)
        tokens = self.dec_spatial_transformer(tokens, video_shape=video_shape, is_spatial=True)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # to pixels
        first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]
        first_frame = self.to_pixels_first_frame(first_frame_token)

        if rest_frames_tokens.shape[1] != 0:
            rest_frames = self.to_pixels(rest_frames_tokens)
            recon_video = torch.cat((first_frame, rest_frames), dim=2)
        else:
            recon_video = first_frame
        
        return recon_video
    

    def forward(self, tokens, is_image, mask=None, use_checkpoint=False):
        # expected input: b d t h w -> b t h w d
        if tokens.shape[2] > 1:
            first_frame_tokens = tokens[:, :, 0:1]
            rest_frames_tokens = tokens[:, :, 1:]
            rest_frames_tokens = self.temporal_up(rest_frames_tokens)
            tokens = torch.cat([first_frame_tokens, rest_frames_tokens], dim=2)

        tokens = self.spatial_up(tokens)
        tokens = tokens.permute(0, 2, 3, 4, 1).contiguous()

        if not use_checkpoint:
            recon_video = self.decode(tokens)
        else:
            recon_video = checkpoint.checkpoint(self.decode, tokens)

        # handle shape if we are training on images only
        returned_recon = rearrange(
            recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()

        return returned_recon