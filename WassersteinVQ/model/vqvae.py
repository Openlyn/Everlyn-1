import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import Pack
from utils.util import load_config
from model.wasserstein_quantizer import WassersteinQuantizer
from model.discriminator import NLayerDiscriminator, weights_init
from model.lpips import LPIPS
from model.encoder_decoder import Encoder, Decoder

class VQVAE(nn.Module):
    def __init__(self, args):
        super(VQVAE, self).__init__()
        self.args = args
        config = load_config(args.ae_config_path, display=True)
        self.encoder = Encoder(**config.model.params.ddconfig)
        self.decoder = Decoder(**config.model.params.ddconfig)
        self.unshuffle = nn.PixelUnshuffle(args.factor)
        self.shuffle = nn.PixelShuffle(args.factor)
        self.quantizer = WassersteinQuantizer(args)
        self.lpips = LPIPS().eval()
        self.discriminator = NLayerDiscriminator(input_nc=3,
                                                n_layers=2,
                                                use_actnorm=False,
                                                ndf=64
                                                ).apply(weights_init)
        self.quant_conv = torch.nn.Conv2d(args.feature_dim, args.latent_dim, kernel_size=1)
        self.post_quant_conv = torch.nn.Conv2d(args.latent_dim, args.feature_dim, kernel_size=1)



    def quantized_feature_to_img(self, z_dec: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_multiscale_token(self, x): 
        ## x image [32, 3, 256, 256]
        ## f [32, D, 16, 16]
        z_enc = self.quant_conv(self.encoder(x))
        multiscale_token = self.quantizer.obtain_multiscale_image_token(z_enc)
        return multiscale_token

    def img_to_reconstructed_img(self, x, last_one = False):
        z_enc = self.quant_conv(self.encoder(x))
        multiscale_quantized_feature = self.quantizer.obtain_multiscale_quantized_feature(z_enc)
        if last_one:
            return self.decoder(self.post_quant_conv(multiscale_quantized_feature[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(z_dec)).clamp_(-1, 1) for z_dec in multiscale_quantized_feature]
    
    def multiscale_token_to_img(self, multiscale_token, last_one=False):
        multis_cale_quantized_feature = self.quantizer.multiscale_token_to_multiscale_quantized_feature(multiscale_token)
        if last_one:
            return self.decoder(self.post_quant_conv(multis_cale_quantized_feature[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(multis_cale_quantized_feature[level])).clamp_(-1, 1) for level in range(len(self.args.ms_token_size))]

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, discriminator_weight, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight

    def forward(self, x, data_iter_step, step=0):
        ## x size: [B, 3, resolution, resolution]

        ## z_enc size: [B, codebook_dim, h, w]
        z_enc = self.unshuffle(self.quant_conv(self.encoder(x)))

        ## multi-scale residual quantization
        z_dec, vq_loss, wasserstein_loss, commit_loss, level_quant_error, perplexity, codebook_utilization = self.quantizer(z_enc)
        x_rec = self.decoder(self.post_quant_conv(self.shuffle(z_dec)))

        ## compute loss
        rec_loss = torch.mean(torch.abs(x.contiguous() - x_rec.contiguous()))
        lpips_loss = torch.mean(self.lpips(x.contiguous(), x_rec.contiguous()))

        ##update encoder-decoder
        if step == 0:
            logits_fake = self.discriminator(x_rec)
            g_loss = -torch.mean(logits_fake)
            d_weight = self.calculate_adaptive_weight(rec_loss + self.args.lambd * lpips_loss, g_loss, self.args.rate_d, last_layer=self.decoder.conv_out.weight)

            if data_iter_step > self.args.disc_start:
                ae_loss = rec_loss + self.args.alpha * vq_loss +  self.args.beta * commit_loss + self.args.gamma * wasserstein_loss + self.args.lambd * lpips_loss + d_weight * g_loss
            else:
                ae_loss = rec_loss + self.args.alpha * vq_loss +  self.args.beta * commit_loss + self.args.gamma * wasserstein_loss + self.args.lambd * lpips_loss + 0 * g_loss

            loss_pack = Pack(ae_loss=ae_loss, rec_loss=rec_loss, commit_loss=commit_loss, vq_loss=vq_loss, lpips_loss=lpips_loss, wasserstein_loss=wasserstein_loss, g_loss=g_loss, perplexity=perplexity, codebook_utilization=codebook_utilization)
            return ae_loss, loss_pack, level_quant_error

        ##upadte discriminator
        if step == 1:
            logits_real = self.discriminator(x.contiguous().detach().clone())
            logits_fake = self.discriminator(x_rec.detach().clone())
            d_loss = self.hinge_d_loss(logits_real, logits_fake)
            loss_pack = Pack(d_loss=d_loss)
            return d_loss, loss_pack

    def collect_eval_info(self, x):
        ## x:[B, 3, resolution, resolution]
        ## encoder [B, codebook_dim, h, w]
        z_enc = self.unshuffle(self.quant_conv(self.encoder(x)))

        z_dec, wasserstein_distance, codebook_histogram, level_quant_error, level_commit_error = self.quantizer.collect_eval_info(z_enc)
        x_rec = self.decoder(self.post_quant_conv(self.shuffle(z_dec))).clamp(-1.0, 1.0)
        rec_loss = torch.mean(torch.abs(x.contiguous() - x_rec.contiguous()))
        
        return x_rec, rec_loss, wasserstein_distance, codebook_histogram, level_quant_error, level_commit_error

