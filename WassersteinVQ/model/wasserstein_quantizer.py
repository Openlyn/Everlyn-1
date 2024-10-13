from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dist
from torch import distributed as tdist
from model.base_quantizer import BaseQuantizer

class WassersteinQuantizer(BaseQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def forward(self, z_enc):
        B, C, H, W = z_enc.shape
        z_rest = z_enc
        z_dec = torch.zeros_like(z_rest)
        token_cat: List[torch.Tensor] = []
        z_cat: List[torch.Tensor] = []

        with torch.cuda.amp.autocast(enabled=False):
            commit_loss: torch.Tensor = 0.0
            vq_loss: torch.Tensor = 0.0
            wasserstein_loss: torch.Tensor = 0.0
            wasserstein_last: torch.Tensor = 0.0
            wasserstein_all: torch.Tensor = 0.0

            level_quant_error : List[torch.Tensor] = []     
            for level, pn in enumerate(self.args.ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != len(self.args.ms_token_size) -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
                z_cat.append(z_downscale)

                if (level == len(self.args.ms_token_size) -1):
                    wasserstein_last = self.calc_wasserstein_loss_single(z_downscale)

                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)
                
                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                token_cat.append(token)

                embed = self.embedding(token)
                commit_loss = (embed.detach() - z_downscale).square().sum(1).mean() * self.args.ms_token_size[level]                    

                token_Bhw = token.view(B, pn, pn)
                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != len(self.args.ms_token_size) -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                
                z_dec = z_dec + z_upscale
                z_rest = z_rest - z_upscale.detach()

                quant_error =  (z_enc.detach() - z_dec.detach()).square().sum(1).mean()
                level_quant_error.append(quant_error)  

            vq_loss =  (z_enc - z_dec.detach()).square().sum(1).mean()
            token_cat = torch.cat(token_cat, 0)
            z_cat = torch.cat(z_cat, 0)

            codebook_histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            if self.training:
                if dist.initialized(): handler = tdist.all_reduce(codebook_histogram, async_op=True)

            if self.training and dist.initialized():
                handler.wait()
                
            codebook_usage_counts = (codebook_histogram > 0).float().sum()
            codebook_utilization = codebook_usage_counts.item() / self.args.codebook_size
            
            avg_probs = codebook_histogram/codebook_histogram.sum(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            ### compute wasserstein distance
            wasserstein_all = self.calc_wasserstein_loss_single(z_cat)
            commit_loss *= 1. / sum(self.args.ms_token_size)
            z_dec = (z_dec - z_enc).detach().add_(z_enc)
            
            wasserstein_loss = wasserstein_last + wasserstein_all
        return z_dec, vq_loss, wasserstein_loss, commit_loss, level_quant_error, perplexity, codebook_utilization

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        z_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            level_quant_error : List[torch.Tensor] = []
            level_commit_error: List[torch.Tensor] = []

            for level, pn in enumerate(self.args.ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != len(self.args.ms_token_size) -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
                z_cat.append(z_downscale)

                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)
                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                token_cat.append(token)

                embed = self.embedding(token)
                commit_error = (embed.detach() - z_downscale.detach()).square().sum(1).mean()
                level_commit_error.append(commit_error)

                token_Bhw = token.view(B, pn, pn)
                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != len(self.args.ms_token_size) -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()

                z_dec.add_(z_upscale)
                z_rest.sub_(z_upscale)

                quant_error = (z_dec.detach() - z_enc.detach()).square().sum(1).mean()
                level_quant_error.append(quant_error)

            ## updating codebook
            token_cat = torch.cat(token_cat, 0)
            z_cat = torch.cat(z_cat, 0)

            codebook_histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            
            if dist.initialized(): 
                handler1 = tdist.all_reduce(codebook_histogram, async_op=True)
                handler1.wait()

            ### compute wasserstein distance
            wasserstein_distance = self.calc_wasserstein_loss(z_cat.detach())

        return z_dec, wasserstein_distance, codebook_histogram, level_quant_error, level_commit_error