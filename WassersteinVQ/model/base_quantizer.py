from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dist
from torch import distributed as tdist       

class BaseQuantizer(nn.Module):
    def __init__(self, args):
        super(BaseQuantizer, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.codebook_size, args.codebook_dim)
        self.embedding.weight.data.normal_(0, args.std)
        self.embedding.weight.requires_grad = False

    def calc_wasserstein_loss_single(self, z):
        N = z.size(0)
        D = z.size(1)

        z_mean = z.mean(0)
        z_covariance = torch.mm((z - torch.mean(z, dim=0, keepdim=True)).t(), z - torch.mean(z, dim=0, keepdim=True))/N

        ### calculation mean part
        part_mean =  torch.sum(torch.multiply(z_mean, z_mean))

        ### 1/2 covariance of codebook vectors
        std = self.args.std
        S, Q = torch.linalg.eigh(z_covariance)
        sqrt_S = torch.sqrt(torch.diag(F.relu(S)) + 1e-8)
        z_sqrt_covariance = torch.mm(torch.mm(Q, sqrt_S), Q.T)

        part_covariance = torch.trace(z_covariance  - 2.0 * std * z_sqrt_covariance)
        part_covariance += D * std * std

        wasserstein_loss = torch.sqrt(part_mean + F.relu(part_covariance) + 1e-8)
        return wasserstein_loss

    ## the wasserstein distance between feature vectors and codebook vectors
    def calc_wasserstein_loss(self, z):
        ### compute the mean and covariance of feature
        N = z.size(0)
        D = z.size(1)

        z_mean = z.mean(0)
        z_covariance = torch.mm((z - torch.mean(z, dim=0, keepdim=True)).t(), z - torch.mean(z, dim=0, keepdim=True))/N
        
        ### compute the mean and covariance of codebook vectors
        c = self.embedding.weight
        c_mean = c.mean(0)
        c_covariance = torch.mm((c - torch.mean(c, dim=0, keepdim=True)).t(), c - torch.mean(c, dim=0, keepdim=True))/self.args.codebook_size

        ### calculation of part1
        part_mean =  torch.sum(torch.multiply(z_mean - c_mean, z_mean - c_mean))
        d_covariance = torch.mm(z_covariance, c_covariance)
        
        ### 1/2 d_covariance
        S, Q = torch.linalg.eigh(d_covariance)
        sqrt_S = torch.sqrt(torch.diag(F.relu(S)) + 1e-10)
        d_sqrt_covariance = torch.mm(torch.mm(Q, sqrt_S), Q.T)

        #############calculation of part2
        part_covariance = F.relu(torch.trace(z_covariance + c_covariance - 2.0 * d_sqrt_covariance))
        wasserstein_loss = torch.sqrt(part_mean + part_covariance + 1e-10)
        return wasserstein_loss

    ## continous feature (from encoder) into multi-scale image token
    ## r1, r2, r3, ..., rK
    def obtain_multiscale_image_token(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone() 

        ## output, multis_cale_image_token
        ret : List[torch.Tensor] = [] 

        ## patch_hws = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)]
        patch_hws =  [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in self.args.ms_token_size] 
        for step, (ph, pw) in enumerate(patch_hws):
            z_downscale = F.interpolate(z_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (step != len(patch_hws) -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)

            ## distance [B*ph*pw, vocab_size]
            distance = torch.sum(z_downscale.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            distance.addmm_(z_downscale, self.embedding.weight.data.T, alpha=-2, beta=1)

            ## token [B*ph*pw]
            token = torch.argmin(distance, dim=1)

            token_Bhw = token.view(B, ph, pw)
            z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (step != len(patch_hws) -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
            z_rest.sub_(z_upscale)
            ret.append(token.reshape(B, ph*pw))
        return ret

    ## continous feature (from encoder) quantized feature (to decoder)
    ## \hat{z1}, \hat{z1}+\hat{z2},..., \hat{z1}+...+\hat{zK}
    def obtain_multiscale_quantized_feature(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        ## output, multis_cale_quantized_feature
        ret : List[torch.Tensor] = []

        ## patch_hws = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)]
        patch_hws =  [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in self.args.ms_token_size] 
        for step, (ph, pw) in enumerate(patch_hws):
            z_downscale =  F.interpolate(z_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (step != len(patch_hws) -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
            ## distance [B*ph*pw, vocab_size]
            distance = torch.sum(z_downscale.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            distance.addmm_(z_downscale, self.embedding.weight.data.T, alpha=-2, beta=1)

            ## token [B*ph*pw]
            token = torch.argmin(distance, dim=1)

            token_Bhw = token.view(B, ph, pw)
            z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (step != len(patch_hws) -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
            z_dec.add_(z_upscale)
            z_rest.sub_(z_upscale)
            ret.append(z_dec.clone())
        return ret

    ## r1, r2, r3, ..., rK to \hat{z1}, \hat{z1}+\hat{z2},..., \hat{z1}+...+\hat{zK} (z_dec)
    def multiscale_token_to_multiscale_quantized_feature(self, multiscale_token):
        B = multiscale_token[0].shape[0]
        H = W = self.args.ms_token_size[-1] ## H = W = 16
        C = self.args.codebook_dim

        ## multis_cale_quantized_feature
        ret : List[torch.Tensor] = []
        z_dec = multiscale_token[0].new_zeros(B, C, H, W, dtype=torch.float32)
        for level, pn in enumerate(self.args.ms_token_size): # from small to large
            token = multiscale_token[level].view(B, pn, pn)
            z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != self.args.ms_token_size -1) else self.embedding(token).permute(0, 3, 1, 2).contiguous()
            z_dec.add_(z_upscale)
            ret.append(z_dec.clone())
        return ret

    ### For training GPT models
    def obtain_contextualized_embedding(self, multiscale_token):
        next_scales = []
        B = multiscale_token[0].shape[0]
        C = self.args.codebook_dim
        H = W = self.args.ms_token_size[-1]
        num_level = len(self.args.ms_token_size)

        token_embedding = multiscale_token[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.args.ms_token_size[0]
        for level in range(num_level-1):
            level_embedding = F.interpolate(self.embedding(multiscale_token[level]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            token_embedding.add_(level_embedding)
            pn_next = self.args.ms_token_size[level+1]
            next_scales.append(F.interpolate(token_embedding, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) 

    ### for VAR inference (generation phase)
    def obtain_next_autoregressive_input(self, level, f_hat, predicted_token):
        H = W = self.args.ms_token_size[-1]
        pn = self.args.ms_token_size[level]
        if level != len(self.args.ms_token_size)-1:
            h = F.interpolate(self.embedding(predicted_token).transpose_(1, 2).view(B, C, pn, pn), size=(H, W), mode='bicubic')
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.args.ms_token_size[level+1], self.args.ms_token_size[scale+1]), mode='area')
        else:
            h = self.embedding(predicted_token).transpose_(1, 2).view(B, C, pn, pn)
            f_hat.add_(h)
            return f_hat, f_hat


if __name__ == '__main__':
    ## check torch.linalg.eigh 
    N = 20000
    A = torch.randn(N, 3)

    A_mean = A.mean(0)
    A_covariance = torch.mm((A - torch.mean(A, dim=0, keepdim=True)).t(), A - torch.mean(A, dim=0, keepdim=True))/N

    print("A_covariance:", A_covariance)

    S, Q = torch.linalg.eigh(A_covariance)
    sqrt_S = torch.sqrt(torch.diag(S))
    print("sqrt_S:", sqrt_S)
    A_sqrt_covariance = torch.mm(torch.mm(Q, sqrt_S), Q.T)
    print("A_sqrt_covariance:", A_sqrt_covariance)

    A2_covariance = torch.mm(A_sqrt_covariance, A_sqrt_covariance)
    print("A2_covariance:", A2_covariance )

    normal_S = torch.diag(S)
    A_normal_covariance = torch.mm(torch.mm(Q, normal_S), Q.T)

    print("A_normal_covariance:", A_normal_covariance)