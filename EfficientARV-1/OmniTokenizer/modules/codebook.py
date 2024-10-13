from enum import unique
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ..utils import shift_dim

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=1.0, usage_sigma=0.99, fp32_quant=False):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())
        self.register_buffer('codebook_usage', torch.zeros(n_codes))

        self.call_cnt = 0
        self.usage_sigma = usage_sigma

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

        self.fp32_quant = fp32_quant

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))
    

    def calculate_batch_codebook_usage_percentage(self, batch_encoding_indices):
        # Flatten the batch of encoding indices into a single 1D tensor
        all_indices = batch_encoding_indices.flatten()
        
        # Obtain the total number of encoding indices in the batch to calculate percentages
        total_indices = all_indices.numel()
        
        # Initialize a tensor to store the percentage usage of each code
        codebook_usage_percentage = torch.zeros(self.n_codes, device=all_indices.device)
        
        # Count the number of occurrences of each index and get their frequency as percentages
        unique_indices, counts = torch.unique(all_indices, return_counts=True)
        # Calculate the percentage
        percentages = (counts.float() / total_indices)
        
        # Populate the corresponding percentages in the codebook_usage_percentage tensor
        codebook_usage_percentage[unique_indices.long()] = percentages
        
        return codebook_usage_percentage
    


    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2) # [bthw, c]
        
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, ncode]

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs) # [bthw, ncode]
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:]) # [b, t, h, w, ncode]

        embeddings = F.embedding(encoding_indices, self.embeddings) # [b, t, h, w, c]
        embeddings = shift_dim(embeddings, -1, 1) # [b, c, t, h, w]

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        try:
            usage = self.calculate_batch_codebook_usage_percentage(encoding_indices)
        except:
            usage = torch.zeros(self.n_codes, device=encoding_indices.device)
        

        # print(usage.shape, torch.zeros(self.n_codes).shape)

        if self.call_cnt == 0:
            self.codebook_usage.data = usage
        else:
            self.codebook_usage.data = self.usage_sigma * self.codebook_usage.data + (1 - self.usage_sigma) * usage

        self.call_cnt += 1
        # avg_distribution = self.codebook_usage.data.sum() / self.n_codes
        avg_usage = (self.codebook_usage.data > (1/self.n_codes)).sum() / self.n_codes
            
        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity, avg_usage=avg_usage, batch_usage=usage)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings
