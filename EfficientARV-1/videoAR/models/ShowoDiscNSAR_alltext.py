import argparse
from cgitb import text
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange, repeat
from OmniTokenizer.utils import shift_dim, accuracy, comp_getattr, ForkedPdb
from .llama_gpt_ns_disc_alltext_msf_visemb import GPT_models
from OmniTokenizer.modules.encoders import Labelator, SOSProvider, Identity
import numpy as np
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import Optional
from tqdm import tqdm
import os
import json
import math
from ..utils.utils import save_videos_grid
from .showo.modeling_showo import Showo
import copy

from transformers.integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings

class DiagonalGaussianDistribution:
    def __init__(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
    ):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic

        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)

    def sample(self, generator = None) -> torch.FloatTensor:
        x = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * x

    def mode(self):
        return self.mean

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        dims = list(range(1, self.mean.ndim))

        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        dims = list(range(1, self.mean.ndim))

        if self.deterministic:
            return torch.Tensor([0.0])

        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )
### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True, **kwargs):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs

def sample_all(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True, **kwargs):        
    logits = logits / max(temperature, 1e-5)
    b, n, c = logits.shape
    logits = rearrange(logits,'b n c -> (b n) c')
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    probs = rearrange(probs, '(b n) c -> b n c', b=b, n=n)
    idx = rearrange(idx, '(b n) c -> b n c', b=b, n=n).squeeze(-1)
    return idx, probs
     
class LlamaVAR(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                gpt_model,
                gpt_type,
                vocab_size,
                latent_size,
                num_classes,
                cls_token_num,
                resid_dropout_p,
                ffn_dropout_p,
                token_dropout_p,
                class_dropout_p,
                sample_every_n_latent_frames,
                hidden_dim,
                max_frames,
                showo_pretrained_model_path,
                ckpt_path=None,
                ignore_keys=[],
                sample_size=64,
                caption_dim=2048,
                vae_dim=8,
                ):
        super().__init__()
      
        self.sample_size = sample_size  
        self.sample_every_n_latent_frames = sample_every_n_latent_frames
        self.patch_nums = [(5,1,1), (5,2,2),(5,3,3),(5,4,4),(5,5,5),(5,6,6),(5,8,8),(5, 10, 10), (5,16,16), (5,32,32)]
        self.num_text_tokens = cls_token_num
        self.embed_dim = hidden_dim
        
        self.L = sum(np.prod(pn) for pn in self.patch_nums)
        d = torch.cat([torch.full((np.prod(pn),), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2) #11L
        self.lvl_pos_emb_idx =  dT[:,0].contiguous()
        self.mask = torch.where(d >= dT, 1, 0).reshape(1, 1, self.L, self.L).contiguous().to(torch.bool) # 1 x 1 x L x L
        
        init_std = math.sqrt(1 / hidden_dim/ 3)
        # self.scale_embeddings = nn.Embedding(len(self.patch_nums), hidden_dim)
        # nn.init.trunc_normal_(self.scale_embeddings.weight.data, mean=0, std=init_std)
        
        self.num_token_first_scale = np.prod(self.patch_nums[0])
        # self.pos_start = nn.Parameter(torch.empty(1, cls_token_num, hidden_dim))
        # nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # self.tok_embeddings = nn.Embedding(vocab_size, hidden_dim)
        # nn.init.trunc_normal_(self.tok_embeddings.weight.data, mean=0, std=init_std)
        
        self.caption_embeder = CaptionEmbedder(caption_dim, hidden_dim, class_dropout_p, token_num=cls_token_num)
        self.pre_transformer = MLP(in_features=hidden_dim, hidden_features=hidden_dim, out_features=hidden_dim)
        
        self.apply(self._init_weights)
        
        self.transformer = Showo.from_pretrained(showo_pretrained_model_path)
        self.vid_vocab_size = vocab_size
        self.old_vocab_size = self.transformer.vocab_size
        self.transformer.showo.resize_token_embeddings(self.vid_vocab_size + self.old_vocab_size)
        lm_head = nn.Linear(self.transformer.showo.config.hidden_size, self.vid_vocab_size)
        self.transformer.showo.set_output_embeddings(lm_head)
        self.mask_token_id = self.transformer.config.mask_token_id
    
        # self.post_transformer = MLP(in_features=hidden_dim, hidden_features=hidden_dim, out_features=vae_dim)
        
        # Zero-out output layers:
        # nn.init.constant_(self.output.weight, 0)
        if ckpt_path is not None and len(ckpt_path)>0:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.old_embedding = copy.deepcopy(self.transformer.showo.get_input_embeddings()).requires_grad_(False)
    
    def keep_original_embedding(self):
        is_quantized = hasattr(self.transformer.showo, "hf_quantizer") and self.transformer.showo.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed
            params = [self.transformer.showo.get_input_embeddings().weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                self.transformer.showo.get_input_embeddings().weight.data[:self.old_vocab_size] = self.old_embedding.weight.data[:self.old_vocab_size]
        else:
            self.transformer.showo.get_input_embeddings().weight.data[:self.old_vocab_size] = self.old_embedding.weight.data[:self.old_vocab_size]
    
    def get_trainable_params(self):
        params = {
            # 'transformer.showo.lm_head.weight':self.transformer.showo.lm_head.named_parameters(), 
            # 'transformer.showo.model.embed_tokens.weight':self.transformer.showo.model.embed_tokens.named_parameters()
            'transformer': self.transformer.named_parameters(),
            'pre_transformer': self.pre_transformer.named_parameters(),
            'caption_embeder': self.caption_embeder.named_parameters()
            }
        return params
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
            
    @classmethod
    def from_pretrained(cls, config_file, pretrained_model_path=None, **kwargs):
        import os
        from omegaconf import OmegaConf
        import json
        if not os.path.exists(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        
        if config_file.endswith('.yaml'):
            config = OmegaConf.load(config_file)['model_kwargs']
            config = OmegaConf.to_container(config)
        elif config_file.endswith('.json'):
            with open(config_file, 'r') as f:
                config = json.load(f)
        model = cls(**config, **kwargs)
        if pretrained_model_path is not None:
            model.init_from_ckpt(pretrained_model_path)
        return model
    
    @classmethod
    def from_checkpoints(cls, pretrained_model_path, subfolder=None, patch_size=2, transformer_additional_kwargs={}):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            if not os.path.isfile(model_file):
                raise RuntimeError(f"{model_file} does not exist")
            state_dict = torch.load(model_file, map_location="cpu")
             
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        
        params = [p.numel() for n, p in model.named_parameters()]
        print(f"### Parameters: {sum(params) / 1e6} M")
        return model
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        print(f'load pretrained weights {path} ...')
        sd = torch.load(path, map_location="cpu")['model']
        tmp_state_dict = {}
        for k in list(sd.keys()):
            keep = True
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
                    keep = False
            if keep:
                tmp_state_dict[f'transformer.{k}'] = sd[k]
        sd = tmp_state_dict
        
        tmp_state_dict = {} 
        for key in list(sd.keys()):
            if key not in self.state_dict().keys():
                print(f'unexpcted key {key}')
                continue
            
            if key in self.state_dict().keys() and self.state_dict()[key].size() == sd[key].size():
                tmp_state_dict[key] = sd[key]
            else:
                print(key, f"Size don't match, skip, expect {self.state_dict()[key].size()} but got {sd[key].size()}")
        sd = tmp_state_dict
        
        m, u = self.load_state_dict(sd, strict=False)
        print(f'missing keys: {m}')
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        
        params = [p.numel() for n, p in self.named_parameters()]
        print(f"### Parameters: {sum(params) / 1e6} M")
    
    def prepare_ground_truth(self, vae, latents):
        patch_thws = self.patch_nums
        num_scales = len(patch_thws)
        f_rest = latents.clone()
        f_hat = torch.zeros_like(f_rest)
        b, c, t, h, w = latents.shape
        embeddings_list = []
        idx_list = []
        SN = len(patch_thws)
        for si, (pt, ph, pw) in enumerate(patch_thws):
            if si != num_scales-1:
                z_NC = F.interpolate(f_rest, size=(pt, ph, pw), mode='area')
            else:
                z_NC = f_rest
            embeddings, idx = vae.features_to_idx_latents(z_NC)
            # embeddings = F.interpolate(embeddings, size=(t, h, w), mode='trilinear') if si != SN-1 else embeddings
            # f_rest = f_rest - embeddings
            embeddings_list.append(embeddings)
            idx_list.append(rearrange(idx, 'b t h w -> b (t h w)'))
        return embeddings_list, idx_list
    
    def token_drop(self, labels, force_drop_ids=None, dropout_prob=0.1):
        """
        Drops labels to enable classifier-free guidance.
        """
        # sample random tokens from the vocabulary
        random_tokens = torch.randint_like(
            labels, low=self.old_vocab_size, high=self.vid_vocab_size + self.old_vocab_size, device=labels.device
        )
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape, device=labels.device) < dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, random_tokens, labels)
        return labels
    
    def prepare_inputs(self, vae, idx_list):
        next_scales = []
        B = idx_list[0].shape[0]
        cur_ps = self.patch_nums[0]
        SN = len(self.patch_nums)
        t, h, w = self.patch_nums[-1]
        c = self.embed_dim

        f_sum = None
        for si in range(SN-1):
            idx = idx_list[si]
            idx = self.token_drop(idx)
            next_ps = self.patch_nums[si + 1]
            embeddings = self.transformer.get_embeddings(idx)
            embeddings = rearrange(embeddings, 'b (t h w) c -> b c t h w', t=cur_ps[0], h=cur_ps[1], w=cur_ps[2])
            embeddings = F.interpolate(embeddings, size=next_ps, mode='trilinear')
            if f_sum is None:
                f_sum = embeddings
            else:
                f_sum = embeddings
                # f_sum = F.interpolate(f_sum, size=next_ps, mode='trilinear') + embeddings
            next_input = rearrange(f_sum, 'b c t h w -> b ( t h w) c')
            next_scales.append(next_input)
            cur_ps = next_ps
        return next_scales
            
    def forward(self, batch):
        
        #text_emb = batch['text_emb']
        text_input_ids = batch['text_input_ids']
        text_emb = self.transformer.get_embeddings(text_input_ids)
        # accelerator = batch['accelerator']
        # if accelerator.is_main_process:
        #     print('main ', vis_tokens.shape)
        # else:
        #     print('minor ', vis_tokens.shape)
        vae = batch['vae']
        # pixel_values = batch['pixel_values']
        # is_image = pixel_values.ndim == 4
        # latents, _ = vae.encode(pixel_values, is_image, include_embeddings=True)
        idx = batch['latent_idx']
        latents = vae.codebook.dictionary_lookup(idx)
        latents = rearrange(latents,'b t h w c -> b c t h w')
        _, target_indices = self.prepare_ground_truth(vae, latents)
        next_scale_input = self.prepare_inputs(vae, [t + self.old_vocab_size for t in target_indices])
        
        B = text_emb.shape[0]
        # B L C, L= sum all scale tokens without text
        text_emb = self.caption_embeder(text_emb, train=self.training)
        # text_emb = text_emb + self.pos_start.expand(B, -1, -1) #start position encoding
        
        # scale_emb = self.scale_embeddings(self.lvl_pos_emb_idx.to(text_emb.device)).expand(B, -1, -1)
        # text_emb[:,-self.num_token_first_scale:,] += scale_emb[:,:self.num_token_first_scale,]
        
        next_scale_input = torch.cat(next_scale_input, dim=1)
        next_scale_input = self.pre_transformer(next_scale_input)
        # next_scale_input += scale_emb[:, self.num_token_first_scale:,]
        
        
        target_indices = torch.cat(target_indices, dim=1)
        
        text_mask = batch['text_mask'] # B 120 
        text_mask[:, -self.num_token_first_scale:] = True #the first level tokens 
        n_text_token = text_mask.shape[1]
        vis_token_mask = self.mask[:, :, self.num_token_first_scale:, self.num_token_first_scale:] #no first level
        vis_token_mask = vis_token_mask.expand(B, -1, -1, -1).to(text_emb.device) # B 1 L-1 L-1
        text_mask = repeat(text_mask, 'b n -> b c L n', c=1, L=text_mask.shape[1]+vis_token_mask.shape[2]) # B 1 L-1+120 120
        pad_mask = torch.zeros((B, 1, n_text_token, vis_token_mask.shape[3])).to(text_emb.device, dtype=torch.bool)
        vis_token_mask = torch.cat([pad_mask, vis_token_mask], dim=2) # # B 1 L-1+120 L-1
        attn_mask = torch.cat([text_mask, vis_token_mask], dim=3) # B 1 L-1+120 L-1+120
        
        max_seq_len = attn_mask.shape[2]
        eye_matrix = repeat(torch.eye(max_seq_len, max_seq_len).to(text_emb.device), 'h w -> B l h w', B=B, l=1)
        assert eye_matrix.shape == attn_mask.shape
        attn_mask = (attn_mask.to(text_emb.dtype) * (1 - eye_matrix) + eye_matrix).to(torch.bool)

        
        assert text_emb[:, self.num_text_tokens-self.num_token_first_scale:].shape[1] + next_scale_input.shape[1] == target_indices.shape[1]
        assert attn_mask.shape[2] == text_emb.shape[1] + next_scale_input.shape[1]
        
        logits, loss = self.transformer(
            input_ids=None,
            input_embeddings=torch.cat([text_emb, next_scale_input], dim=1),
            attention_mask=attn_mask,
            labels=target_indices,
            max_seq_length=text_emb.shape[1],
        )

        # sample_kwargs = {'cfg_scale':1.0, 'temperature':1.0, 'top_k':1000, 'top_p':1.0, 'sample_logits': False}
        # idx, _ = sample_all(logits[:, -target_indices.shape[1]:], **sample_kwargs)
        # acc = (idx==target_indices).float().mean(dim=-1).mean()
        # print('accuracy: ', acc)
        
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        # pred_tokens = pred_tokens[:, prefix_len-1:]
        # pred_tokens = self.post_transformer(pred_tokens)
        # pred_tokens = rearrange(pred_tokens, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        # mean, logvar = pred_tokens.chunk(2, dim=1)
        # posterior = DiagonalGaussianDistribution(mean, logvar)
        return loss
    
    def compute_feat_sum(self, vae, idx_list, latents=None):
    
        SN = len(idx_list)
        t, h, w = self.patch_nums[-1]
        c = self.embed_dim
        
        f_sum = None
        for si in range(SN):
            if si != SN-1:
                idx = idx_list[si]
                embeddings = vae.codebook.dictionary_lookup(idx) # B L C
                next_ps = self.patch_nums[si]
                embeddings = rearrange(embeddings, 'b (t h w) c -> b c t h w', t=next_ps[0], h=next_ps[1], w=next_ps[2])
                embeddings = F.interpolate(embeddings, size=(t, h, w), mode='trilinear')
                if f_sum is None:
                    f_sum = embeddings
                else:
                    f_sum += embeddings
            else:
                idx = idx_list[si]
                embeddings = vae.codebook.dictionary_lookup(idx) # B L C
                next_ps = self.patch_nums[si]
                embeddings = rearrange(embeddings, 'b (t h w) c -> b c t h w', t=next_ps[0], h=next_ps[1], w=next_ps[2])
                f_sum += embeddings
        embeddings, idx = vae.features_to_idx_latents(f_sum)
        if latents is not None:
            print('feature sum mean error', torch.abs(f_sum - latents).mean())
            print('embeddings mean error', torch.abs(embeddings - latents).mean())
            breakpoint()
        idx = rearrange(idx, 'b t h w -> b (t h w)')
        return idx
            
    @torch.no_grad()
    def validate(self, batch, **sampling_kwargs):
        
        # text_emb = batch['text_emb']
        text_input_ids = batch['text_input_ids']
        text_emb = self.transformer.get_embeddings(text_input_ids)
        # accelerator = batch['accelerator']
        # if accelerator.is_main_process:
        #     print('main ', vis_tokens.shape)
        # else:
        #     print('minor ', vis_tokens.shape)
        vae = batch['vae']
        # pixel_values = batch['pixel_values']
        # is_image = pixel_values.ndim == 4
        # latents, _ = vae.encode(pixel_values, is_image, include_embeddings=True)
        idx = batch['latent_idx']
        latents = vae.codebook.dictionary_lookup(idx)
        latents = rearrange(latents,'b t h w c -> b c t h w')
        _, target_indices = self.prepare_ground_truth(vae, latents)
        next_scale_input = self.prepare_inputs(vae, [t + self.old_vocab_size for t in target_indices])
 
        B = text_emb.shape[0]
        # B L C, L= sum all scale tokens without text
        text_emb = self.caption_embeder(text_emb, train=self.training)
        # text_emb = text_emb + self.pos_start.expand(B, -1, -1)
        # scale_emb = self.scale_embeddings(self.lvl_pos_emb_idx.to(text_emb.device)).expand(B, -1, -1)
        # text_emb[:,-self.num_token_first_scale:,] += scale_emb[:,:self.num_token_first_scale,]
        
        next_scale_input = torch.cat(next_scale_input, dim=1)
        next_scale_input = self.pre_transformer(next_scale_input)
        # next_scale_input += scale_emb[:, self.num_token_first_scale:,]
        
        target_indices = torch.cat(target_indices, dim=1)
        
        text_mask = batch['text_mask'] # B 120 
        text_mask[:, -self.num_token_first_scale:] = True #the first level tokens 
        n_text_token = text_mask.shape[1]
        vis_token_mask = self.mask[:, :, self.num_token_first_scale:, self.num_token_first_scale:] #no first level
        vis_token_mask = vis_token_mask.expand(B, -1, -1, -1).to(text_emb.device) # B 1 L-1 L-1
        text_mask = repeat(text_mask, 'b n -> b c L n', c=1, L=text_mask.shape[1]+vis_token_mask.shape[2]) # B 1 L-1+120 120
        pad_mask = torch.zeros((B, 1, n_text_token, vis_token_mask.shape[3])).to(text_emb.device, dtype=torch.bool)
        vis_token_mask = torch.cat([pad_mask, vis_token_mask], dim=2) # # B 1 L-1+120 L-1
        attn_mask = torch.cat([text_mask, vis_token_mask], dim=3) # B 1 L-1+120 L-1+120
        
        max_seq_len = attn_mask.shape[2]
        eye_matrix = repeat(torch.eye(max_seq_len, max_seq_len).to(text_emb.device), 'h w -> B l h w', B=B, l=1)
        assert eye_matrix.shape == attn_mask.shape
        attn_mask = (attn_mask.to(text_emb.dtype) * (1 - eye_matrix) + eye_matrix).to(torch.bool)
        

        assert text_emb[:, self.num_text_tokens-self.num_token_first_scale:].shape[1] + next_scale_input.shape[1] == target_indices.shape[1]
        assert attn_mask.shape[2] == text_emb.shape[1] + next_scale_input.shape[1]
        
        logits = self.transformer(
            input_ids=None,
            input_embeddings=torch.cat([text_emb, next_scale_input], dim=1),
            attention_mask=attn_mask,
            labels=None,
            max_seq_length=text_emb.shape[1],
        )

        # logits = logits[:, :, self.old_vocab_size:]
        last_scale = self.patch_nums[-1]
        t, h, w = last_scale
        num_token_last_scale = np.prod(last_scale)

        sampling_kwargs['sample_logits'] = False
        all_idx, _ = sample_all(logits[:, -target_indices.shape[1]:], **sampling_kwargs)
        acc = (all_idx==target_indices).float().mean(dim=-1).mean()
        print('accuracy: ', acc)

        last_logits = logits[:, -self.L:]
        sampling_kwargs['sample_logits'] = True
        tmp_sample_idx, _ = sample_all(last_logits, **sampling_kwargs)
        cur_pos = 0
        sample_idx = []
        for ps in self.patch_nums:
            cur_scale_num_token = np.prod(ps)
            sample_idx.append(tmp_sample_idx[:, cur_pos:cur_pos+cur_scale_num_token])
            cur_pos += cur_scale_num_token
        #sample_idx = self.compute_feat_sum(vae, sample_idx)
        sample_idx = sample_idx[-1]
        
        sampling_kwargs['sample_logits'] = False
        tmp_idx, _ = sample_all(last_logits, **sampling_kwargs)        
        cur_pos = 0
        idx = []
        for ps in self.patch_nums:
            cur_scale_num_token = np.prod(ps)
            idx.append(tmp_idx[:, cur_pos:cur_pos+cur_scale_num_token])
            cur_pos += cur_scale_num_token
        # idx = self.compute_feat_sum(vae, idx)
        idx = idx[-1]
        
        target_idx = []
        cur_pos = 0
        for ps in self.patch_nums:
            cur_scale_num_token = np.prod(ps)
            target_idx.append(target_indices[:, cur_pos:cur_pos+cur_scale_num_token])
            cur_pos += cur_scale_num_token
        # target_idx = self.compute_feat_sum(vae, target_idx, latents)
        target_idx = target_idx[-1]
        
        last_token_idx = torch.cat([target_idx[:,:-500], sample_idx[:,-500:]], dim=1)
        # last_token_latent = torch.cat([vis_tokens[:,:-100], last_token_latent[:,-100:]], dim=1) 
        idx = rearrange(idx, 'b (t h w) -> b t h w', t=t, h=h, w=w)
        sample_idx = rearrange(sample_idx, 'b (t h w) -> b t h w', t=t, h=h, w=w)
        last_token_idx = rearrange(last_token_idx, 'b (t h w) -> b t h w', t=t, h=h, w=w)
        orig_idx = rearrange(target_idx, 'b (t h w) -> b t h w', t=t, h=h, w=w)
        return idx, sample_idx, last_token_idx, orig_idx

    @torch.no_grad()
    def validate_per_token(self, batch, sampling_kwargs):
        
        target_indices = None
        sampling_kwargs['sample_logits'] = False
        latents = None
        
        text_input_ids = batch['text_input_ids']
        text_emb = self.transformer.get_embeddings(text_input_ids)
        vae = batch['vae']
        
        # breakpoint()
        # latents_post = vae.post_vq_conv(latents)
        # vid = vae.decoder(latents_post, False)
        # vid = torch.clamp(vid + 0.5, 0, 1).float().cpu()
        # save_videos_grid(vid, os.path.join("./", f"latents2video-test.gif"))
        # text_emb = batch['text_emb']
        
        B = text_emb.shape[0]
        # B L C, L= sum all scale tokens without text
        text_emb = self.caption_embeder(text_emb, train=self.training)
        # text_emb = text_emb + self.pos_start.expand(B, -1, -1)
        
        # scale_emb = self.scale_embeddings(self.lvl_pos_emb_idx.to(text_emb.device)).expand(B, -1, -1)
        # text_emb[:, -self.num_token_first_scale:] += scale_emb[:,:self.num_token_first_scale]
        
        text_mask = batch['text_mask'] # B 120
        text_mask[:, -self.num_token_first_scale:] = True #the first level tokens  
        n_text_token = text_mask.shape[1]
        vis_token_mask = self.mask[:, :, self.num_token_first_scale:, self.num_token_first_scale:] #no first level
        vis_token_mask = vis_token_mask.expand(B, -1, -1, -1).to(text_emb.device) # B 1 L-1 L-1
        text_mask = repeat(text_mask, 'b n -> b c L n', c=1, L=text_mask.shape[1]+vis_token_mask.shape[2]) # B 1 L-1+120 120
        pad_mask = torch.zeros((B, 1, n_text_token, vis_token_mask.shape[3])).to(text_emb.device, dtype=torch.bool)
        vis_token_mask = torch.cat([pad_mask, vis_token_mask], dim=2) # # B 1 L-1+120 L-1
        attn_mask = torch.cat([text_mask, vis_token_mask], dim=3) # B 1 L-1+120 L-1+120
        
        max_seq_len = attn_mask.shape[2]
        eye_matrix = repeat(torch.eye(max_seq_len, max_seq_len).to(text_emb.device), 'h w -> B l h w', B=B, l=1)
        assert eye_matrix.shape == attn_mask.shape
        attn_mask = (attn_mask.to(text_emb.dtype) * (1 - eye_matrix) + eye_matrix).to(torch.bool)
        
        vae = batch['vae']
        t, h, w = self.patch_nums[-1]
        
        pred_token_idx = []
        
        next_scale_input = None
        cur_scale_token_idx = None
        cur_token_pos = 0
        f_sum = None
        for si in range(len(self.patch_nums)):
            
            ps = self.patch_nums[si]
            num_token_cur_scale = np.prod(ps)
            mask_len = text_emb.shape[1]
            
            if cur_scale_token_idx is not None:
                # embeddings = vae.codebook.dictionary_lookup(cur_scale_token_idx) # B L C
                embeddings = self.transformer.get_embeddings(cur_scale_token_idx + self.old_vocab_size)
                last_ps = self.patch_nums[si - 1]
                embeddings = rearrange(embeddings, 'b (t h w) c -> b c t h w', t=last_ps[0], h=last_ps[1], w=last_ps[2])
                embeddings = F.interpolate(embeddings, size=ps, mode='trilinear')
                if f_sum is None:
                    f_sum = embeddings
                else:
                    f_sum = embeddings # F.interpolate(f_sum, size=ps, mode='trilinear') +
                next_input = rearrange(f_sum, 'b c t h w -> b (t h w) c')
                next_input = self.pre_transformer(next_input)
                # next_input += scale_emb[:, cur_token_pos:cur_token_pos+num_token_cur_scale]
                
                if next_scale_input is None:
                    next_scale_input = next_input
                else:
                    next_scale_input = torch.cat([next_scale_input, next_input], dim=1)    
                mask_len = text_emb.shape[1] + next_scale_input.shape[1]      
                
            logits = self.transformer(
                                input_ids=None,
                                input_embeddings=torch.cat([text_emb, next_scale_input], dim=1) if next_scale_input is not None else text_emb,
                                attention_mask=attn_mask[:, :, :mask_len,:mask_len],
                                labels=None,
                                max_seq_length=text_emb.shape[1],
                            )

            #logits = logits[:, :, self.old_vocab_size:]
            logits = logits[:, -num_token_cur_scale:]
            cur_scale_token_idx, _ = sample_all(logits, **sampling_kwargs)
            #breakpoint()
            pred_token_idx.append(cur_scale_token_idx)
            cur_token_pos += num_token_cur_scale
        
        
        all_pred_idx = torch.cat(pred_token_idx, dim=1)
        if target_indices is not None:
            target_indices = torch.cat(target_indices, dim=1)
            acc = (all_pred_idx[:, -num_token_cur_scale:]==target_indices[:, -num_token_cur_scale:]).float().mean(dim=-1).mean()
            print('accuracy: ', acc)

        # embeddings, idx = vae.features_to_idx_latents(f_sum)
        idx = rearrange(pred_token_idx[-1], 'b (t h w) -> b t h w', t=t, h=h, w=w)
        return idx
    
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out


    @torch.no_grad()
    def generate(self, text_emb, text_emb_uncond, target_size, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
        assert self.training is False
        t, h, w = target_size
        max_new_tokens = np.prod(target_size)
        
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(text_emb) + self.transformer.cls_embedding.uncond_embedding
            cond_combined = torch.cat([text_emb, cond_null])
        else:
            cond_combined = text_emb
        T = text_emb.shape[1]      
        T_new = T + max_new_tokens
        max_seq_length = T_new
        max_batch_size = text_emb.shape[0]

        device = text_emb.device
        with torch.device(device):
            max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
            self.transformer.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, 
                                          dtype=text_emb.dtype)
        
        if emb_masks is not None:
            assert emb_masks.shape[0] == max_batch_size
            assert emb_masks.shape[-1] == T
            if cfg_scale > 1.0:
                self.transformer.causal_mask[:, :, :T] = self.transformer.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
            else:
                self.transformer.causal_mask[:, :, :T] = self.transformer.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

            eye_matrix = torch.eye(self.transformer.causal_mask.size(1), self.transformer.causal_mask.size(2), device=device)
            self.transformer.causal_mask[:] = self.transformer.causal_mask * (1 - eye_matrix) + eye_matrix
        
        # create an empty tensor of the expected final shape and fill in the current tokens
        seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
        pred_tokens = []
        input_pos = torch.arange(0, T, device=device)
        next_token = self.prefill(cond_combined, input_pos, cfg_scale, **sampling_kwargs)
        seq[:, T:T+1] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        generated_tokens = self.decode_n_tokens(next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
        seq[:, T+1:] = torch.cat(generated_tokens, dim=1)
        self.transformer.clear_cache()
        
        return seq[:, T:]

    @torch.no_grad()
    def decode_n_tokens(
        self, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
        cfg_scale: float, cfg_interval: int, **sampling_kwargs):
        new_tokens, new_probs = [], []
        cfg_flag = True
        for i in range(num_new_tokens):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
                if cfg_interval > -1 and i > cfg_interval:
                    cfg_flag = False
                next_token, _ = self.decode_one_token(
                    cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
                )
                input_pos += 1
                new_tokens.append(next_token.clone())
                cur_token = next_token.view(-1, 1)
        return new_tokens
    
    @torch.no_grad()
    def decode_one_token(self, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
        assert input_pos.shape[-1] == 1
        if cfg_scale > 1.0:
            x_combined = torch.cat([x, x])
            logits, _, _ = self.transformer(idx=x_combined, cond_idx=None, token_embeddings=None, input_pos=input_pos)
            logits_combined = logits
            cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
            if cfg_flag:
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = cond_logits
        else:
            logits, _, _ = self.transformer(idx=x, cond_idx=None, token_embeddings=None, input_pos=input_pos)
        token = sample(logits, **sampling_kwargs)
        return token

    @torch.no_grad()
    def prefill(self, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
        if cfg_scale > 1.0:
            logits, _, _ = self.transformer(None, cond_idx, token_embeddings=None, input_pos=input_pos)
            logits_combined = logits
            cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits, _, _ = self.transformer(None, cond_idx, token_embeddings=None, input_pos=input_pos)
        
        return sample(logits, **sampling_kwargs)[0]

    @torch.no_grad()
    def post_project_token(self, tokens: torch.Tensor, cfg_scale: float):
        tokens = self.post_transformer(tokens)
        if cfg_scale > 1.0:
            tokens_cond, tokens_uncond = torch.split(tokens, len(tokens) // 2, dim=0)
            tokens = tokens_uncond + (tokens_cond - tokens_cond) * cfg_scale # b 1 vae_dim*2
        
        mean, logvar = tokens.chunk(2, dim=2)
        token_dist = DiagonalGaussianDistribution(mean, logvar)
        token = token_dist.mode()
        return token # b 1 vae_dim
            
            
    def smooth_output(self, vae, video, mini_batch_encoder, mini_batch_decoder):
        if video.size()[2] <= mini_batch_encoder:
            return video
        prefix_index_before = mini_batch_encoder // 2
        prefix_index_after = mini_batch_encoder - prefix_index_before
        pixel_values = video[:, :, prefix_index_before:-prefix_index_after]
        
        if vae.slice_compression_vae:
            latents = vae.encode(pixel_values)[0]
            latents = latents.sample()
        else:
            new_pixel_values = []
            for i in range(0, pixel_values.shape[2], mini_batch_encoder):
                with torch.no_grad():
                    pixel_values_bs = pixel_values[:, :, i: i + mini_batch_encoder, :, :]
                    pixel_values_bs = vae.encode(pixel_values_bs)[0]
                    pixel_values_bs = pixel_values_bs.sample()
                    new_pixel_values.append(pixel_values_bs)
            latents = torch.cat(new_pixel_values, dim = 2)
                
        if vae.slice_compression_vae:
            middle_video = vae.decode(latents)[0]
        else:
            middle_video = []
            for i in range(0, latents.shape[2], mini_batch_decoder):
                with torch.no_grad():
                    start_index = i
                    end_index = i + mini_batch_decoder
                    latents_bs = vae.decode(latents[:, :, start_index:end_index, :, :])[0]
                    middle_video.append(latents_bs)
            middle_video = torch.cat(middle_video, 2)
        video[:, :, prefix_index_before:-prefix_index_after] = (video[:, :, prefix_index_before:-prefix_index_after] + middle_video) / 2
        return video
    
    def decode_latents(self, latents, vae):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        if vae.quant_conv.weight.ndim==5:
            mini_batch_encoder = vae.mini_batch_encoder
            mini_batch_decoder = vae.mini_batch_decoder
            if vae.slice_compression_vae:
                video = vae.decode(latents)[0]
            else:
                video = []
                for i in range(0, latents.shape[2], mini_batch_decoder):
                    with torch.no_grad():
                        start_index = i
                        end_index = i + mini_batch_decoder
                        latents_bs = vae.decode(latents[:, :, start_index:end_index, :, :])[0]
                        video.append(latents_bs)
                video = torch.cat(video, 2)

            video = video.clamp(-1, 1)
            video = self.smooth_output(vae, video, mini_batch_encoder, mini_batch_decoder).cpu().clamp(-1, 1)
        else:
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            video = []
            for frame_idx in tqdm(range(latents.shape[0])):
                video.append(vae.decode(latents[frame_idx:frame_idx+1]).sample)
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float()
        return video
    


