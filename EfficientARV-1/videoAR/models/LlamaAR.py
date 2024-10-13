import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange
from OmniTokenizer.utils import shift_dim, accuracy, comp_getattr, ForkedPdb
from .llama_gpt import GPT_models
from OmniTokenizer.modules.encoders import Labelator, SOSProvider, Identity
import numpy as np
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import Optional
from tqdm import tqdm
import os
import json

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
                sample_every_n_latent_frames,
                hidden_dim,
                max_frames,
                ckpt_path=None,
                ignore_keys=[],
                sample_size=64,
                caption_dim=2048,
                vae_dim=512,
                ):
        super().__init__()
      
        self.sample_size = sample_size  
        self.sample_every_n_latent_frames = sample_every_n_latent_frames
        
        self.transformer = GPT_models[gpt_model](
                vocab_size=vocab_size,
                block_size=latent_size ** 2,
                num_classes=num_classes,
                cls_token_num=cls_token_num,
                model_type=gpt_type,
                resid_dropout_p=resid_dropout_p,
                ffn_dropout_p=ffn_dropout_p,
                token_dropout_p=token_dropout_p,
                caption_dim=caption_dim,
                max_frames=max_frames
                )
        
        # self.caption_embeder = CaptionEmbedder(caption_dim, hidden_dim, class_dropout_prob)
        self.pre_transformer = MLP(in_features=vae_dim, hidden_features=hidden_dim, out_features=hidden_dim)
        self.post_transformer = MLP(in_features=hidden_dim, hidden_features=2*vae_dim, out_features=2*vae_dim)
        self.apply(self._init_weights)
        
        if ckpt_path is not None and len(ckpt_path)>0:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
        # Zero-out output layers:
        # nn.init.constant_(self.output.weight, 0)

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

    def forward(self, batch):
        
        text_emb = batch['text_emb']
        vis_tokens = batch['latents']
        attn_mask = batch['mask']
        # accelerator = batch['accelerator']
        # if accelerator.is_main_process:
        #     print('main ', vis_tokens.shape)
        # else:
        #     print('minor ', vis_tokens.shape)
        b, c, t, h, w = vis_tokens.shape
        vis_tokens = rearrange(vis_tokens, 'b c t h w -> b (t h w) c')
        prefix_len = text_emb.shape[1]
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        # text_emb = self.caption_embeder(text_emb, train=self.training)
        vis_tokens = self.pre_transformer(vis_tokens)
        # token_embedings = torch.cat([text_emb, vis_tokens], dim=1)
        # make the prediction
        _, _, pred_tokens, = self.transformer(idx=None, cond_idx=text_emb, 
                                              token_embeddings=vis_tokens[:, :-1],
                                              mask=attn_mask[:, :, :-1,:-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        pred_tokens = pred_tokens[:, prefix_len-1:]
        pred_tokens = self.post_transformer(pred_tokens)
        pred_tokens = rearrange(pred_tokens, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        mean, logvar = pred_tokens.chunk(2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)

        return posterior
    
    @torch.no_grad()
    def validate(self, batch):
        text_emb = batch['text_emb']
        vis_tokens = batch['latents']
        attn_mask = batch['mask']
        # accelerator = batch['accelerator']
        # if accelerator.is_main_process:
        #     print('main ', vis_tokens.shape)
        # else:
        #     print('minor ', vis_tokens.shape)
        b, c, t, h, w = vis_tokens.shape
        vis_tokens = rearrange(vis_tokens, 'b c t h w -> b (t h w) c')
        prefix_len = text_emb.shape[1]
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        # text_emb = self.caption_embeder(text_emb, train=self.training)
        vis_tokens_embedding = self.pre_transformer(vis_tokens)
        # token_embedings = torch.cat([text_emb, vis_tokens], dim=1)
        # make the prediction
        _, _, pred_tokens, = self.transformer(idx=None, cond_idx=text_emb, token_embeddings=vis_tokens_embedding[:, :-1], mask=attn_mask[:, :, :-1,:-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        pred_tokens = pred_tokens[:, prefix_len-1:]
        pred_tokens = self.post_transformer(pred_tokens)
        pred_tokens = rearrange(pred_tokens, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        mean, logvar = pred_tokens.chunk(2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)

        last_token_latent = rearrange(mean, 'b c t h w -> b (t h w) c')
        last_token_latent = torch.cat([vis_tokens[:,:-100], last_token_latent[:,-100:]], dim=1) 
        last_token_latent = rearrange(last_token_latent, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        
        mean_images = mean
        sample_images = posterior.sample()
        return mean_images, sample_images, last_token_latent

    @torch.no_grad()
    def validate_per_token(self, batch):
        text_emb = batch['text_emb']
        attn_mask = batch['mask']
        # accelerator = batch['accelerator']
        # if accelerator.is_main_process:
        #     print('main ', vis_tokens.shape)
        # else:
        #     print('minor ', vis_tokens.shape)
        t, h, w = batch['target_size']
        # vis_tokens = rearrange(vis_tokens, 'b c t h w -> b (t h w) c')
        prefix_len = text_emb.shape[1]
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        # text_emb = self.caption_embeder(text_emb, train=self.training)
        num_tokens = int(t*h*w)
        pred_tokens = []
        next_token = None
        for i in range(num_tokens):
            if next_token is not None:
                vis_tokens_embedding = self.pre_transformer(torch.cat(pred_tokens, dim=1))
                mask_len = vis_tokens_embedding.shape[1] + prefix_len
            else:
                vis_tokens_embedding = None
                mask_len = prefix_len
            # token_embedings = torch.cat([text_emb, vis_tokens], dim=1)
            # make the prediction
            _, _, pred_token, = self.transformer(idx=None, cond_idx=text_emb, token_embeddings=vis_tokens_embedding, mask=attn_mask[:, :, :mask_len,:mask_len])
            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            pred_token = pred_token[:, -1:]
            pred_token = self.post_transformer(pred_token)

            mean, logvar = pred_token.chunk(2, dim=2)
            posterior = DiagonalGaussianDistribution(mean, logvar)
            pred_tokens.append(mean)
            next_token = mean
            
        # sample_images = posterior.sample()
        mean_images = torch.cat(pred_tokens, dim=1)
        mean_images = rearrange(mean_images, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        return mean_images
    
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
        # seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
        pred_tokens = []
        input_pos = torch.arange(0, T, device=device)
        next_token = self.prefill(cond_combined, input_pos, cfg_scale, **sampling_kwargs)
        pred_tokens.append(next_token)

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        generated_tokens = self.decode_n_tokens(next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
        pred_tokens.extend(generated_tokens)
        pred_tokens = torch.cat(pred_tokens, dim=1)
        latents = rearrange(pred_tokens, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        self.transformer.clear_cache()
        
        return latents

    @torch.no_grad()
    def decode_n_tokens(
        self, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
        cfg_scale: float, cfg_interval: int, **sampling_kwargs):
        new_tokens, new_probs = [], []
        cfg_flag = True
        for i in range(num_new_tokens):
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token= self.decode_one_token(
                cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            cur_token = next_token
        return new_tokens
    
    @torch.no_grad()
    def decode_one_token(self, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
        assert input_pos.shape[-1] == 1
        if cfg_scale > 1.0:
            x_combined = torch.cat([x, x])
            x_combined = self.pre_transformer(x_combined)
            _, _, h = self.transformer(None, cond_idx=None, token_embeddings=x_combined, input_pos=input_pos)
        else:
            x = self.pre_transformer(x)
            _, _, h = self.transformer(None, cond_idx=None, token_embeddings=x, input_pos=input_pos)
        token = self.post_project_token(h[:, -1:], cfg_scale)
        return token

    @torch.no_grad()
    def prefill(self, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
        if cfg_scale > 1.0:
            _, _, h = self.transformer(None, cond_idx, token_embeddings=None, input_pos=input_pos)
        else:
            _, _, h = self.transformer(None, cond_idx, token_embeddings=None, input_pos=input_pos)
        token = self.post_project_token(h[:, -1:], cfg_scale)
        return token

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
    


