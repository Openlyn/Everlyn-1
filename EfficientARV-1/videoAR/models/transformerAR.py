import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange
from OmniTokenizer.utils import shift_dim, accuracy, comp_getattr, ForkedPdb
from .gpt import GPT, top_k_top_p_filtering, sample_with_past_cfg
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
        
class Net2NetTransformer(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 args_dict,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="video",
                 cond_stage_key="label",
                 pkeep=1.0,
                 sos_token=0,
                 vqvae_ckpt=None,
                 sample_size=64,
                 caption_dim=2048,
                 class_dropout_prob=0.1,
                 vae_dim=512
                 ):
        super().__init__()
        self.class_cond_dim = args_dict['class_cond_dim']
        self.be_unconditional = args_dict['unconditional']
        self.sos_token = sos_token
        
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.vtokens = args_dict['vtokens']
        self.sample_every_n_latent_frames = args_dict['sample_every_n_latent_frames'] if 'sample_every_n_latent_frames' in args_dict else 0
        
        
        self.starts_with_sos = args_dict['starts_with_sos']
        self.sos_provider = SOSProvider(self.sos_token)
        self.p_drop_cond = args_dict['p_drop_cond']
        self.class_first = args_dict['class_first']

        if self.be_unconditional:
            self.starts_with_sos = False

        self.cond_stage_vocab_size = args_dict['class_cond_dim']
        self.first_stage_vocab_size = args_dict['vocab_size']
        gpt_vocab_size = self.first_stage_vocab_size + self.cond_stage_vocab_size
        if self.starts_with_sos:
            gpt_vocab_size += 1
        
        self.transformer = GPT(gpt_vocab_size, args_dict['block_size'], n_layer=args_dict['n_layer'], 
                            n_head=args_dict['n_head'], n_embd=args_dict['n_embd'], vtokens_pos=args_dict['vtokens_pos'], 
                            n_unmasked=args_dict['n_unmasked'], 
                            embd_pdrop=args_dict['transformer_dropout'], resid_pdrop=args_dict['transformer_dropout'], 
                            attn_pdrop=args_dict['transformer_dropout'])
        
        self.caption_embeder = CaptionEmbedder(caption_dim, args_dict['n_embd'], class_dropout_prob)
        self.pre_transformer = MLP(in_features=vae_dim, hidden_features=args_dict['n_embd'], out_features=args_dict['n_embd'])
        self.post_transformer = MLP(in_features=args_dict['n_embd'], hidden_features=2*vae_dim, out_features=2*vae_dim)
        self.post_quant_conv = nn.Conv3d(2*vae_dim, 2*vae_dim, kernel_size=1)
        
        if ckpt_path is not None and len(ckpt_path)>0:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
        self.pkeep = pkeep
        

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
            
    @classmethod
    def from_pretrained(cls, config_file, pretrained_model_path=None):
        import os
        from omegaconf import OmegaConf
        if not os.path.exists(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        
        config = OmegaConf.load(config_file)['model_kwargs']
        gpt_args = config.pop('gpt_args', OmegaConf.create())
        model = cls(OmegaConf.to_container(gpt_args), **OmegaConf.to_container(config))
        if pretrained_model_path is not None:
            model.init_from_ckpt(pretrained_model_path)
        return model
        
    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, patch_size=2, transformer_additional_kwargs={}):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

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
        
        params = [p.numel() if "attn_temporal." in n else 0 for n, p in model.named_parameters()]
        print(f"### Attn temporal Parameters: {sum(params) / 1e6} M")
        
        return model
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        print(f'load pretrained weights {path} ...')
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        
        tmp_state_dict = {} 
        for key in list(sd.keys()):
            if key in self.state_dict().keys() and self.state_dict()[key].size() == sd[key].size():
                tmp_state_dict[key] = sd[key]
            else:
                print(key, "Size don't match, skip")
        sd = tmp_state_dict
        
        m, u = self.load_state_dict(sd, strict=False)
        print(f'missing keys: {m}')
        print(f'unexpected keys: {u}')
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        
        params = [p.numel() for n, p in self.named_parameters()]
        print(f"### Parameters: {sum(params) / 1e6} M")


    def forward(self, batch, cbox=None):
        text_emb = batch['text_emb']
        vis_tokens = batch['latents']
        b, c, t, h, w = vis_tokens.shape
        vis_tokens = rearrange(vis_tokens, 'b c t h w -> b (t h w) c')
        prefix_len = text_emb.shape[1]
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        text_emb = self.caption_embeder(text_emb, train=self.training)
        vis_tokens = self.pre_transformer(vis_tokens)
        token_embedings = torch.cat([text_emb, vis_tokens], dim=1)
        # make the prediction
        _, pred_tokens, _ = self.transformer(embeddings=token_embedings)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        pred_tokens = pred_tokens[:, prefix_len:]
        pred_tokens = self.post_transformer(pred_tokens)
        pred_tokens = rearrange(pred_tokens, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        mean, logvar = pred_tokens.chunk(2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)

        return posterior
        

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out


    @torch.no_grad()
    def sample(self, steps, text_emb, text_emb_uncond, scale_cfg=False, cfg_ratio=0.5):

        text_emb = self.caption_embeder(text_emb, train=False)
        text_emb_uncond = self.caption_embeder(text_emb_uncond, train=False)
        # x is conditioning
        sample = text_emb
        x = sample
        cond_len = sample.shape[1]
        past = None
        
        sample_uncond = text_emb_uncond
        past_uncond = None
        x_uncond = sample_uncond

        latents = []
        for n in range(steps):
            ratio = n if scale_cfg else 1
            _, x, present = self.transformer.forward_with_past(embeddings=x, past=past,
                                                                    past_length=(n+cond_len-1))
            _, x_uncond, present_uncond = self.transformer.forward_with_past(embeddings=x_uncond, past=past_uncond, 
                                                                past_length=(n+cond_len-2), forward_uncond=True) 
            
            if past is None:
                past = [present]
                past_uncond = [present_uncond]
                # past_uncond = [present]
                # past_uncond = [present[:, :, :, :, :1, :]]
            else:
                past.append(present)
                past_uncond.append(present_uncond)
                # past_uncond.append(present_uncond)

            x = x[:, -1:, :]
            x_uncond = x_uncond[:, -1:, :]
            
            x = self.post_transformer(x)
            x_uncond = self.post_transformer(x_uncond)
            
            t = cfg_ratio * ratio
            x = (1 + t) * x - t * x_uncond # b 1 vae_dim*2
            latents.append(x)
            
            x, _ = x.chunk(2, dim=2) # b 1 vae_dim
            x = self.pre_transformer(x)
            x_uncond = x

        del past
        latents = torch.cat(latents, dim=1)
        return latents

    @torch.no_grad()
    def generate(self, vae, steps, text_emb, text_emb_uncond, target_size, scale_cfg=False, cfg_ratio=0.5):
        t, h, w = target_size
        latents = self.sample(steps, text_emb, text_emb_uncond, scale_cfg, cfg_ratio)
        latents = rearrange(latents, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        mean, logvar = latents.chunk(2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        sample = posterior.sample()
        video = self.decode_latents(sample, vae)
        return video
    
    @torch.no_grad()
    def encode_to_z(self, x, is_image):
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
        else:
            x, targets = self.first_stage_model.encode(x, is_image, include_embeddings=True)
            if self.sample_every_n_latent_frames > 0:
                x = x[:, :, ::self.sample_every_n_latent_frames]
                targets = targets[:, ::self.sample_every_n_latent_frames]
            x = shift_dim(x, 1, -1)
            targets = targets.reshape(targets.shape[0], -1)
        return x, targets

    @torch.no_grad()
    def encode_to_c(self, c, is_image):
        if isinstance(self.cond_stage_model, Labelator) or isinstance(self.cond_stage_model, SOSProvider):
            quant_c, indices = self.cond_stage_model.encode(c)
        else:
            quant_c, indices = self.cond_stage_model.encode(c, is_image, include_embeddings=True)
        
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    def get_input(self, key, batch):
        x = batch[key]
        # if x.dtype == torch.double:
            # x = x.float()
        return x

    def get_xc(self, batch, N=None):
        """x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]"""
        if isinstance(batch, dict):
            x = batch[self.first_stage_key]
            c = batch[self.cond_stage_key]
        
        else:
            assert isinstance(batch, list) and len(batch) == 1
            x = batch[0][self.first_stage_key]
            c = batch[0][self.cond_stage_key]

        if N is not None:
            x = x[:N]
            c = c[:N]
        
        return x, c

    def shared_step(self, batch, batch_idx):
        if not self.vtokens:
            self.first_stage_model.eval()
        x, c = self.get_xc(batch)
        if self.args.vtokens_pos:
            cbox = batch['cbox']
        else:
            cbox = None
        
        logits, target = self(x, c, cbox)
        # print(logits.shape, target.shape)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        acc1, acc5 = accuracy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), topk=(1, 5))
        return loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        sch = self.lr_schedulers()
        opt = self.optimizers()

        loss, acc1, acc5 = self.shared_step(batch, batch_idx)
        # print(batch_idx, loss)

        self.manual_backward(loss)

        cur_global_step = self.global_step
        if (cur_global_step + 1) % self.grad_accumulates == 0:
            if self.grad_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=self.grad_clip_val)
                
            opt.step()
            
            sch.step(cur_global_step)
            opt.zero_grad()

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        if self.args.vtokens_pos:
            no_decay.add('vtokens_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

        lr_min = self.args.lr_min
        train_iters = self.args.max_steps
        warmup_steps = self.args.warmup_steps
        warmup_lr_init = self.args.warmup_lr_init

       
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min = lr_min,
            t_initial = train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_steps,
            cycle_mul = 1.,
            cycle_limit=1,
            t_in_epochs=True,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

    def log_images(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]

        logits, _ = self(x, c)
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        
        index = torch.clamp(ix-self.cond_stage_vocab_size, min=0, max=self.first_stage_vocab_size-1).squeeze(-1)
        predict = self.first_stage_model.decode(index, is_image=(x.ndim==4))
        
        is_image = x.ndim == 4
        _, targets = self.first_stage_model.encode(x, is_image, include_embeddings=True)
        x_recon = self.first_stage_model.decode(targets, is_image=is_image)
        
        log["inputs"] = x
        log["predict"] = predict
        log["recon"] = x_recon
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]
        
        top_k = 4096
        top_p = 0.9
        sample_logits = True
        cfg_ratio = 0.5
        class_first = False
        scale_cfg = False
        temperature = 1.0
        latent_shape = [
            (17 - 1) // 4 + 1, 256 // 8, 256 // 8
        ]
        steps = np.prod(latent_shape)
        is_image = x.ndim == 4
        # logits, targets = self(x, c)   
        # B, L, C = logits.shape
        # logits = rearrange(logits, 'B L C->(B L) C')
        # if top_k is not None:
        #     logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        
        # probs = F.softmax(logits, dim=-1)
        
        # if not sample_logits:
        #     _, ix = torch.topk(probs, k=1, dim=-1)
        # else:
        #     ix = torch.multinomial(probs, num_samples=1)
        # ix = rearrange(ix, '(b l) c -> b l c', b=B, l=L).squeeze(-1)
        # ix = torch.cat([targets[:,:-1], ix[:,-1:]], dim=1)

        cn = torch.tensor([0]).to(c.device)
        _, c_indices = self.encode_to_c(cn, is_image)
        ix = sample_with_past_cfg(c_indices, self.transformer, steps=steps,
                            sample_logits=False, top_k=top_k, callback=None,
                            temperature=temperature, top_p=top_p, cfg_ratio=cfg_ratio, class_first=class_first, scale_cfg=scale_cfg) 
        # breakpoint()
        n_cond = self.cond_stage_vocab_size if not self.starts_with_sos else self.cond_stage_vocab_size + 1
        index = torch.clamp(ix-n_cond, min=0, max=self.first_stage_vocab_size-1)
        predict = self.first_stage_model.decode(index, is_image=is_image)
        
        
        _, targets = self.first_stage_model.encode(x, is_image, include_embeddings=True)
        x_recon = self.first_stage_model.decode(targets, is_image=is_image)

        log["inputs"] = x
        log["recon"] = x_recon
        log["predict"] = predict
        return log


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
                    pixel_values_bs = self.vae.encode(pixel_values_bs)[0]
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
        video = video.cpu().float().numpy()
        return video
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--stft_vqvae', type=str, help='path to vqgan ckpt, or model name to download pretrained')
        parser.add_argument('--unconditional', action='store_true')
        parser.add_argument('--base_lr', type=float, default=4.5e-06)
        # VideoGPT hyperparmeters
        parser.add_argument('--vocab_size', type=int, default=16384)
        parser.add_argument('--first_stage_vocab_size', type=int, default=16384)
        
        parser.add_argument('--starts_with_sos', action="store_true")
        parser.add_argument('--p_drop_cond', type=float, default=None)
        parser.add_argument('--class_first', action="store_true")
        parser.add_argument('--block_size', type=int, default=256)
        parser.add_argument('--n_layer', type=int, default=48)
        parser.add_argument('--n_head', type=int, default=24)
        parser.add_argument('--n_embd', type=int, default=1536)
        parser.add_argument('--n_unmasked', type=int, default=0)
        parser.add_argument('--transformer_dropout', type=float, default=0.)
        parser.add_argument('--sample_every_n_latent_frames', type=int, default=0)
        
        parser.add_argument('--first_stage_key', type=str, default='video', choices=['video'])
        parser.add_argument('--cond_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])

        parser.add_argument('--lr_min', type=float, default=0.)
        parser.add_argument('--warmup_steps', type=int, default=0)
        parser.add_argument('--warmup_lr_init', type=float, default=0.)
        parser.add_argument('--grad_accumulates', type=int, default=1)
        parser.add_argument('--grad_clip_val', type=float, default=1.0)

        return parser

