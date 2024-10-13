# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import PatchEmbed, Timesteps, TimestepEmbedding
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, is_torch_version
from einops import rearrange
from torch import nn
from typing import Dict, Optional, Tuple

from .attention import (SelfAttentionTemporalTransformerBlock,
                        TemporalTransformerBlock)
from .patch import Patch1D, PatchEmbed3D, PatchEmbedF3D, UnPatch1D, TemporalUpsampler3D, CasualPatchEmbed3D

try:
    from diffusers.models.embeddings import PixArtAlphaTextProjection
except:
    from diffusers.models.embeddings import \
        CaptionProjection as PixArtAlphaTextProjection

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            
            self.resolution_embedder.linear_2 = zero_module(self.resolution_embedder.linear_2)
            self.aspect_ratio_embedder.linear_2 = zero_module(self.aspect_ratio_embedder.linear_2)

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            conditioning = timesteps_emb

        return conditioning

class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class TimePositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, c, f, h, w = x.size()
        x = rearrange(x, "b c f h w -> (b h w) f c")
        x = x + self.pe[:, :x.size(1)]
        x = rearrange(x, "(b h w) f c -> b c f h w", b=b, h=h, w=w)
        return self.dropout(x)
    
@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    """
    A 3D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        # block type
        basic_block_type: str = "motionmodule",
        # enable_uvit
        enable_uvit: bool = False,

        # 3d patch params
        patch_3d: bool = False,
        fake_3d: bool = False,
        time_patch_size: Optional[int] = None,

        casual_3d: bool = False,
        casual_3d_upsampler_index: Optional[list] = None,

        # motion module kwargs
        motion_module_type = "VanillaGrid",
        motion_module_kwargs = None,

        # time position encoding
        time_position_encoding_before_transformer = False
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.enable_uvit = enable_uvit
        inner_dim = num_attention_heads * attention_head_dim
        self.basic_block_type = basic_block_type
        self.patch_3d = patch_3d
        self.fake_3d = fake_3d
        self.casual_3d = casual_3d
        self.casual_3d_upsampler_index = casual_3d_upsampler_index

        conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        assert sample_size is not None, "Transformer3DModel over patched input must provide sample_size"

        self.height = sample_size
        self.width = sample_size

        self.patch_size = patch_size
        self.time_patch_size = self.patch_size if time_patch_size is None else time_patch_size
        interpolation_scale = self.config.sample_size // 64  # => 64 (= 512 pixart) has interpolation scale 1
        interpolation_scale = max(interpolation_scale, 1)
        
        if self.casual_3d:
            self.pos_embed = CasualPatchEmbed3D(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                time_patch_size=self.time_patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )
        elif self.patch_3d:
            if self.fake_3d:
                self.pos_embed = PatchEmbedF3D(
                    height=sample_size,
                    width=sample_size,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    embed_dim=inner_dim,
                    interpolation_scale=interpolation_scale,
                )
            else:
                self.pos_embed = PatchEmbed3D(
                    height=sample_size,
                    width=sample_size,
                    patch_size=patch_size,
                    time_patch_size=self.time_patch_size,
                    in_channels=in_channels,
                    embed_dim=inner_dim,
                    interpolation_scale=interpolation_scale,
                )
        else:
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )

        # 3. Define transformers blocks
        if self.basic_block_type == "motionmodule":
            self.transformer_blocks = nn.ModuleList(
                [
                    TemporalTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                    )
                    for d in range(num_layers)
                ]
            )
        elif self.basic_block_type == "kvcompression_motionmodule":
            self.transformer_blocks = nn.ModuleList(
                [
                    TemporalTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        kvcompression=False if d < 14 else True,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                    )
                    for d in range(num_layers)
                ]
            )
        elif self.basic_block_type == "selfattentiontemporal":
            self.transformer_blocks = nn.ModuleList(
                [
                    SelfAttentionTemporalTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                    )
                    for d in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                    )
                    for d in range(num_layers)
                ]
            )
        
        if self.casual_3d:
            self.unpatch1d = TemporalUpsampler3D()
        elif self.patch_3d and self.fake_3d:
            self.unpatch1d = UnPatch1D(inner_dim, True)

        if self.enable_uvit:
            self.long_connect_fc = nn.ModuleList(
                [
                    nn.Linear(inner_dim, inner_dim, True) for d in range(13)
                ]
            )
            for index in range(13):
                self.long_connect_fc[index] = zero_module(self.long_connect_fc[index])

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            if self.patch_3d and not self.fake_3d:
                self.proj_out_2 = nn.Linear(inner_dim, self.time_patch_size * patch_size * patch_size * self.out_channels)
            else:
                self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
        elif norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
            if self.patch_3d and not self.fake_3d:
                self.proj_out = nn.Linear(inner_dim, self.time_patch_size * patch_size * patch_size * self.out_channels)
            else:
                self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            self.use_additional_conditions = self.config.sample_size == 128
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.gradient_checkpointing = False
        
        self.time_position_encoding_before_transformer = time_position_encoding_before_transformer
        if self.time_position_encoding_before_transformer:
            self.t_pos = TimePositionalEncoding(max_len = 4096, d_model = inner_dim)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        inpaint_latents: torch.Tensor = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer3DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(encoder_hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 1)
        # 1. Input
        if self.casual_3d:
            video_length, height, width = (hidden_states.shape[-3] - 1) // self.time_patch_size + 1, hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        elif self.patch_3d:
            video_length, height, width = hidden_states.shape[-3] // self.time_patch_size, hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        else:
            video_length, height, width = hidden_states.shape[-3], hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states = rearrange(hidden_states, "b c f h w ->(b f) c h w")

        hidden_states = self.pos_embed(hidden_states)
        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            batch_size = hidden_states.shape[0] // video_length
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
        hidden_states = rearrange(hidden_states, "(b f) (h w) c -> b c f h w", f=video_length, h=height, w=width)

        # hidden_states
        # bs, c, f, h, w => b (f h w ) c
        if self.time_position_encoding_before_transformer:
            hidden_states = self.t_pos(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        
        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        skips = []
        skip_index = 0
        for index, block in enumerate(self.transformer_blocks):
            if self.enable_uvit:
                if index >= 15:
                    long_connect = self.long_connect_fc[skip_index](skips.pop())
                    hidden_states = hidden_states + long_connect
                    skip_index += 1

            if self.casual_3d_upsampler_index is not None and index in self.casual_3d_upsampler_index:
                hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w", f=video_length, h=height, w=width)
                hidden_states = self.unpatch1d(hidden_states)
                video_length = (video_length - 1) * 2 + 1
                hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c", f=video_length, h=height, w=width)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                args = {
                    "basic": [],
                    "motionmodule": [video_length, height, width],
                    "selfattentiontemporal": [video_length, height, width],
                    "kvcompression_motionmodule": [video_length, height, width],
                }[self.basic_block_type]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    *args,
                    **ckpt_kwargs,
                )
            else:
                kwargs = {
                    "basic": {},
                    "motionmodule": {"num_frames":video_length, "height":height, "width":width},
                    "selfattentiontemporal": {"num_frames":video_length, "height":height, "width":width},
                    "kvcompression_motionmodule": {"num_frames":video_length, "height":height, "width":width},
                }[self.basic_block_type]
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    **kwargs
                )

            if self.enable_uvit:
                if index < 13:
                    skips.append(hidden_states)

        if self.fake_3d and self.patch_3d:
            hidden_states = rearrange(hidden_states, "b (f h w) c -> (b h w) c f", f=video_length, w=width, h=height)
            hidden_states = self.unpatch1d(hidden_states)
            hidden_states = rearrange(hidden_states, "(b h w) c f -> b (f h w) c", w=width, h=height)

        # 3. Output
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        if self.patch_3d:
            if self.fake_3d:
                hidden_states = hidden_states.reshape(
                    shape=(-1, video_length * self.patch_size, height, width, self.patch_size, self.patch_size, self.out_channels)
                )
                hidden_states = torch.einsum("nfhwpqc->ncfhpwq", hidden_states)
            else:
                hidden_states = hidden_states.reshape(
                    shape=(-1, video_length, height, width, self.time_patch_size, self.patch_size, self.patch_size, self.out_channels)
                )
                hidden_states = torch.einsum("nfhwopqc->ncfohpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, video_length * self.time_patch_size, height * self.patch_size, width * self.patch_size)
            )
        else:
            hidden_states = hidden_states.reshape(
                shape=(-1, video_length, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nfhwpqc->ncfhpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, video_length, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)

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
        
        if model.state_dict()['pos_embed.proj.weight'].size() != state_dict['pos_embed.proj.weight'].size():
            new_shape   = model.state_dict()['pos_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['pos_embed.proj.weight'] = state_dict['pos_embed.proj.weight'].unsqueeze(2).expand(new_shape).clone()
                state_dict['pos_embed.proj.weight'][:, :, :-1] = 0
            else:
                model.state_dict()['pos_embed.proj.weight'][:, :4, :, :] = state_dict['pos_embed.proj.weight']
                model.state_dict()['pos_embed.proj.weight'][:, 4:, :, :] = 0
                state_dict['pos_embed.proj.weight'] = model.state_dict()['pos_embed.proj.weight']
                
        if model.state_dict()['proj_out.weight'].size() != state_dict['proj_out.weight'].size():
            new_shape   = model.state_dict()['proj_out.weight'].size()
            state_dict['proj_out.weight'] = torch.tile(state_dict['proj_out.weight'], [patch_size, 1])

        if model.state_dict()['proj_out.bias'].size() != state_dict['proj_out.bias'].size():
            new_shape   = model.state_dict()['proj_out.bias'].size()
            state_dict['proj_out.bias'] = torch.tile(state_dict['proj_out.bias'], [patch_size])

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