import logging
import string
from packaging import version

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import disabled_train
from minigpt4.models.modeling_shikra import ShikraBase, ShikraLlamaForCausalLM



IGNORE_INDEX = -100
# DEFAULT_IMAGE_TOKEN = IMAGE_PLACEHOLDER
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"



@registry.register_model("shikra")
class Shikra(ShikraBase):
    """
    Shikra Vicuna model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/shikra_vicuna7b.yaml",
    }

    def __init__(
        self,
        vision_tower=r'openai/clip-vit-large-patch14',
        mm_vision_select_layer=-2,
        merged_ckpt="",
        cache_dir=None,
        model_max_length=2048,
        shikra_version="v1",
        freeze_backbone=False,
        mm_use_im_start_end=True,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,
        apply_fsdp=None,
        max_txt_len=128,
        max_output_txt_len=256,
        low_resource=False,  # use 8 bit and put vit in cpu
        bf16=False, 
        fp16=True,
        system_message="",
    ):
        super().__init__()

        self.low_resource = low_resource
        self.system_message = system_message
        if self.low_resource:
            quantization_kwargs = dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            )
        else:
            quantization_kwargs = dict()

        self.llama_model = ShikraLlamaForCausalLM.from_pretrained(
            merged_ckpt,
            cache_dir=cache_dir,
            **quantization_kwargs
        )
        self.llama_model.config.use_cache = False
        if freeze_backbone:
            self.llama_model.model.requires_grad_(False)

        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            merged_ckpt,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )

        self.version_check(
            self.llama_model, self.llama_tokenizer, shikra_version, merged_ckpt
        )

        self.model_vision_dict = self.llama_model.model.initialize_vision_modules(
            vision_tower=vision_tower,
            mm_vision_select_layer=mm_vision_select_layer,
            pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
        )
        vision_config = self.model_vision_dict['vision_config']
        self.quantization_check(
            self.llama_model, vision_tower, fp16, bf16
        )

        self.llama_model.config.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        if tune_mm_mlp_adapter:
            self.llama_model.requires_grad_(False)
            for p in self.llama_model.model.mm_projector.parameters():
                p.requires_grad = True

        self.llama_model.config.freeze_mm_mlp_adapter = freeze_mm_mlp_adapter
        if freeze_mm_mlp_adapter:
            for p in self.llama_model.model.mm_projector.parameters():
                p.requires_grad = False

        self.llama_model.config.mm_use_im_start_end = mm_use_im_start_end
        vision_config.use_im_start_end = mm_use_im_start_end

        self.llama_model.initialize_vision_tokenizer(
            mm_use_im_start_end=mm_use_im_start_end,
            tokenizer=self.llama_tokenizer,
            device="cuda",
            tune_mm_mlp_adapter=tune_mm_mlp_adapter,
            pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.fsdp_check(
            self.llama_model, apply_fsdp
        )

    def forward(self, samples):
        image = samples["image"]
        instruction = samples["instruction_input"] if "instruction_input" in samples else None

        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.model_vision_dict['image_token_len']
        instruction = instruction.replace(DEFAULT_IMAGE_PATCH_TOKEN, replace_token)

        input_ids = self.llama_tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        ###TODO: targets, attention_mask
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self, 
        # samples,
        prompt,
        image,
        images_cd,
        prompt_cd,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        max_new_tokens=300,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        output_attentions=False,
        use_cache=False,
        # ours
        opera_decoding=False,
        key_position=None,
        scale_factor=1.0,
        threshold=1,
        num_attn_candidates=5,
        penalty_weights=1.0,
    ):
        self.llama_tokenizer.padding_side = "left"

        # image = samples["image"]

        # instruction = samples["prompt"] if "prompt" in samples else None

        image = image


        if prompt_cd is not None:
            instruction_cd = prompt_cd
            bs = image.size(0)
            if isinstance(instruction_cd, str):
                instruction_cd = [instruction_cd] * bs
            else:
                assert len(instruction_cd) == bs, "The number of prompts must be equal to the batch size."
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.model_vision_dict['image_token_len']
            instruction_cd = [p.replace('<ImageHere>', replace_token) for p in instruction_cd]
            instruction_cd = [self.system_message + p for p in instruction_cd]

            input_tokens_cd = self.llama_tokenizer(
                instruction_cd,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(image.device)

            bos = torch.ones([bs, 1],
                            dtype=torch.int64,
                            device=image.device) * self.llama_tokenizer.bos_token_id

            with self.maybe_autocast():
                input_ids_cd = torch.cat([bos, input_tokens_cd.input_ids], dim=1)
        else:
            input_ids_cd = None


        instruction = prompt

        bs = image.size(0)

        if isinstance(instruction, str):
            instruction = [instruction] * bs
        else:
            assert len(instruction) == bs, "The number of prompts must be equal to the batch size."

        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.model_vision_dict['image_token_len']
        instruction = [p.replace('<ImageHere>', replace_token) for p in instruction]
        instruction = [self.system_message + p for p in instruction]

        input_tokens = self.llama_tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len+self.model_vision_dict['image_token_len'],
            add_special_tokens=False
        ).to(image.device)
        # print(input_tokens.input_ids.shape)

        # with self.maybe_autocast():
        #     inputs_embeds = self.llama_model.get_input_embeddings()(input_tokens.input_ids)

        bos = torch.ones([bs, 1],
                         dtype=torch.int64,
                         device=image.device) * self.llama_tokenizer.bos_token_id
        # bos_embeds = self.embed_tokens(bos)
        # atts_bos = input_tokens.attention_mask[:, :1]

        with self.maybe_autocast():
            # print(bos_embeds.shape, inputs_embeds.shape)
            input_ids = torch.cat([bos, input_tokens.input_ids], dim=1)
            # inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
            # attention_mask = torch.cat([atts_bos, input_tokens.attention_mask], dim=1)
            if key_position is None:
                image_start_pos = torch.where(input_ids == 32001)[1][0].item()
                image_end_pos = torch.where(input_ids == 32002)[1][0].item()
                key_position = {
                    "image_start": image_start_pos, 
                    "image_end": image_end_pos, 
                    "response_start": input_ids.shape[1]
                }

            outputs = self.llama_model.generate(
                input_ids=input_ids,
                images_cd=images_cd,
                input_ids_cd=input_ids_cd,
                use_cache=True,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                # max_length=512,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                bos_token_id=self.llama_tokenizer.bos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                # repetition_penalty=repetition_penalty,
                # length_penalty=length_penalty,
                # num_return_sequences=num_captions,
                images=image,
                output_attentions=output_attentions,
                # opera
                opera_decoding=opera_decoding,
                key_position=key_position,
                scale_factor=scale_factor,
                threshold=threshold,
                num_attn_candidates=num_attn_candidates,
                penalty_weights=penalty_weights,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        outputs[outputs == 1] = 2 # convert output id 1 to 2 (eos_token_id)
        output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.split('ASSISTANT:')[-1].strip() for text in output_text]
        return output_text


    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds


    @classmethod
    def from_config(cls, cfg):
        vision_tower = cfg.get("vit_model", r'openai/clip-vit-large-patch14')
        mm_vision_select_layer = cfg.get("mm_vision_select_layer", -2)
        merged_ckpt = cfg.get("merged_ckpt", "")
        cache_dir = cfg.get("cache_dir", None)
        model_max_length = cfg.get("model_max_length", 2048)
        shikra_version = cfg.get("version", "v1")
        freeze_backbone = cfg.get("freeze_backbone", False)
        mm_use_im_start_end = cfg.get("mm_use_im_start_end", True)
        pretrain_mm_mlp_adapter = cfg.get("pretrain_mm_mlp_adapter", None)
        tune_mm_mlp_adapter = cfg.get("tune_mm_mlp_adapter", False)
        freeze_mm_mlp_adapter = cfg.get("freeze_mm_mlp_adapter", False)
        apply_fsdp = cfg.get("apply_fsdp", None)
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)
        low_resource = cfg.get("low_resource", False)
        bf16 = cfg.get("bf16", False)
        fp16 = cfg.get("fp16", False)
        system_message = cfg.get("system_message", "")

        model = cls(
            vision_tower=vision_tower,
            mm_vision_select_layer=mm_vision_select_layer,
            merged_ckpt=merged_ckpt,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            shikra_version=shikra_version,
            freeze_backbone=freeze_backbone,
            mm_use_im_start_end=mm_use_im_start_end,
            pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
            tune_mm_mlp_adapter=tune_mm_mlp_adapter,
            freeze_mm_mlp_adapter=freeze_mm_mlp_adapter,
            apply_fsdp=apply_fsdp,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            low_resource=False,  # use 8 bit and put vit in cpu
            bf16=bf16, fp16=fp16,
            system_message=system_message,
        )

        return model