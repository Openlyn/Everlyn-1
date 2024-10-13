import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.llava_llama import LlavaLlamaForCausalLM
from minigpt4.models.base_model import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig


# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
NUM_IMAGE_TOKENS = 576


@registry.register_model("llava-1.5")
class LLaVa(BaseModel):
    """
    LLaVa-1.5 model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/llava-1.5_vicuna7b.yaml",
        "vicuna13b": "configs/models/llava-1.5_vicuna13b.yaml",
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
        load_8bit=False, 
        load_4bit=False, 
        device_map="auto", 
        device="cuda",
    ):
        super().__init__()

        kwargs = {"device_map": device_map}
        self.system_message = system_message

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        self.llama_tokenizer = AutoTokenizer.from_pretrained(merged_ckpt, use_fast=False)
        self.llama_model = LlavaLlamaForCausalLM.from_pretrained(
            merged_ckpt, low_cpu_mem_usage=True, **kwargs)

        mm_use_im_start_end = getattr(self.llama_model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.llama_model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.llama_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        vision_tower = self.llama_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples):
        image = samples["image"]

        instruction = samples["prompt"] if "prompt" in samples else None

        bs = image.size(0)

        if isinstance(instruction, str):
            instruction = [instruction] * bs
        else:
            assert len(instruction) == bs, "The number of prompts must be equal to the batch size."

        instruction = [p.replace('<ImageHere>', '<image>') for p in instruction]
        instruction = [self.system_message + p for p in instruction]

        input_ids = self.tokenizer_image_token(instruction, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        ###TODO: targets, attention_mask
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_ids=inputs_ids,
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
        return_dict_in_generate=False,
        use_cache=False,
        # ours
        opera_decoding=False,
        key_position=None,
        scale_factor=1.0,
        threshold=1,
        num_attn_candidates=5,
        penalty_weights=1.0,
        cd_alpha = None,
        cd_beta = None,
        sample_greedy = None
    ):
        self.llama_tokenizer.padding_side = "left"

        # image = samples["image"]
        image = image
        # img_embeds, atts_img = self.encode_img(image)

        # instruction = samples["prompt"] if "prompt" in samples else None

        if prompt_cd is not None:
            instruction_cd = prompt_cd
            bs = image.size(0)
            if isinstance(instruction_cd, str):
                instruction_cd = [instruction_cd] * bs
            else:
                assert len(instruction_cd) == bs, "The number of prompts must be equal to the batch size."
            instruction_cd = [self.system_message + p for p in instruction_cd]
            chunks_before, chunks_after = [], []
            for p in instruction_cd:
                chunk_before, chunk_after = p.split('<ImageHere>')
                chunks_before.append(chunk_before)
                chunks_after.append(chunk_after)

            tokens_before = self.llama_tokenizer(
                chunks_before,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(image.device).input_ids

            tokens_after = self.llama_tokenizer(
                chunks_after,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(image.device).input_ids

            bos = torch.ones([bs, 1],
                    dtype=torch.int64,
                    device=image.device) * self.llama_tokenizer.bos_token_id

            image_token = torch.ones([bs, 1],
                         dtype=torch.int64,
                         device=image.device) * IMAGE_TOKEN_INDEX

            with self.maybe_autocast():
                input_ids_cd = torch.cat([bos, tokens_before, image_token, tokens_after], dim=1)
        else:
            input_ids_cd = None


        instruction = prompt

        bs = image.size(0)

        if isinstance(instruction, str):
            instruction = [instruction] * bs
        else:
            assert len(instruction) == bs, "The number of prompts must be equal to the batch size."

        instruction = [self.system_message + p for p in instruction]

        chunks_before, chunks_after = [], []
        for p in instruction:
            chunk_before, chunk_after = p.split('<ImageHere>')
            chunks_before.append(chunk_before)
            chunks_after.append(chunk_after)

        tokens_before = self.llama_tokenizer(
            chunks_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False
        ).to(image.device).input_ids

        tokens_after = self.llama_tokenizer(
            chunks_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False
        ).to(image.device).input_ids

        bos = torch.ones([bs, 1],
                         dtype=torch.int64,
                         device=image.device) * self.llama_tokenizer.bos_token_id

        image_token = torch.ones([bs, 1],
                         dtype=torch.int64,
                         device=image.device) * IMAGE_TOKEN_INDEX

        with self.maybe_autocast():
            input_ids = torch.cat([bos, tokens_before, image_token, tokens_after], dim=1)

            if key_position is None:
                key_position = {
                    "image_start": tokens_before.shape[1]+1, 
                    "image_end": tokens_before.shape[1]+NUM_IMAGE_TOKENS, 
                    "response_start": input_ids.shape[1]+NUM_IMAGE_TOKENS-1,
                }

            output_ids = self.llama_model.generate(
                input_ids=input_ids,
                input_ids_cd=input_ids_cd,
                # images = image,
                images_cd=images_cd,
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
                return_dict_in_generate=return_dict_in_generate,
                # opera
                opera_decoding=opera_decoding,
                key_position=key_position,
                scale_factor=scale_factor,
                threshold=threshold,
                num_attn_candidates=num_attn_candidates,
                penalty_weights=penalty_weights,
                sample_greedy = sample_greedy
            )
        if return_dict_in_generate == True:
            # input_token_len = input_ids.shape[1]
            # n_diff_input_output = (input_ids != output_ids.sequences[:, :input_token_len]).sum().item()
            # if n_diff_input_output > 0:
            #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            # output_text = self.llama_tokenizer.batch_decode(output_ids.sequences[:, input_token_len:], skip_special_tokens=True)
            # output_text = [text.split('###')[0].strip() for text in output_text]
            output_text = output_ids
        else:
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            output_text = self.llama_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            output_text = [text.split('###')[0].strip() for text in output_text]
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
        load_8bit = cfg.get("load_8bit", False)
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
            low_resource=low_resource,  # use 8 bit and put vit in cpu
            bf16=bf16, fp16=fp16,
            system_message=system_message,
        )

        return model