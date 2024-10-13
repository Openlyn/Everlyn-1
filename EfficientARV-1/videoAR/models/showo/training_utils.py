# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

import copy
import os
import random
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    # set seed first
    set_seed(seed)

    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMA:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

        self.model_cls = model_cls
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, path, model_cls) -> "EMA":
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(model.parameters(), model_cls=model_cls, model_config=model.config)

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        self.copy_to(model.parameters())
        model.save_pretrained(path)

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.to(param.device).data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without:
        affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights to `restore()`")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)

        # Better memory-wise.
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            self.shadow_params = shadow_params
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")


# calculates entropy over each pixel distribution
def pixel_entropy_per_percent_masked_bucket(logits, input_ids, mask_id):
    # only calculated entropy over image tokens that were masked in the original image
    masked_tokens = input_ids == mask_id
    num_masked_pixels = masked_tokens.sum(-1)

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    entropy_per_pixel = -((probs * log_probs).sum(-1))

    # the predictions for non-masked aren't used, so set their entropies to zero
    entropy_per_pixel[~masked_tokens] = 0

    entropy_per_image_numerator = entropy_per_pixel.sum(-1)
    entropy_per_image = entropy_per_image_numerator / num_masked_pixels

    total_buckets = 10
    masked_buckets = input_ids_to_masked_buckets(input_ids, mask_id, total_buckets)

    entropy_by_masked_bucket = average_by_buckets(entropy_per_image, masked_buckets, total_buckets)

    return entropy_by_masked_bucket


# calculates entropy over the averaged distribution of pixels for the whole image
def image_entropy_per_percent_masked_bucket(logits, input_ids, mask_id):
    # only calculated entropy over image tokens that were masked in the original image
    masked_tokens = input_ids == mask_id
    num_masked_pixels = masked_tokens.sum(-1, keepdim=True)

    pixel_probs = F.softmax(logits, dim=-1)
    pixel_probs[~masked_tokens] = 0
    image_probs_numerator = pixel_probs.sum(-2)
    image_probs = image_probs_numerator / num_masked_pixels

    image_log_probs = image_probs.log()

    entropy_per_image = -((image_probs * image_log_probs).sum(-1))

    total_buckets = 10
    masked_buckets = input_ids_to_masked_buckets(input_ids, mask_id, total_buckets)

    entropy_by_masked_bucket = average_by_buckets(entropy_per_image, masked_buckets, total_buckets)

    return entropy_by_masked_bucket


def cross_entropy_per_percent_masked_bucket(logits, labels, input_ids, mask_id, output_size, label_smoothing):
    cross_entropy_per_image = F.cross_entropy(
        logits.view(-1, output_size),
        labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    total_buckets = 10
    masked_buckets = input_ids_to_masked_buckets(input_ids, mask_id, total_buckets)

    cross_entropy_by_percent_masked_bucket = average_by_buckets(cross_entropy_per_image, masked_buckets, total_buckets)

    return cross_entropy_by_percent_masked_bucket


def token_probability_distributions_per_percent_masked_bucket(logits, input_ids, mask_id):
    probs = F.softmax(logits, dim=-1)

    total_buckets = 10
    masked_buckets = input_ids_to_masked_buckets(input_ids, mask_id, total_buckets)

    data = []

    for bucket_idx in range(total_buckets):
        indices_for_bucket = masked_buckets[masked_buckets == bucket_idx]

        # It's ok if none were noised in the range of this bucket. This
        # function will be called for a later training step where it's likely
        # there will be an element noised in the range.
        if indices_for_bucket.shape[0] == 0:
            continue

        index_for_bucket = indices_for_bucket[0]

        image_probs = probs[index_for_bucket]

        # find the index of a masked pixel for the image
        input_ids_for_image = input_ids[index_for_bucket]
        masked_pixels_probs = image_probs[input_ids_for_image == mask_id]

        masked_pixel_probs = masked_pixels_probs[0]

        masked_pixel_probs = masked_pixel_probs.cpu().numpy()

        for masked_pixel_prob in masked_pixel_probs:
            data.append({"bucket": bucket_idx, "masked_pixel_prob": masked_pixel_prob})

    df = pd.DataFrame(data)

    return df


def average_by_buckets(values, masked_buckets, total_buckets):
    unique_buckets, bucket_counts = masked_buckets.unique(dim=0, return_counts=True)

    numerator = torch.zeros(total_buckets, device=values.device)

    numerator.scatter_add_(0, masked_buckets, values)

    # default value is one because the buckets for which there aren't
    # any values will have a numerator of zero. So we just need to not divide
    # by zero.
    denominator = torch.ones(total_buckets, device=values.device, dtype=torch.long)
    denominator[unique_buckets] = bucket_counts

    averaged_by_buckets = numerator / denominator

    return averaged_by_buckets


def input_ids_to_masked_buckets(input_ids, mask_id, total_buckets=10):
    assert total_buckets == 10

    masked_percent = (input_ids == mask_id).sum(-1) / input_ids.shape[-1]

    # we do not formally use timesteps to noise images. Instead, we mask a percent
    # of the pixels. We don't want to log entropy for every mask percent between 0 and 1,
    # and we also want to track how the entropy evolves over time w/in a range of mask
    # percents that should have similar entropy. So we bucket the masked percents into a
    # fixed number of buckets

    # we could generalize this later if needed but for now, let's just assume a fixed
    # number of 10 buckets.

    # How this maps to a bucket index:
    # (mask) * bucket_index +
    # (mask_1) * bucket_index_1
    #
    # -> Where the mask is true will be set to the expected bucket index,
    # where the mask is false will be set to 0.
    #
    # Given the probabilities are between 0 and 1, each masked_percent will get mapped
    # to a timestep by one and only one of the masks.

    masked_buckets = (
        ((0 < masked_percent) & (masked_percent <= 0.1)) * 0
        + ((0.1 < masked_percent) & (masked_percent <= 0.2)) * 1
        + ((0.2 < masked_percent) & (masked_percent <= 0.3)) * 2
        + ((0.3 < masked_percent) & (masked_percent <= 0.4)) * 3
        + ((0.4 < masked_percent) & (masked_percent <= 0.5)) * 4
        + ((0.5 < masked_percent) & (masked_percent <= 0.6)) * 5
        + ((0.6 < masked_percent) & (masked_percent <= 0.7)) * 6
        + ((0.7 < masked_percent) & (masked_percent <= 0.8)) * 7
        + ((0.8 < masked_percent) & (masked_percent <= 0.9)) * 8
        + ((0.9 < masked_percent) & (masked_percent <= 1.0)) * 9
    )

    return masked_buckets
