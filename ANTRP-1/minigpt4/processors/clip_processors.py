"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from minigpt4.common.registry import registry
from minigpt4.processors.base_processor import BaseProcessor
from minigpt4.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPImageProcessor



@registry.register_processor("clip_image_train")
class ClipImageTrainProcessor(BaseProcessor):
    def __init__(self, proc_type, do_normalize=True):
        super().__init__()

        self.transform = CLIPImageProcessor.from_pretrained(proc_type)
        self.transform.do_normalize = True if do_normalize else False

    def __call__(self, item):
        return self.transform.preprocess(item, return_tensors='pt')['pixel_values'][0]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        proc_type = cfg.get("proc_type", r'openai/clip-vit-large-patch14')

        do_normalize = cfg.get("do_normalize", True)

        return cls(proc_type=proc_type, do_normalize=do_normalize)


@registry.register_processor("clip_image_eval")
class ClipImageEvalProcessor(BaseProcessor):
    def __init__(self, proc_type, do_normalize=True):
        super().__init__()

        self.transform = CLIPImageProcessor.from_pretrained(proc_type)
        self.transform.do_normalize = True if do_normalize else False

    def __call__(self, item):
        return self.transform.preprocess(item, return_tensors='pt')['pixel_values'][0]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        proc_type = cfg.get("proc_type", r'openai/clip-vit-large-patch14')

        do_normalize = cfg.get("do_normalize", True)

        return cls(proc_type=proc_type, do_normalize=do_normalize)

@registry.register_processor("clip_image_train_336")
class ClipImageTrainProcessor(BaseProcessor):
    def __init__(self, proc_type, do_normalize=True):
        super().__init__()

        self.transform = CLIPImageProcessor.from_pretrained(proc_type)
        self.transform.do_normalize = True if do_normalize else False

    def __call__(self, item):
        return self.transform.preprocess(item, return_tensors='pt')['pixel_values'][0]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        proc_type = cfg.get("proc_type", r'openai/clip-vit-large-patch14-336')

        do_normalize = cfg.get("do_normalize", True)

        return cls(proc_type=proc_type, do_normalize=do_normalize)


@registry.register_processor("clip_image_eval_336")
class ClipImageEvalProcessor(BaseProcessor):
    def __init__(self, proc_type, do_normalize=True):
        super().__init__()

        self.transform = CLIPImageProcessor.from_pretrained(proc_type)
        self.transform.do_normalize = True if do_normalize else False

    def __call__(self, item):
        return self.transform.preprocess(item, return_tensors='pt')['pixel_values'][0]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        proc_type = cfg.get("proc_type", r'openai/clip-vit-large-patch14-336')

        do_normalize = cfg.get("do_normalize", True)

        return cls(proc_type=proc_type, do_normalize=do_normalize)