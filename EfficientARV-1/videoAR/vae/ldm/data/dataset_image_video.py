import glob
import json
import os
import pickle
import random
import shutil
import tarfile
from functools import partial

import albumentations
import cv2
import numpy as np
import PIL
import torchvision.transforms.functional as TF
import yaml
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_set_timeout
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import (BatchSampler, Dataset, Sampler)
from tqdm import tqdm

from ..modules.image_degradation import (degradation_fn_bsr,
                                         degradation_fn_bsr_light)


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.sampler_pos_start = 0
        self.sampler_pos_reload = 0

        self.num_samples_random = len(self.sampler)
        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}
            
    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        for index_sampler, idx in enumerate(self.sampler):
            if self.sampler_pos_reload != 0 and self.sampler_pos_reload < self.num_samples_random:
                if index_sampler < self.sampler_pos_reload:
                    self.sampler_pos_start = (self.sampler_pos_start + 1) % self.num_samples_random
                    continue
                elif index_sampler == self.sampler_pos_reload:
                    self.sampler_pos_reload = 0
                        
            content_type = self.dataset.data.get_type(idx)
            bucket = self.bucket[content_type]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                yield self.bucket['video']
                self.bucket['video'] = []
            elif len(self.bucket['image']) == self.batch_size:
                yield self.bucket['image']
                self.bucket['image'] = []
            self.sampler_pos_start = (self.sampler_pos_start + 1) % self.num_samples_random

class ImageVideoDataset(Dataset): 
    # update __getitem__() from ImageNetSR. If timeout for Pandas70M, throw exception.
    # If caught exception(timeout or others), try another index until successful and return.
    def __init__(self, size=None, video_size=128, video_len=25,
                 degradation=None, downscale_f=4,  random_crop=True, min_crop_f=0.25, max_crop_f=1.,
                 s_t=None, slice_interval=None, data_root=None
                ):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop
        self.s_t = s_t
        self.slice_interval = slice_interval

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.video_rescaler = albumentations.SmallestMaxSize(max_size=video_size, interpolation=cv2.INTER_AREA)
        self.video_len = video_len
        self.video_size = video_size
        self.data_root = data_root

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)
        else:
            interpolation_fn = {
                "cv_nearest": cv2.INTER_NEAREST,
                "cv_bilinear": cv2.INTER_LINEAR,
                "cv_bicubic": cv2.INTER_CUBIC,
                "cv_area": cv2.INTER_AREA,
                "cv_lanczos": cv2.INTER_LANCZOS4,
                "pil_nearest": PIL.Image.NEAREST,  
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:                
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)
    
    def get_type(self, index):
        return self.base[index].get('type', 'image')

    def __getitem__(self, i):
        @func_set_timeout(3) # time wait 3 seconds
        def get_video_item(example):
            if self.data_root is not None:
                video_reader = VideoReader(os.path.join(self.data_root, example['file_path']))
            else:
                video_reader = VideoReader(example['file_path'])
            video_length = len(video_reader)
            
            clip_length = min(video_length, (self.video_len - 1) * self.slice_interval + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.video_len, dtype=int)

            pixel_values = video_reader.get_batch(batch_index).asnumpy()
            
            del video_reader
            out_images = []
            LR_out_images = []
            min_side_len = min(pixel_values[0].shape[:2])

            crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
            crop_side_len = int(crop_side_len)
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
            else:
                self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

            imgs = np.transpose(pixel_values, (1, 2, 3, 0))
            imgs = self.cropper(image=imgs)["image"]
            imgs = np.transpose(imgs, (3, 0, 1, 2))
            for img in imgs:
                image = self.video_rescaler(image=img)["image"]
                out_images.append(image[None, :, :, :])
                if self.pil_interpolation:
                    image_pil = PIL.Image.fromarray(image)
                    LR_image = self.degradation_process(image_pil)
                    LR_image = np.array(LR_image).astype(np.uint8)
                else:
                    LR_image = self.degradation_process(image=image)["image"]
                LR_out_images.append(LR_image[None, :, :, :])

            example = {}
            example['image'] = (np.concatenate(out_images) / 127.5 - 1.0).astype(np.float32)
            example['LR_image'] = (np.concatenate(LR_out_images) / 127.5 - 1.0).astype(np.float32)
            return example

        example = self.base[i]
        if example.get('type', 'image') == 'video':
            while True:  
                try:
                    example = self.base[i]
                    return get_video_item(example)
                except FunctionTimedOut:
                    print("stt catch: Function 'extract failed' timed out.")
                    i = random.randint(0, self.__len__() - 1)
                except Exception as e:
                    print('stt catch', e)
                    i = random.randint(0, self.__len__() - 1)
        elif example.get('type', 'image') == 'image':
            while True:
                try:
                    example = self.base[i]
                    if self.data_root is not None:
                        image = Image.open(os.path.join(self.data_root, example['file_path']))
                    else:
                        image = Image.open(example['file_path'])
                    image = image.convert("RGB")
                    image = np.array(image).astype(np.uint8)

                    min_side_len    = min(image.shape[:2])
                    crop_side_len   = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
                    crop_side_len   = int(crop_side_len)

                    if self.center_crop:
                        self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

                    else:
                        self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

                    image = self.cropper(image=image)["image"]

                    image = self.image_rescaler(image=image)["image"]

                    if self.pil_interpolation:
                        image_pil = PIL.Image.fromarray(image)
                        LR_image = self.degradation_process(image_pil)
                        LR_image = np.array(LR_image).astype(np.uint8)

                    else:
                        LR_image = self.degradation_process(image=image)["image"]
                    
                    example = {}
                    example["image"] = (image/127.5 - 1.0).astype(np.float32)
                    example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)
                    return example
                except Exception as e:
                    print("catch", e)
                    i = random.randint(0, self.__len__() - 1)

class CustomSRTrain(ImageVideoDataset):
    def __init__(self, data_json_path, **kwargs):
        self.data_json_path = data_json_path
        super().__init__(**kwargs)

    def get_base(self):
        return [ann for ann in json.load(open(self.data_json_path))]

class CustomSRValidation(ImageVideoDataset):
    def __init__(self, data_json_path, **kwargs):
        self.data_json_path = data_json_path
        super().__init__(**kwargs)
        self.data_json_path = data_json_path

    def get_base(self):
        return [ann for ann in json.load(open(self.data_json_path))][:100] + \
              [ann for ann in json.load(open(self.data_json_path))][-100:]
