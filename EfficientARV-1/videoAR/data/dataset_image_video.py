import csv
from functools import cache
import io
import json
import math
import os
import random
from threading import Thread

import albumentations
import cv2
import gc
import numpy as np
import torch
import torchvision.transforms as transforms
from func_timeout import func_timeout, FunctionTimedOut
from decord import VideoReader
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
from contextlib import contextmanager
import pickle
VIDEO_READER_TIMEOUT = 20

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
                 image_batch_size: int,
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
        self.image_batch_size = image_batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.image_batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    # breakpoint()
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

class ImageVideoDataset(Dataset):
    def __init__(
            self,
            ann_path, data_root=None,
            video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
            image_sample_size=512,
            video_repeat=0,
            text_drop_ratio=0.00,
            enable_bucket=False,
            video_length_drop_start=0.1, 
            video_length_drop_end=0.9,
            cache_dir=None,
        ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        elif ann_path.endswith('.parquet'):
            import pandas as pd
            import pyarrow.parquet as pq
            with open(ann_path, mode="rb") as file:
                df = pq.read_table(file)
            df = df.to_pandas()
            dataset = df['data']
            del df
    
        self.data_root = data_root
        self.cache_dir = cache_dir
        # It's used to balance num of images and videos.
        self.dataset = []
        for data in dataset:
            if data.get('type', 'image') != 'video':
                self.dataset.append(data)
        if video_repeat > 0:
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(video_sample_size[0]),
                transforms.CenterCrop(video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
    
    def get_batch(self, idx, data_info):
        # data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            if self.cache_dir is not None:
                if self.data_root is not None:
                    cache_path = f'{self.data_root}/{self.cache_dir}/{"/".join(video_id.split("/")[1:])}'
                else:
                    cache_path = video_dir.split('/')
                    cache_path[-3] = self.cache_dir
                    cache_path = '/'.join(cache_path)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                vid_cache_path = cache_path[:-4]+'_vid.pkl'
                text_cache_path = cache_path[:-4]+'_txt.pkl'
                
                if os.path.isfile(vid_cache_path) and os.path.isfile(text_cache_path):
                    with open(vid_cache_path, 'rb') as f:
                        data_item = pickle.load(f)
                    latent_idx = data_item['latent_idx']

                    with open(text_cache_path, 'rb') as f:
                        data_item = pickle.load(f)
                    text_embedding, text_emb_masks = data_item['text_embedding'], data_item['text_emb_masks']
                    return None, text, 'video', vid_cache_path, text_cache_path, latent_idx, text_embedding, text_emb_masks
            # breakpoint()
            with VideoReader_contextmanager(video_dir, num_threads=1) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start))
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                
                if video_length < clip_length or video_length < self.video_sample_n_frames:
                    raise ValueError(f"No Enough Frames for the sample stride {self.video_sample_stride}.")
                
                start_idx_end = max(int(self.video_length_drop_start * video_length), video_length - clip_length)   
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), start_idx_end)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    print(e)
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                if pixel_values.shape[0] < self.video_sample_n_frames:
                    raise ValueError(f"No enough frames from video. only {pixel_values.shape[0]} frames")
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video', vid_cache_path, text_cache_path, None, None, None
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, vid_cache_path, text_cache_path, latent_idx, text_embedding, text_emb_masks = self.get_batch(idx, data_info_local)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                sample['vid_cache_path'] = vid_cache_path
                sample['text_cache_path'] = text_cache_path
                sample['latent_idx'] = latent_idx
                sample['text_embedding'] = text_embedding
                sample['text_emb_masks'] = text_emb_masks
                if 'label' in data_info_local:
                    sample['label'] = int(data_info_local['label'])
                
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)
        return sample

if __name__ == "__main__":
    dataset = ImageVideoDataset(
        ann_path="test.json"
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))