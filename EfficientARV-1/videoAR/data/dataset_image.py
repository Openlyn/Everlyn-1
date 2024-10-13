import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class CC15M(Dataset):
    def __init__(
            self,
            json_path, 
            video_folder=None,
            resolution=512,
            enable_bucket=False,
        ):
        print(f"loading annotations from {json_path} ...")
        self.dataset = json.load(open(json_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        
        self.enable_bucket = enable_bucket
        self.video_folder = video_folder

        resolution = tuple(resolution) if not isinstance(resolution, int) else (resolution, resolution)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(resolution[0]),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_id, name = video_dict['file_path'], video_dict['text']

        if self.video_folder is None:
            video_dir = video_id
        else:
            video_dir = os.path.join(self.video_folder, video_id)

        pixel_values = Image.open(video_dir).convert("RGB")
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length-1)

        if not self.enable_bucket:
            pixel_values = self.pixel_transforms(pixel_values)
        else:
            pixel_values = np.array(pixel_values)

        sample = dict(pixel_values=pixel_values, text=name)
        return sample

if __name__ == "__main__":
    dataset = CC15M(
        csv_path="/mnt_wg/zhoumo.xjq/CCUtils/cc15m_add_index.json",
        resolution=512,
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))