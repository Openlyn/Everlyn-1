import random
import torch
import sys  
import os
import pandas as pd
import numpy as np
import PIL.Image as PImage
from PIL import Image
from glob import glob
import torchvision.datasets as datasets
import torch.utils.data as data
from utils import dist
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import InterpolationMode, transforms
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.datasets import DatasetFolder

from data.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from data.lsun_church import LSUNChurchesDataset
from data.lsun_bedroom import LSUNBedroomsDataset
from data.celeba import CelebADataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

paths = {
    "ImageNet": "ImageNet-1k",
    "CelebA": "CelebA",
    "CelebA-HQ": "CelebA-HQ/celeba_hq",
    "LSUN-Bedrooms": "LSUN-Bedrooms/data0/lsun/bedroom/",
    "LSUN-Churches": "LSUN-Churches",
    "FFHQ": "FFHQ",
    "JourneyDB": "/online1/other/dataset/JourneyDB/data/train/imgs"
}

# normalize x from [0, 1] to [-1, 1] by (x*2) - 1
def normalize(x):  
    return x.add(x).add_(-1)

class AlbumentationsDatasetFolder(DatasetFolder):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None):
        super().__init__(root, loader, extensions, transform=transform, target_transform=target_transform)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = np.array(sample).astype(np.uint8)

        if self.transform:
            sample = self.transform(image=sample)['image']
            sample = (sample / 127.5 - 1.0).to(dtype=torch.float32)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target

def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img

class Transforms:
     def __init__(self, transform):
        self.transform=transform

     def __call__(self, image):
        return self.transform(image=image)["image"]

def build_transform(args):
    final_reso = args.resolution
    transforms=A.Compose([
        A.SmallestMaxSize(max_size=final_reso),
        A.RandomCrop(height=final_reso, width=final_reso),
        ToTensorV2(transpose_mask=True)])
    return transforms

'''
def build_transform(args):
    final_reso = args.resolution
    mid_reso = round(args.resize_rate * args.resolution) 
    if args.dataset_name == "ImageNet" or args.dataset_name == "JourneyDB":
        transform = transforms.Compose([
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), 
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(), 
            normalize])
    elif args.dataset_name in ["LSUN-Bedrooms", "LSUN-Churches", "CelebA", "CelebA-HQ", "FFHQ"]:
        transform = transforms.Compose([
            transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS), 
            transforms.ToTensor(), 
            normalize])
    return transform
'''

def build_dataloader(args):
    if args.dataset_name != "JourneyDB":
        data_path = os.path.join(args.dataset_dir, paths[args.dataset_name])
    else:
        data_path = paths[args.dataset_name]

    transform = build_transform(args)

    if args.dataset_name == "ImageNet":
        train_set = AlbumentationsDatasetFolder(root=os.path.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)
        val_set = AlbumentationsDatasetFolder(root=os.path.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)
    elif args.dataset_name == "CelebA":
        train_set = CelebADataset(root=data_path, split='train', transform=transform)
        val_set = CelebADataset(root=data_path, split='val', transform=transform)
    elif args.dataset_name == "CelebA-HQ":
        train_set = AlbumentationsDatasetFolder(root=os.path.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)
        val_set = AlbumentationsDatasetFolder(root=os.path.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)
    elif args.dataset_name == "LSUN-Bedrooms":
        train_set = LSUNBedroomsDataset(root=os.path.join(args.dataset_dir, args.dataset_name), split='train', transform=transform)
        val_set = LSUNBedroomsDataset(root=os.path.join(args.dataset_dir, args.dataset_name), split='val', transform=transform)
    elif args.dataset_name == "LSUN-Churches":
        train_set = LSUNChurchesDataset(root=data_path, split='train', transform=transform)
        val_set = LSUNChurchesDataset(root=data_path, split='val', transform=transform)
    elif args.dataset_name == "FFHQ":
        train_set = AlbumentationsDatasetFolder(root=os.path.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)
        val_set = AlbumentationsDatasetFolder(root=os.path.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)
    elif args.dataset_name == "JourneyDB":
        train_set = AlbumentationsDatasetFolder(root=data_path, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)
        val_set = AlbumentationsDatasetFolder(root=data_path, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)

    if dist.is_local_master():
        print("dataset name:", args.dataset_name)
        print("len train_set:", len(train_set))
        print("len val_set:", len(val_set))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    #train_sampler = None
    train_dataloader = DataLoader(
        dataset=train_set, num_workers=args.workers, pin_memory=True,
        batch_size=args.batch_size, shuffle=False,
        sampler=train_sampler, drop_last=True
    )

    val_dataloader = DataLoader(
        val_set, num_workers=args.workers, pin_memory=True,
        batch_size=args.batch_size*2, sampler=EvalDistributedSampler(val_set, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )
    return train_dataloader, val_dataloader, train_sampler, len(train_set), len(val_set)
