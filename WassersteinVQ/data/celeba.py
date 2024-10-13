import os
import torch
import pandas as pd
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image

class CelebADataset(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root (string): Directory with all the images and annotations.
            split (string): One of 'train', 'val', or 'test' to select the split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform

        # Load the partition file which indicates the split of each image
        self.partition = pd.read_csv(os.path.join(root, 'list_eval_partition.csv'),)
        # Load attribute labels
        self.attributes = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'), )
        # print(self.partition.iloc[:, 0].values)
        
        # Filter the dataframe based on the split
        if split == 'train':
            self.data = self.attributes[self.partition.iloc[:, 1] == 0]
        elif split == 'val':
            self.data = self.attributes[self.partition.iloc[:, 1] == 1]
        elif split == 'test':
            self.data = self.attributes[self.partition.iloc[:, 1] == 2]
        else:
            raise ValueError("Invalid split type. Choose from 'train', 'val', 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, 'img_align_celeba', self.data.iloc[idx, 0])
        image = Image.open(img_name)

        # Extract attributes as a tensor
        attributes = (torch.tensor(self.data.iloc[idx, 1:].values.astype(int)) > 0).int()

        if self.transform:
            image = self.transform(image)

        return image, attributes

