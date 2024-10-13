import os
import pickle
from torch.utils.data import Dataset
from torchvision.datasets.lsun import LSUNClass
from PIL import Image
import io

class LSUNChurchesDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        Initialize the dataset.

        Parameters:
        root (str): Path to the LMDB database.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = os.path.join(root, f'church_outdoor_{split}_lmdb')
        self.split = split
        self.transform = transform
        self.data = LSUNClass(self.root, transform=transform)

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an image from the dataset.

        Parameters:
        idx (int): Index of the data to load.

        Returns:
        image: Image loaded from the database.
        """
        image, _ = self.data[idx]

        return image, 0
