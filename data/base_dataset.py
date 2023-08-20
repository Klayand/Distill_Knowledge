"""
    This file is used to create specific dataset:
    1. all images are in one folder
    2. only a dict to store ground truth. Key are image names, values are ground truth labels.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, image_path, groudtruth_path):
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.AutoAugment(),
            transforms.ToTensor(),
            # calculate the mean and std from the whole dataset
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])

        self.image_path = image_path

        # file postfix can be rewritten
        self.images = [img for img in os.listdir(self.image_path) if img.endswith('.jpg')]
        self.gt = np.load(groudtruth_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = Image.open(os.path.join(self.image_path, image_name))  # PIL

        return self.transform(image), self.gt[image_name]


















