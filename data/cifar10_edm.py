import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


class CIFAR10EDM(Dataset):
    def __init__(self, img_path, gt_path=None, transform=transforms.ToTensor()):
        self.img_path = img_path
        self.images = [img for img in os.listdir(img_path) if img.endswith('png')]

        self.transforms = transform

        if gt_path:
            self.gt = np.load(gt_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(os.path.join(self.img_path, image_path))  # PIL object

        image = self.transforms(image)

        # Not conditional diffusion, so just return the image.
        return image


def get_edm_cifar10_loader(
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        transform=transforms.ToTensor()
):
    dataset = CIFAR10EDM(img_path='./resources/data/edm-cifar10', transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader




