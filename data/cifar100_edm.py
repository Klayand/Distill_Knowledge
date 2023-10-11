import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


class CIFAR100EDM(Dataset):
    def __init__(self, npz_path, transform=transforms.ToTensor()):
        self.npz_path = npz_path

        self.data = np.load(self.npz_path)
        self.images, self.labels = self.data['image'], self.data['label']

        self.transforms = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image_array = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image_array)  # PIL object
        image = self.transforms(image)

        return image, label


def get_edm_cifar100_loader(
        batch_size=256,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        transform=transforms.ToTensor(),
        ddp=False,
):
    dataset = CIFAR100EDM(npz_path='../resources/data/edm-cifar100.npz', transform=transform)

    if ddp:
        ddp_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=ddp_sampler
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return dataloader
