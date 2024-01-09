import numpy as np
from PIL import Image
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Tuple
import random
import torch
import os

__all__ = ["get_imagenet10_loader", "get_imagenet_loader"]


class ImageNet10(ImageNet):
    def __init__(self, *args, target_class: Tuple[int], maximal_images: int or None = None, **kwargs):
        super(ImageNet10, self).__init__(*args, **kwargs)
        self.target_class = list(target_class)
        result = []
        for x, y in self.samples:
            if y in self.target_class:
                result.append((x, y))
        random.shuffle(result)
        self.maximal_images = maximal_images
        self.samples = result

    def __len__(self):
        if self.maximal_images is not None:
            return self.maximal_images
        return len(self.samples)


class TinyImageNet(Dataset):
    def __init__(self, root='resources/data/tiny-imagenet', mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        if self.mode == 'train':
            data = np.load(os.path.join(self.root, 'train.npz'))
            self.images = data['images']
            self.labels = data['labels']

        elif self.mode == 'test':
            data = np.load(os.path.join(self.root, 'test.npz'))
            self.images = data['images']
            self.labels = data['labels']

        else:
            raise ValueError('Invalid mode. Mode should be train, or test.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_imagenet_loader(
    root="./resources/data/ImageNet/",
    split="val",
    batch_size=1,
    num_workers=4,
    pin_memory=False,
    shuffle=False,
    transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ]),
    ddp=False
):
    assert split in ["val", "train"]
    set = ImageNet(root, split, transform=transform)

    # train transform
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((256, 256)),
    #         # transforms.AutoAugment(transforms.AutoAugmentPolicy),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ]
    # )

    if ddp:
        ddp_sampler = torch.utils.data.distributed.DistributedSampler(set)
        loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            sampler=ddp_sampler)
    else:
        loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    return loader


def get_imagenet10_loader(
    target_class=(0, 100, 200, 300, 400, 500, 600, 700, 800, 900),
    maximum_images=None,
    root="./resources/data/ImageNet/",
    split="val",
    batch_size=1,
    num_workers=4,
    pin_memory=False,
    shuffle=False,
    transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ]),
    ddp=False
):
    assert split in ["val", "train"]
    set = ImageNet10(root, split, target_class=target_class, transform=transform, maximal_images=maximum_images)

    if ddp:
        ddp_sampler = torch.utils.data.distributed.DistributedSampler(set)
        loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            sampler=ddp_sampler)
    else:
        loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    return loader


def get_tinyImageNet_loader(
    root="./resources/data/ImageNet/",
    mode="train",
    batch_size=1,
    num_workers=4,
    pin_memory=False,
    shuffle=False,
    transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ]),
    ddp=False
):
    assert mode in ["test", "train"]

    set = TinyImageNet(root, mode, transform=transform)

    if ddp:
        ddp_sampler = torch.utils.data.distributed.DistributedSampler(set)
        loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle,
                            sampler=ddp_sampler)
    else:
        loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    return loader




