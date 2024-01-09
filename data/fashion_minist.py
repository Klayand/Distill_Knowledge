from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

__all__ = ["get_fashion_mnist_test", "get_fashion_mnist_train"]


def get_fashion_mnist_train(
    batch_size=256,
    num_workers=40,
    pin_memory=True,
):
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    set = FashionMNIST("./resources/data/fashion_mnist/", train=True, download=True, transform=transform)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    return loader


def get_fashion_mnist_test(
    batch_size=256,
    num_workers=40,
    pin_memory=True,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    set = FashionMNIST("./resources/data/fashion_mnist/", train=False, download=True, transform=transform)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return loader
