import torch
from torch import nn


def Adam(model: nn.Module, lr=1e-3) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)


def SGD(model: nn.Module, lr=0.1) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)


def AdamW(model: nn.Module, lr=1e-3) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr)


def RMSprop(model: nn.Module, lr=1e-3) -> torch.optim.Optimizer:
    return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)


def Adagrad(model: nn.Module, lr=1e-3) -> torch.optim.Optimizer:
    return torch.optim.Adagrad(model.parameters(), lr=lr)
