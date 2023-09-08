import torch
from torch import nn, Tensor
from abc import abstractmethod


class BaseSampler(nn.Module):
    def __init__(self, unet: nn.Module,
                 device=torch.device('cuda')):
        super(BaseSampler, self).__init__()
        self.unet = unet
        self.device = device
        self._model_init()

    def _model_init(self):
        self.eval().requires_grad_(False).to(self.device)

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
