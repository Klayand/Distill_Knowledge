import torch
from kornia import augmentation as KA
from torch.utils.tensorboard import SummaryWriter

from generators.AutoAug import AutoAugment
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

p = lambda x: nn.Parameter(torch.tensor(x))


class DifferentiableAutoAug(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(DifferentiableAutoAug, self).__init__()
        self.aug = KA.AugmentationSequential(AutoAugment(policy="cifar10"))
        self.device = device

    def forward(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        original_x = x.clone()
        original_y = y.clone()

        x = self.aug(x)

        # return x, y
        return torch.cat([x.detach(), original_x], dim=0), torch.cat([y.detach(), original_y], dim=0)
