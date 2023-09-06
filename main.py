import torch
import kornia.augmentation as KA
from torch import nn
from generators.AutoAug import AutoAugment

in_tensor = torch.rand(6, 3, 30, 30)
autoaug = AutoAugment()
aug = KA.AugmentationSequential(autoaug)
aug.train()
aug.requires_grad_(True)
torch.sum(aug(in_tensor)).backward()
for name, module in aug.named_parameters():
    print(name, module.grad)

