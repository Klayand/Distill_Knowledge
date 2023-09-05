import torch
import kornia.augmentation as KA
from torch import nn
from generators.AutoAug import AutoAugment

in_tensor = torch.rand(5, 3, 30, 30)
# autoaug = AutoAugment()
# aug = KA.AugmentationSequential(autoaug)
# aug.train()
# aug.requires_grad_(True)
# torch.sum(aug(in_tensor)).backward()
# for name, module in aug.named_modules():
#     print(name, module)
#     for param_name, _ in module.named_parameters():
#         print(param_name)
#     print("-" * 30)
# KA.ColorJitter
from generators.ops import translate_x

aug = translate_x(1, 3)
aug(in_tensor).sum().backward()
for param_name, param in aug.named_parameters():
    print(param_name, param, param.grad)
    print("-" * 30)
