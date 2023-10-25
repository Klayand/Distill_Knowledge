"""
    @Author: Zikai Zhou
    All backbones have been verified by Zikai Zhou for Knowledge Distillation task.
    Thanks for the open source of Megvii-Research
"""

from .mobilenetv2 import *
from .resnet import *
from .resnet_imagenet import *
from .resnetv2 import *

# from .RSC import *
from .ShuffleNetv1 import *
from .ShuffleNetv2 import *
from .vgg import *
from .wrn import *
from .PyramidNet import *
from .ConvNet import *
from .batchensemble import *
from .NormBasedModel import (
    CIFARNormModel,
    ImageNetNormModel,
    MnistNormModel,
    MnistmNormModel,
    PacsNormModel,
    SvhnNormModel,
    UspsNormModel,
)

model_dict = {
    "mobilenetV2": mobilenetV2,
    "mobilenetv2_T_w": mobilenetv2_T_w,
    "ShuffleV1": ShuffleV1,
    "ShuffleV2": ShuffleV2,
    "resnet8": resnet8,
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet8x4": resnet8x4,
    "resnet32x4": resnet32x4,
    "resnet14x05": resnet14x05,
    "resnet20x05": resnet20x05,
    "resnet20x0375": resnet20x0375,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "resnet18_imagenet": resnet18_imagenet,
    "resnet34_imagenet": resnet34_imagenet,
    "resnet50_imagenet": resnet50_imagenet,
    "vgg8": vgg8,
    "vgg8_bn": vgg8_bn,
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn,
    "pyramidnet272": pyramidnet272,
    "pyramidnet164": pyramidnet164,
    "wrn": wrn,
    "wrn_16_1": wrn_16_1,
    "wrn_16_2": wrn_16_2,
    "wrn_16_4": wrn_16_4,
    "wrn_40_1": wrn_40_1,
    "wrn_40_2": wrn_40_2,
    "wrn_40_4": wrn_40_4,
    "wrn_40_10": wrn_40_10,
}
