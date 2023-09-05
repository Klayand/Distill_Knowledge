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
