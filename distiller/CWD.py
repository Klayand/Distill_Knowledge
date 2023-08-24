import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller


class ChannelWiseDivergence(Distiller):
    pass