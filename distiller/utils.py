"""
    ref from megvii-research
    verified by Zikai Zhou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReg(nn.Module):
    """ Convolutional Regression for FitNet"""

    def __init__(self, student_shape, teacher_shape, use_relu=True):
        super(ConvReg, self).__init__()
