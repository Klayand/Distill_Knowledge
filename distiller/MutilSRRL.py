"""
    An expansion of SRRL. Using multi-student features and multi-teacher feature.
    Proposer: Zikai Zhou
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller
from .utils import get_feature_shapes
from .KD import kd_loss



