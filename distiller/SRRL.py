"""
   Knowledge Distillation via Softmax Regression Representation Learning.
   Align feature + Use frozen teacher classifier.

   Still waiting for reproducing different versions of L_sr, L_fm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller

class SRRL(Distiller):
    def __init__(self, teacher, student, ce_weight=1.0,
                 alpha=, beta=, ):