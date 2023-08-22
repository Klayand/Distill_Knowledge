import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller
from.KD import KD


class FitNet(Distiller):
    """ FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, ce_weight, feature_weight, hint_layer, temperature=None):
        super(FitNet, self).__init__(student=student, teacher=teacher)

        self.ce_loss_weight = ce_weight
        self.feature_weight = feature_weight
        self.hint_layer = hint_layer
















