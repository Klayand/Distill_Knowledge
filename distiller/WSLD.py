import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


class WSLD(Distiller):
    """ Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective """

    def __init__(self, teacher, student, ce_weight=1.0,
                 alpha=2.5, temperature=2, num_classes=100):
        super(WSLD, self).__init__(teacher=teacher, student=student)
        """
        Args:
            ce_weight(float): CE loss coefficient.
            alpha(float):   WSL loss coefficient.
            temperature(int): Temperature coefficient.
            num_classes(int): Default 1000
        """
        self.ce_weight = ce_weight
        self.alpha = alpha
        self.temperature = temperature
        self.num_classes = num_classes

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)


        loss_dict = {
            'loss_ce': loss_ce,
            'loss_wsl': loss_wsl
        }

        total_loss = loss_ce + loss_wsl

        return logits_student, loss_dict, total_loss