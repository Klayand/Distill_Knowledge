"""
    From official repo:
        https://github.com/hunto/image_classification_sota/blob/d9662f7df68fe46b973c4580b7c9b896cedcd301/lib/models/losses/dist_kd.py
        introduce intra loss and inter loss, all use logits
        vanilla KD only considers the inter loss between each sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(
        a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps
    )


def inter_class_relation(teacher_logits, student_logits):
    return 1 - pearson_correlation(student_logits, teacher_logits).mean()


def intra_class_relation(teacher_logits, student_logits):
    return inter_class_relation(teacher_logits.transpose(0, 1), student_logits.transpose(0, 1))


class DIST(Distiller):
    """ Knowledge Distillation from A Stronger Teacher """
    def __init__(self, teacher, student, ce_weight=1.0,
                 beta=1.0, gamma=1.0, temperature=4.0):
        super(DIST, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        teacher_logits = (logits_teacher / self.temperature).softmax(dim=1)
        student_logits = (logits_student / self.temperature).softmax(dim=1)

        loss_inter = self.beta * (self.temperature**2) * inter_class_relation(teacher_logits, student_logits)
        loss_intra = self.gamma * (self.temperature**2) * intra_class_relation(teacher_logits, student_logits)

        loss_dist = loss_intra + loss_inter

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_inter': loss_inter,
            'loss_intra': loss_intra,
            'loss_dist': loss_dist
        }

        total_loss = loss_dist + loss_ce

        return logits_student, loss_dict, total_loss