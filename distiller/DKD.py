"""
    From https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py official repo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


def get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()

    return mask


def get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()

    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)

    return rt


def dkd_loss(student_logits, teacher_logits, target,
             alpha, beta, temperature):
    # follow the dim=1 to compute the logits
    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)

    gt_mask = get_gt_mask(student_logits, target)
    other_mask = get_other_mask(student_logits, target)

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)

    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2) / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        teacher_logits / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        student_logits / temperature - 1000.0 * gt_mask, dim=1
    )

    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature ** 2) / target.shape[0]
    )

    return alpha * tckd_loss + beta * nckd_loss


class DKD(Distiller):
    """ Decoupled Knowledge Distillation """

    def __init__(self, teacher, student, ce_weight=1.0,
                 alpha=1.0, beta=8.0, temperature=4):
        super(DKD, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_dkd = dkd_loss(
            student_logits=logits_student,
            teacher_logits=logits_teacher,
            target=target,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_dkd': loss_dkd
        }

        total_loss = loss_ce + loss_dkd

        return logits_student, loss_dict, total_loss
