import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


def wsl_loss(teacher_logits, student_logits, target, temperature, num_classes):
    soft_teacher_logits = teacher_logits / temperature
    soft_student_logits = student_logits / temperature

    teacher_probs = F.softmax(soft_teacher_logits, dim=1)

    ce_loss = -torch.sum(teacher_probs * F.log_softmax(soft_student_logits, dim=1), dim=1, keepdim=True)

    student_logits_detach = student_logits.detach()
    teacher_logits_detach = teacher_logits.detach()

    log_softmax_student = F.log_softmax(student_logits_detach, dim=1)
    log_softmax_teacher = F.log_softmax(teacher_logits_detach, dim=1)

    one_hot_labels = F.one_hot(target, num_classes=num_classes).float()

    ce_loss_student = -torch.sum(one_hot_labels * log_softmax_student, dim=1, keepdim=True)
    ce_loss_teacher = -torch.sum(one_hot_labels * log_softmax_teacher, dim=1, keepdim=True)

    focal_weight = ce_loss_student / (ce_loss_teacher + 1e-7)
    ratio_lower = torch.zeros_like(focal_weight)

    focal_weight = torch.max(focal_weight, ratio_lower)
    focal_weight = 1 - torch.exp(-focal_weight)

    ce_loss = focal_weight * ce_loss

    loss = (temperature**2) * torch.mean(ce_loss)

    return loss


class WSLD(Distiller):
    """Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective"""

    def __init__(self, teacher, student, ce_weight=1.0, alpha=2.5, temperature=2, num_classes=100):
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

        loss_wsl = self.alpha * wsl_loss(
            teacher_logits=logits_teacher,
            student_logits=logits_student,
            target=target,
            temperature=self.temperature,
            num_classes=self.num_classes,
        )

        loss_dict = {"loss_ce": loss_ce, "loss_wsl": loss_wsl}

        total_loss = loss_ce + loss_wsl

        return logits_student, loss_dict, total_loss
