"""
    Correlation Congruence for Knowledge Distillation
        introduce instances congruence between teacher and student, behaving like KD
        introduce instances congruence between samples in teacher and student, behaving like NST
        However, in NST, it uses features and logits. In CCKD, it just uses logits?????.

        loss function can be reconstructed.

        import torch

        tensor = torch.arange(0, 15).reshape(3, -1)
        print(tensor)

        row_list = []
        for index, i in enumerate(tensor):
            row_list.append(i)
        print(torch.stack(row_list))
        print(torch.cat(row_list))

        enumerate(tensor) 能拿到第一个维度的张量 [N, P] ---> [1, P]


        torch.stack()   在新的维度进行叠加
        eg: tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  6,  7,  8,  9],
                    [10, 11, 12, 13, 14]])
            tensor([0, 1, 2, 3, 4])
            tensor([5, 6, 7, 8, 9])
            tensor([10, 11, 12, 13, 14])
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  6,  7,  8,  9],
                    [10, 11, 12, 13, 14]])

        torch.cat()  在指定的已有的维度进行叠加
        eg: tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  6,  7,  8,  9],
                    [10, 11, 12, 13, 14]])
            tensor([0, 1, 2, 3, 4])
            tensor([5, 6, 7, 8, 9])
            tensor([10, 11, 12, 13, 14])
            tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller
from .KD import kd_loss


def compute_binlinear_loss(teacher_logits, student_logits, batch_size):
    teacher_metric = torch.matmul(teacher_logits, torch.t(teacher_logits))
    student_metric = torch.matmul(student_logits, torch.t(student_logits))

    cc_loss = torch.dist(student_metric, teacher_metric, 2)
    return cc_loss / (batch_size ** 2)


def compute_guassin_rbf(teacher_logits, student_logits, batch_size, p, gamma):
    teacher_row_list = list()
    student_row_list = list()

    for index, teacher_output in enumerate(teacher_logits):
        teacher_row = 1
        right_term = torch.matmul(teacher_output, torch.t(teacher_output))  # [1, N]
        for p in range(1, p + 1):
            left_term = ((2 * gamma) ** p) / (math.factorial(p))
            teacher_row += left_term * (right_term ** p)

        teacher_row *= math.exp(-2 * gamma)
        teacher_row = torch.tensor(teacher_row)
        teacher_row_list.append(teacher_row.squeeze(0)) # [N]

    teacher_metric = torch.stack(teacher_row_list)  # [NxN]  ---> [N ,N]

    for index, student_output in enumerate(student_logits):
        student_row = 1
        right_term = torch.matmul(student_output, torch.t(student_output))  # [1, N]
        for p in range(1, p + 1):
            left_term = ((2 * gamma) ** p) / (math.factorial(p))
            student_row += left_term * (right_term ** p)

        student_row *= math.exp(-2 * gamma)
        student_row = torch.tensor(student_row)
        student_row_list.append(student_row.squeeze(0))  # [N]

    student_metric = torch.stack(student_row_list)  # [NxN]  ---> [N ,N]

    cc_loss = torch.dist(student_metric, teacher_metric, 2)
    return cc_loss / (batch_size ** 2)


def compute_new_mmd_loss(teacher_logits, student_logits, batch_size):
    teacher_list = []
    student_list = []

    for index, instance in enumerate(teacher_logits):
        instance = instance.mean(-1)  # [1]
        for index, other_instance in enumerate(teacher_logits):
            sample_value = torch.abs(instance - other_instance.mean(-1))
            teacher_list.append(sample_value)

    teacher_metric = torch.stack(teacher_list).reshape(teacher_logits.shape[0], -1)

    for index, instance in enumerate(student_logits):
        instance = instance.mean(-1)  # [1]
        for index, other_instance in enumerate(student_logits):
            sample_value = torch.abs(instance - other_instance.mean(-1))
            student_list.append(sample_value)

    student_metric = torch.stack(student_list).reshape(student_logits.shape[0], -1)  # [1, 1, ...., 1] ---> [N, N]

    cc_loss = torch.dist(student_metric, teacher_metric, 2)
    return cc_loss / (batch_size ** 2)


def compute_mmd_loss(teacher_logits, student_logits, batch_size):
    teacher_expanded_tensor = teacher_logits.unsqueeze(1)  # [N, 1, P]
    teacher_expanded_tensor = teacher_expanded_tensor.expand(-1, teacher_logits.shape[0], -1)  # [N, N, P]
    student_expanded_tensor = student_logits.unsqueeze(1).expand(-1, student_logits.shape[0], -1)

    diff_teacher_tensor = torch.abs(teacher_expanded_tensor - teacher_logits).sum(-1)  # [N, N]
    diff_student_tensor = torch.abs(student_expanded_tensor - student_logits).sum(-1)  # [N, N]

    cc_loss = torch.dist(diff_student_tensor, diff_teacher_tensor, 2)
    return cc_loss / (batch_size ** 2)


class MMDCCKD(Distiller):
    def __init__(self, teacher, student,
                 alpha=0.0, beta=0.003,
                 temperature=4):
        super(MMDCCKD, self).__init__(teacher=teacher, student=student)

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward_train(self, image, target, **kwargs):
        batch_size = image.shape[0]

        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.alpha * F.cross_entropy(logits_student, target)
        loss_kd = (1 - self.alpha) * kd_loss(logits_student, logits_teacher, temperature=self.temperature)

        loss_cckd = self.beta * compute_mmd_loss(
            teacher_logits=logits_teacher,
            student_logits=logits_student,
            batch_size=batch_size
        )

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_cckd": loss_cckd
        }

        total_loss = loss_ce + loss_kd + loss_cckd

        return logits_student, loss_dict, total_loss


class BilinearCCKD(Distiller):
    def __init__(self, teacher, student,
                 alpha=0.0, beta=0.003,
                 temperature=4):
        super(BilinearCCKD, self).__init__(teacher=teacher, student=student)

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward_train(self, image, target, **kwargs):
        batch_size = image.shape[0]

        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.alpha * F.cross_entropy(logits_student, target)
        loss_kd = (1 - self.alpha) * kd_loss(logits_student, logits_teacher, temperature=self.temperature)

        loss_cckd = self.beta * compute_binlinear_loss(teacher_logits=logits_teacher, student_logits=logits_student,
                                                       batch_size=batch_size)

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_cckd": loss_cckd
        }

        total_loss = loss_ce + loss_kd + loss_cckd

        return logits_student, loss_dict, total_loss


class GaussianRBF(Distiller):
    def __init__(self, teacher, student,
                 alpha=0.0, beta=0.003,
                 p=2, gamma=0.4,
                 temperature=4):
        super(GaussianRBF, self).__init__(teacher=teacher, student=student)

        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.gamma = gamma
        self.temperature = temperature

    def forward_train(self, image, target, **kwargs):
        batch_size = image.shape[0]

        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.alpha * F.cross_entropy(logits_student, target)
        loss_kd = (1 - self.alpha) * kd_loss(logits_student, logits_teacher, temperature=self.temperature)

        loss_cckd = self.beta * compute_guassin_rbf(
            teacher_logits=logits_teacher,
            student_logits=student_feature,
            batch_size=batch_size,
            p=self.p,
            gamma=self.gamma
        )

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_cckd": loss_cckd
        }

        total_loss = loss_ce + loss_kd + loss_cckd

        return logits_student, loss_dict, total_loss
