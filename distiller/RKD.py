"""
    Original author: mmrazor
    verified by Zikai Zhou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller


def compute_angle(pred):
    """
        Calculate the angle-wise relational potential which measures the angle formed
        by the three examples in the output representation space.
    """

    pred_vec = pred.unsqueeze(0) - pred.unsqueeze(1) # (N, N, C)
    norm_pred_vec = F.normalize(pred_vec, p=2, dim=2)

    # torch.bmm: mat1: (N, M, P) mat2: (N, P, S) ---> (N, M, S)
    angle = torch.bmm(norm_pred_vec,
                      norm_pred_vec.transpose(1, 2)).view(-1) # (N*N*N, )

    return angle


def angle_wise_loss(teacher_logits, student_logits, with_l2_norm=True):
    """ Calculate the angle-wise distillation loss"""

    student_logits = student_logits.view(student_logits.shape[0], -1)
    teacher_logits = teacher_logits.view(teacher_logits.shape[0], -1)

    if with_l2_norm:
        student_logits = F.normalize(student_logits, p=2, dim=1)
        teacher_logits = F.normalize(teacher_logits, p=2, dim=1)

    teacher_angle = compute_angle(teacher_logits)
    student_angle = compute_angle(student_logits)

    return F.smooth_l1_loss(student_angle, teacher_angle)


def euclidean_distance(pred, squared=False, eps=1e-12):
    """
    Calculate the Euclidean distance between the two examples in the output representation space

    Args:
        pred(torch.Tensor): The prediction of the teacher or student with shape (N, classes)
        square(bool): Whether to calculate the squred Euclidean distances. Defaults to False.
        eps(float): The minimum Euclidean distance between the two examples. Defaults to 1e-12.
    """
    pred_squre = pred.pow(2).sum(dim=-1) # (N,)
    prod = torch.mm(pred, pred.t()) # (N, N)
    distance = (pred_squre.unsqueeze(1) + pred_squre.unsqueeze(0)
                - 2 * prod).clamp(min=eps)  # (N, N)

    if not squared:
        distance = distance.sqrt()

    distance = distance.clone()
    distance[range(len(prod)), range(len(prod))] = 0

    return distance


def distance_wise_loss(teacher_logits, student_logits, squared=False, with_l2_norm=True):
    """ Calculate distance-wise distillation loss. """

    student_logits = student_logits.view(student_logits.shape[0], -1)
    teacher_logits = teacher_logits.view(teacher_logits.shape[0], -1)

    if with_l2_norm:
        student_logits = F.normalize(student_logits, p=2, dim=1)
        teacher_logits = F.normalize(teacher_logits, p=2, dim=1)

    teacher_distance = euclidean_distance(teacher_logits, squared)

    # mu is a normalization factor for distance
    mu_t = teacher_distance[teacher_distance > 0].mean()
    teacher_distance = teacher_distance / mu_t

    student_distance = euclidean_distance(student_logits, squared)
    mu_s = student_distance[student_distance > 0].mean()
    student_distance = student_distance / mu_s

    return F.smooth_l1_loss(student_distance, teacher_distance)  # Huber loss


class DistanceWiseRKD(Distiller):
    """ Relational Knowledge Distillation for Distance-Wise function """

    def __init__(self, teacher, student, ce_weight, rkd_weight,
            temperature=None, with_l2_norm=True, squared=False):
        super(DistanceWiseRKD, self).__init__(student=student, teacher=teacher)
        self.ce_weight = ce_weight
        self.rkd_weight = rkd_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.squared = squared

    def forward_train(self, image, target,
                      combined_KD=False, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_distance_wise = self.rkd_weight * distance_wise_loss(
            teacher_logits=logits_teacher,
            student_logits=logits_student,
            squared=self.squared,
            with_l2_norm=self.with_l2_norm
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_distance_wise': loss_distance_wise,
        }

        total_loss = loss_ce + loss_distance_wise

        if combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss


class AngleWiseRKD(Distiller):
    """ Relational Knowledge Distillation for Angle-Wise function """

    def __init__(self, teacher, student, ce_weight, rkd_weight,
                 temperature=None, with_l2_norm=True):
        super(AngleWiseRKD, self).__init__(student=student, teacher=teacher)
        self.ce_weight = ce_weight
        self.rkd_weight = rkd_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm

    def forward_train(self, image, target,
                      combined_KD=False, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_angle_wise = self.rkd_weight * angle_wise_loss(
            teacher_logits=logits_teacher,
            student_logits=logits_student,
            with_l2_norm=self.with_l2_norm
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_angle_wise': loss_angle_wise,
        }

        total_loss = loss_ce + loss_angle_wise

        if combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss