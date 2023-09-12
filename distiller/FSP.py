import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


def keep_spatial_same(feature_i, feature_j):
    i_h, i_w = feature_i.shape[2:]
    j_h, j_w = feature_j.shape[2:]
    min_h, min_w = min(i_h, j_h), min(i_w, j_w)

    if i_h > min_h or i_w > min_w:
        feature_i = F.adaptive_max_pool2d(feature_i, (min_h, min_w))

    if j_h > min_h or j_w > min_w:
        feature_j = F.adaptive_max_pool2d(feature_j, (min_h, min_w))

    feature_i = feature_i.flatten(2)  # [N, C, H ,W] ----》 [N, C, HW]
    feature_j = feature_j.flatten(2)

    return feature_i, feature_j


def fsp_loss(features):
    G_metric = []
    for i in range(len(features) - 1):
        feature_i, feature_j = keep_spatial_same(features[i], features[i + 1])
        metric = torch.matmul(feature_i, feature_j.transpose(1, 2)) / feature_i.shape[-1]  # [N, C, HW] * [N, HW, C`] ----> [N, C, C`]
        G_metric.append(metric)

    return torch.stack(G_metric, dim=0)


def compute_fsp_loss(teacher_features, student_features, lambda_):
    # keep the same spatial size
    teacher_metric = fsp_loss(teacher_features)
    student_metric = fsp_loss(student_features)

    # the number of the L2 loss layers must be the same
    min_len = min(teacher_metric.shape[0], student_metric.shape[0])  # [N`, N, C, C`]

    loss = 0
    batch_size = teacher_metric.shape[1]

    for i in range(min_len):
        if teacher_metric[i].shape != student_metric[i].shape:
            # if having different shape, we should transform it from [N, C, C`] ---> [N, CC`]
            teacher_output = teacher_metric[i].reshape(teacher_metric[i].shape[0], -1)
            student_output = student_metric[i].reshape(student_metric[i].shape[0], -1)
        else:
            teacher_output = teacher_metric[i]
            student_output = student_metric[i]

        loss += lambda_ * torch.dist(teacher_output, student_output).sum() / batch_size

    return loss


class FPS(Distiller):
    """A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning"""

    def __init__(self,
                 teacher,
                 student,
                 ce_weight=1.0,
                 lambda_=3,
                 combined_KD=False,
                 temperature=4,
    ):
        super(FPS, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.lambda_ = lambda_
        self.combine_KD = combined_KD
        self.temperature = temperature

    def forward_train(self, imgae, target, **kwargs):
        logits_student, student_feature = self.student(imgae)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(imgae)

        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_fsp = compute_fsp_loss(
            teacher_features=teacher_feature["features"][1:],
            student_features=student_feature["features"][1:],
            lambda_=self.lambda_
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_fsp': loss_fsp
        }

        total_loss = loss_ce + loss_fsp
        if self.combine_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd

        return logits_student, loss_dict, total_loss