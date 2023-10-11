"""
    Variational Information Distillation for Knowledge Transfer
    Introduce mutual information(MI) to match the teacher distribution and student distribution.
    Also, deriving lower bound for the MI, using KL Divergence.
    Can be applied to logits stage and intermediate feature stages.

    For LogitsBasedVID, I do not use the linear transformation, because for the original part, it uses
    the linear transformation to change the penultimate layer dimension to match the teacher logits.
    But for almost backbones, batch size is the same, so it is not necessary to apply this. So, I
    just use the student logits as pred_mean.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy

from .__base_distiller import Distiller
from .utils import get_feature_shapes


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def vid_loss(regressor, log_scale, student_feature, teacher_feature, eps):
    # pooling for dimension match
    s_H, t_H = student_feature.shape[2], teacher_feature.shape[2]
    if s_H > t_H:
        student_feature = F.adaptive_avg_pool2d(student_feature, (t_H, t_H))
    elif s_H < t_H:
        teacher_feature = F.adaptive_avg_pool2d(teacher_feature, (s_H, s_H))
    else:
        pass

    pred_mean = regressor(student_feature)
    pred_var = torch.log(1.0 + torch.exp(log_scale)) + eps  # [Teacher_Channel,]
    # [1, Teacher_Channel, 1, 1] ----> make sure the divide operation right
    pred_var = pred_var.view(1, -1, 1, 1).to(pred_mean)

    neg_log_prob = 0.5 * ((pred_mean - teacher_feature) ** 2 / pred_var + torch.log(pred_var))
    loss = torch.mean(neg_log_prob)
    return loss


def vid_logits_loss(log_scale, student_logits, teacher_logits, eps):
    pred_var = torch.log(1.0 + torch.exp(log_scale)) + eps
    pred_var = pred_var.view(-1, 1)

    neg_log_prob = 0.5 + ((teacher_logits - student_logits) ** 2) / pred_var + torch.log(pred_var)
    loss = torch.mean(neg_log_prob)
    return loss


class FeatureBasedVID(Distiller):
    def __init__(self,
                 teacher,
                 student,
                 ce_weight=1.0,
                 feature_weight=1.0,
                 eps=1e-5,
                 pred_var=5.0,
                 combined_KD=False,
                 temperature=4,
                 single_stage=False):
        super(FeatureBasedVID, self).__init__(teacher=teacher, student=student)

        self.log_scale = None
        self.regressor = None

        self.ce_weight = ce_weight
        self.feature_weight = feature_weight
        self.eps = eps
        self.pred_var = pred_var
        self.combined_KD = combined_KD
        self.temperature = temperature
        self.single_stage = single_stage

        # input size is (32, 32), if use other datasets, please re-write
        self.teacher_feature_shapes, self.student_feature_shapes = get_feature_shapes(self.teacher, self.student,
                                                                                      input_size=(32, 32))

        self.teacher_channels, self.student_channels = [], []

        for shape in self.teacher_feature_shapes:
            self.teacher_channels.append(shape[1])

        for shape in self.student_feature_shapes:
            self.student_channels.append(shape[1])

    def init_vid_modules(self):
        self.regressor = nn.ModuleList()
        self.log_scale = []

        # zip() will make the number of channel the same
        for student_channel, teacher_channel in zip(self.student_channels, self.teacher_channels):
            regressor = nn.Sequential(
                conv1x1(student_channel, teacher_channel),
                nn.ReLU(),
                nn.BatchNorm2d(teacher_channel),
                conv1x1(teacher_channel, teacher_channel),
                nn.ReLU(),
                nn.BatchNorm2d(teacher_channel),
                conv1x1(teacher_channel, teacher_channel)
            )
            self.regressor.append(regressor)

            log_scale = torch.nn.Parameter(
                # softplus(x) = log(1 + exp(x))
                np.log(np.exp(self.pred_var - self.eps) + 1.0) * torch.ones(teacher_channel)
            )
            self.log_scale.append(log_scale)

    def get_learnable_parameters(self):
        # do not add log_scale
        param = super().get_learnable_parameters()
        for regressor in self.regressor:
            param += list(regressor.parameters())
        return param

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_vid = 0
        for i in range(len(self.regressor)):
            loss_vid += vid_loss(
                regressor=self.regressor[i],
                log_scale=self.log_scale[i],
                student_feature=student_feature['features'] if not self.single_stage else student_feature['features'][
                    -1],
                teacher_feature=teacher_feature['features'] if not self.single_stage else teacher_feature['features'][
                    -1],
                eps=self.eps
            )

        loss_vid = self.feature_weight * loss_vid

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_vid": loss_vid
        }

        total_loss = loss_ce + loss_vid

        if self.combined_KD:
            from KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd

            total_loss += loss_kd

        return logits_student, loss_dict, total_loss


class LogitsBasedVID(Distiller):
    def __init__(self,
                 teacher,
                 student,
                 ce_weight=1.0,
                 feature_weight=1.0,
                 eps=1e-5,
                 pred_var=5.0,
                 combined_KD=False,
                 temperature=4):
        super(LogitsBasedVID, self).__init__(teacher=teacher, student=student)

        self.log_scale = None

        self.ce_weight = ce_weight
        self.feature_weight = feature_weight
        self.eps = eps
        self.pred_var = pred_var
        self.combined_KD = combined_KD
        self.temperature = temperature

        # input size is (32, 32), if use other datasets, please re-write
        self.teacher_feature_shapes, self.student_feature_shapes = get_feature_shapes(self.teacher, self.student,
                                                                                      input_size=(32, 32))

    def init_vid_modules(self):
        self.log_scale = []

        # for the original paper, you need to get the penultimate(倒数第二) layer, but here I do not use the linear layer
        # just directly match the dimension
        log_scale = torch.nn.Parameter(
            np.log(np.exp(self.pred_var - self.eps) + 1.0) * torch.ones(self.teacher_feature_shapes[0][0])  # [N]
        )
        self.log_scale.append(log_scale)

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_vid = self.feature_weight * vid_logits_loss(
            log_scale=self.log_scale,
            student_logits=logits_student,
            teacher_logits=logits_teacher,
            eps=self.eps
        )

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_vid": loss_vid
        }

        total_loss = loss_ce + loss_vid

        if self.combined_KD:
            from KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd

            total_loss += loss_kd

        return logits_student, loss_dict, total_loss
