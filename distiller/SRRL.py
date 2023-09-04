"""
    Knowledge Distillation via Softmax Regression Representation Learning.
    Align feature + Use frozen teacher classifier.
    From original paper, we know it only applies on one layer, for multi-layers, it will hurt the accuracy.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller
from .utils import get_feature_shapes
from .KD import kd_loss


def sr_loss(teacher_logits, student_logits, method: str = "mse", temperature=4):
    if method == "mse":
        loss = F.mse_loss(student_logits, teacher_logits)
    elif method == "ce":
        # here should use teacher_logits or groudtruth?
        loss = F.cross_entropy(student_logits, teacher_logits)
    elif method == "kl":
        # here should directly use kd_loss or just like the equation below?
        # loss = F.kl_div(
        #     student_logits / temperature, teacher_logits / temperature
        # )
        loss = kd_loss(student_logits, teacher_logits, temperature)

    return loss


def fm_loss(teacher_feature, student_feature):
    # normalized or not
    teacher_feature = teacher_feature.view(teacher_feature.size(0), -1)
    student_feature = student_feature.view(student_feature.size(0), -1)

    loss = (teacher_feature.mean(dim=1) - student_feature.mean(dim=1)).pow(2)

    return loss.mean()


class TransferConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_feature = in_dim
        self.out_feature = out_dim

        self.Connectors = nn.Sequential(
            nn.Conv2d(self.in_feature, self.out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_feature),
            nn.ReLU(),
        )

        # initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.Connectors(x)
        return x

    def get_learnable_parameters(self):
        return list(self.Connectors.parameters())

    def get_number_of_parameters(self):
        num_parameters = 0
        for param in self.Connectors.parameters():
            num_parameters += param.numel()
        return num_parameters


class SRRL(Distiller):
    def __init__(self, teacher, student, ce_weight=1.0, alpha=1.0, beta=1.0, method: str = "mse"):
        super(SRRL, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.alpha = alpha
        self.beta = beta
        self.method = method

        # for other dataset, it needs to be rewritten
        self.teacher_shapes, self.student_shapes = get_feature_shapes(teacher, student, input_size=(32, 32))
        self.teacher_shape, self.student_shape = self.teacher_shapes[-1][1], self.student_shapes[-1][1]

        self.connector = TransferConv(self.student_shape, self.teacher_shape)

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        teacher_featuremap = teacher_feature["features"][-1]
        student_featuremap = student_feature["features"][-1]

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_fm = self.alpha * fm_loss(teacher_featuremap.detach(), student_featuremap)

        regression_student_featuremap = self.connector(student_featuremap)
        student_pred_teacher_net = self.teacher(x=None, is_srrl=regression_student_featuremap)

        # get logits, instead of probabilities
        loss_sr = self.beta * sr_loss(logits_teacher, student_pred_teacher_net, method=self.method)

        loss_dict = {"loss_ce": loss_ce, "loss_fm": loss_fm, "loss_sr": loss_sr}

        total_loss = loss_ce + loss_fm + loss_sr

        return logits_student, loss_dict, total_loss

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.connector.get_learnable_parameters()

    def get_extra_parameters(self):
        return self.connector.get_number_of_parameters()
