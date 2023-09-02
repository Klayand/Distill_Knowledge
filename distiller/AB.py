"""
    Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


def ab_loss(teacher_feature, student_feature, margin):
    """
    Piecewise differentiable loss approximating the activation boundaries loss.

    Guide the student learns a separating boundary between activation region
    and deactivation region formed by each neuron in the teacher.
    Args:
        teacher_feature: teacher featuremap.
        student_feature: student featuremap.
        margin: relaxation ofr training stability.

    From https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/models/losses/ab_loss.py
    """
    batch_size = student_feature.shape[0]

    loss = (
            (student_feature + margin)**2 * ((student_feature > -margin) &
                                             (teacher_feature <= 0)).float() +
            (student_feature - margin)**2 * ((student_feature <= margin) &
                                             (teacher_feature > 0)).float()
            )
    loss = torch.abs(loss).sum() / batch_size / 1000 * 3

    return loss


class FullyConnectorAB(Distiller):
    def __init__(self, teacher, student, ce_weight=1.0, ab_weight=1.0,
                 margin=1.0, combined_KD=False, temperature=None):
        super(FullyConnectorAB, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.ab_weight = ab_weight
        self.margin = margin
        self.temperature = temperature
        self.combined_KD = combined_KD

        self.linear_layer = None

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        teacher_featuremap = teacher_feature['features'][-1]
        student_featuremap = student_feature['features'][-1]

        if (teacher_featuremap.shape[1] == student_featuremap.shape[1]
            and
            teacher_featuremap.shape[2] * teacher_featuremap.shape[3] != student_featuremap.shape[2] * student_featuremap.shape[3]):

            self.linear_layer = nn.Linear(student_featuremap.shape[2] * student_featuremap.shape[3], teacher_featuremap.shape[2] * teacher_featuremap.shape[3])
            student_featuremap = self.linear_layer(student_featuremap)

        elif teacher_featuremap.shape[1] != student_featuremap.shape[1]:
            if teacher_featuremap.shape[2] * teacher_featuremap.shape[3] == student_featuremap.shape[2] * student_featuremap.shape[3]:
               raise NotImplementedError("You should use ConvolutionAB")
            else:
                raise NotImplementedError("Both channel dimension and spatial dimension are not the same")

        loss_ab = self.ab_weight * ab_loss(
            teacher_feature=teacher_featuremap,
            student_feature=student_featuremap,
            margin=self.margin
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_ab': loss_ab
        }

        total_loss = loss_ce + loss_ab

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.linear_layer.parameters())

    def get_extra_parameters(self):
        num_parameters = 0
        for param in self.linear_layer.parameters():
            num_parameters += param.numel()
        return num_parameters


class ConvolutionAB(Distiller):
    def __init__(self, teacher, student, ce_weight=1.0, ab_weight=1.0,
                 margin=1.0, combined_KD=False, temperature=None):
        super(ConvolutionAB, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.ab_weight = ab_weight
        self.margin = margin
        self.temperature = temperature
        self.combined_KD = combined_KD

        self.conv_layer = None

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        teacher_featuremap = teacher_feature['features'][-1]
        student_featuremap = student_feature['features'][-1]

        if (teacher_featuremap.shape[2] * teacher_featuremap.shape[3] == student_featuremap.shape[2] * student_featuremap.shape[3]
            and
            teacher_featuremap.shape[1] != student_featuremap.shape[1]):

            self.conv_layer = nn.Conv2d(student_featuremap.shape[1], teacher_featuremap.shape[1], kernel_size=(1, 1))
            student_featuremap = self.conv_layer(student_featuremap)

        elif teacher_featuremap.shape[2] * teacher_featuremap.shape[3] != student_featuremap.shape[2] * student_featuremap.shape[3]:
            if teacher_featuremap.shape[1] == student_featuremap.shape[1]:
                raise NotImplementedError("You should use FullyConnectorAB")
            else:
                raise NotImplementedError("Both channel dimension and spatial dimension are not the same")

        loss_ab = self.ab_weight * ab_loss(
            teacher_feature=teacher_featuremap,
            student_feature=student_featuremap,
            margin=self.margin
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_ab': loss_ab
        }

        total_loss = loss_ce + loss_ab

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_layer.parameters())

    def get_extra_parameters(self):
        num_parameters = 0
        for param in self.conv_layer.parameters():
            num_parameters += param.numel()
        return num_parameters
