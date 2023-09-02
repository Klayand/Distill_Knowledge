"""
    Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


class FullyConnectedAB(Distiller):
    def __init__(self, teacher, student, ce_weight=1.0, ab_weight=1.0,
                 margin=1.0, combined_KD=False, temperature=None, single_stage=True):
        super(FullyConnectedAB, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.ab_weight = ab_weight
        self.margin = margin
        self.temperature = temperature
        self.combined_KD = combined_KD
        self.single_stage = single_stage

        self.linear_layer = None


    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_ab = self.ab_weight * ab_loss(
            teacher_feature=teacher_feature['features'][-1] if self.single_stage else teacher_feature['features'][1:],
            student_feature=student_feature['features'][-1] if self.single_stage else student_feature['features'][1:],
            target=target,
            single_stage=self.single_stage
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


class ConvolutionAB(Distiller):
    def __init__(self, teacher, student, ce_weight=1.0, ab_weight=1.0,
                 margin=1.0, combined_KD=False, temperature=None, single_stage=True):
        super(ConvolutionAB, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.ab_weight = ab_weight
        self.margin = margin
        self.temperature = temperature
        self.combined_KD = combined_KD
        self.single_stage = single_stage

        self.conv_layer = None

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_ab = self.ab_weight * ab_loss(
            teacher_feature=teacher_feature['features'][-1] if self.single_stage else teacher_feature['features'][1:],
            student_feature=student_feature['features'][-1] if self.single_stage else student_feature['features'][1:],
            target=target,
            single_stage=self.single_stage
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
