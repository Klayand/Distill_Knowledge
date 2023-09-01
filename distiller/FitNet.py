import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller
from .utils import ConvReg, get_feature_shapes


class FitNet(Distiller):
    """
    FitNets: Hints for Thin Deep Nets
        For this method you need to get the input data size, also
        you need to make sure the hint layer is wider than guided layer,
        but you can use the adaptive pooling layer to make sure the shape the same
    """

    def __init__(self, student, teacher, combined_KD=False, ce_weight=1.0,
                 feature_weight=50.0, hint_layer=2, with_l2_norm=False,
                 input_size=(32, 32), temperature=None):
        super(FitNet, self).__init__(student=student, teacher=teacher)

        self.ce_weight = ce_weight
        self.feature_weight = feature_weight

        # hint_layer = (0, 1, 2, 3, 4)
        self.hint_layer = hint_layer
        self.temperature = temperature
        self.input_size = input_size
        self.combined_KD= combined_KD

        self.teacher_feature_shapes, self.student_feature_shapes = get_feature_shapes(self.teacher,
                                                                                      self.student,
                                                                                      self.input_size)
        self.conv_reg = ConvReg(
            teacher_shape=self.teacher_feature_shapes[self.hint_layer],
            student_shape=self.student_feature_shapes[self.hint_layer]
        )

        self.with_l2_norm = with_l2_norm

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        regression_student_feature = self.conv_reg(
            student_feature['features'][self.hint_layer]
        )

        teacher_feature_hint = teacher_feature['features'][self.hint_layer]

        if self.with_l2_norm:
            # spatial dimension normalizatiion
            # here you have two options, one is like FitNets, another is more like NST.
            # using 'regression_student_feature_normalized' and 'teacher_feature_hint_normalized' is like NST
            # using 'regression_student_feature' and 'teacher_feature_hint' is like FitNets

            s_N, s_C, s_H, s_W = regression_student_feature.shape
            regression_student_feature_reshape = regression_student_feature.reshape(s_N, s_C, -1)
            regression_student_feature_normalized = F.normalize(regression_student_feature_reshape, dim=2)
            regression_student_feature = regression_student_feature_normalized.reshape(s_N, s_C, s_H, s_W)

            t_N, t_C, t_H, t_W = teacher_feature_hint.shape
            teacher_feature_hint_reshape = teacher_feature_hint.reshape(t_N, t_C, -1)
            teacher_feature_hint_normalized = F.normalize(teacher_feature_hint_reshape, dim=2)
            teacher_feature_hint = teacher_feature_hint_normalized.reshape(t_N, t_C, t_H, t_W)


        loss_mse = self.feature_weight * F.mse_loss(
            teacher_feature_hint, regression_student_feature
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_mse': loss_mse,
        }

        total_loss = loss_ce + loss_mse

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_parameters = 0
        for param in self.conv_reg.parameters():
            num_parameters += param.numel()
        return num_parameters

