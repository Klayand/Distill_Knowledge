import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller
from.KD import KD


def get_feature_shapes(teacher, student, input_size):
    """
    Get the shape of feature for each layer of teacher model or student model

    Args:
        teacher: teacher model
        student: student model
        input_size: the input data size, using it to get the shapes of the features
    """
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        # fix the parameters of student model and teacher model
        _, teacher_features = teacher(data)
        _, student_features = student(data)

    teacher_feature_shapes = [feature.shape for feature in teacher_features['features']]
    student_feature_shapes = [feature.shape for feature in student_features['features']]

    return teacher_feature_shapes, student_feature_shapes


class ConvReg(nn.Module):
    """ Convolution Regression module

        Todo: to make the guided layer shape equals the hint layer shape
    """
    def __init__(self, teacher_shape, student_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu

        s_N, s_C, s_H, s_W = student_shape
        t_N, t_C, t_H, t_W = teacher_shape

        # make sure the shape of teacher equals the shape of student
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_H - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))

        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class FitNet(Distiller):
    """
    FitNets: Hints for Thin Deep Nets
        For this method you need to get the input data size, also
        you need to make sure the hint layer is wider than guided layer,
        but you can use the adaptive pooling layer to make sure the shape the same
    """

    def __init__(self, student, teacher, ce_weight=1.0,
                 feature_weight=100.0, hint_layer=2, with_l2_norm=False,
                 input_size=(32, 32), temperature=None,):
        super(FitNet, self).__init__(student=student, teacher=teacher)

        self.ce_weight = ce_weight
        self.feature_weight = feature_weight

        # hint_layer = (0, 1, 2, 3, 4)
        self.hint_layer = hint_layer
        self.temperature = temperature
        self.input_size = input_size

        self.teacher_feature_shapes, self.student_feature_shapes = get_feature_shapes(self.teacher,
                                                                                      self.student,
                                                                                      self.input_size)
        self.conv_reg = ConvReg(
            teacher_shape=self.teacher_feature_shapes[self.hint_layer],
            student_shape=self.student_feature_shapes[self.hint_layer]
        )

        self.with_l2_norm = with_l2_norm

    def forward_train(self, image, target,
                      combined_KD=False, **kwargs):
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
            regression_student_feature = regression_student_feature.reshape(s_N, s_C, -1)
            regression_student_feature_normalized = F.normalize(regression_student_feature, dim=2)
            regression_student_feature = regression_student_feature_normalized.reshape(s_N, s_C, s_H, s_W)

            t_N, t_C, t_H, t_W = teacher_feature_hint.shape
            teacher_feature_hint = teacher_feature_hint.reshape(t_N, t_C, -1)
            teacher_feature_hint_normalized = F.normalize(teacher_feature_hint, dim=2)
            teacher_feature_hint = teacher_feature_hint_normalized.reshape(t_N, t_C, t_H, t_W)


        loss_mse = self.feature_weight * F.mse_loss(
            teacher_feature_hint, regression_student_feature
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_mse': loss_mse,
        }

        total_loss = loss_ce + loss_mse

        if combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss















