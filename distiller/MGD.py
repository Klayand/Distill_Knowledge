"""
    Masked Generative Distillation
        Using student feature map to generate masked student feature map, then use it to generate new feature map to mimick teacher feature map
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller
from .utils import get_feature_shapes


def compute_mgd_loss(teacher_channels, student_channels, teacher_features, student_features, single_stage):
    num_params = 0
    total_loss = 0

    learnable_parameters = []

    if not single_stage:
        assert len(teacher_channels) == len(student_channels)

        for i in range(len(teacher_channels)):
            mgd_loss = MGDLoss(teacher_channels[i], student_channels[i])
            loss = mgd_loss.forward(teacher_features[i], student_features[i])

            num_params += mgd_loss.get_num_of_parameters()
            total_loss += loss
            learnable_parameters.extend(mgd_loss.get_learnable_params())

    else:
        mgd_loss = MGDLoss(teacher_channels, student_channels)
        loss = mgd_loss.forward(teacher_features, student_features)

        num_params += mgd_loss.get_num_of_parameters()
        total_loss += loss
        learnable_parameters.extend(mgd_loss.get_learnable_params())

    return total_loss, num_params, learnable_parameters


class MGDLoss(nn.Module):
    def __init__(self, teacher_channel, student_channel, alpha=0.00007, beta=0.15):
        super(MGDLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        if student_channel != teacher_channel:
            self.align = nn.Conv2d(student_channel, teacher_channel, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channel, teacher_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channel, teacher_channel, kernel_size=3, padding=1),
        )

        self.to("cuda")

    def forward(self, teacher_feature, student_feature):
        """
        :param teacher_feature: B, C, H, W ----> teacher feature map
        :param student_feature: B, C, H, W ----> student feature map
        """

        if self.align:
            student_feature = self.align(student_feature)

        # keep the same spatial size
        if teacher_feature.shape[-2:] != student_feature.shape[-2:]:
            print("Spatial size cannot match! Force the spatial size to be the same!")
            s_H, s_W, t_H, t_W = (
                student_feature.shape[2],
                student_feature.shape[3],
                teacher_feature.shape[2],
                teacher_feature.shape[3],
            )

            if s_H > t_H:
                f_s = F.adaptive_avg_pool2d(student_feature, (t_H, t_H))
            elif s_H < t_H:
                f_t = F.adaptive_avg_pool2d(teacher_feature, (s_H, s_H))
            else:
                if s_W > t_W:
                    min_size = min(t_W, t_H)
                    f_s = F.adaptive_avg_pool2d(student_feature, (min_size, min_size))
                else:
                    min_size = min(s_W, s_H)
                    f_t = F.adaptive_avg_pool2d(teacher_feature, (min_size, min_size))

        loss = self.alpha * self.get_mgd_loss(student_feature, teacher_feature)

        return loss

    def get_mgd_loss(self, student_feature, teacher_feature):
        loss_mse = nn.MSELoss(reduction="sum")

        N, C, H, W = teacher_feature.shape

        mat = torch.randn((N, C, 1, 1)).to("cuda")
        mat = torch.where(mat < self.beta, 0, 1).to("cuda")  # generate random mask

        masked_feature = torch.mul(student_feature, mat)

        print(self.generation.modules())
        print(masked_feature.shape)
        generated_feature = self.generation(masked_feature)
        assert False
        mgd_loss = loss_mse(generated_feature, teacher_feature) / N

        return mgd_loss

    def get_num_of_parameters(self):
        num_param = 0

        if self.align:
            for param in self.align.parameters():
                num_param += param.numel()

        for param in self.generation.parameters():
            num_param += param.numel()

        return num_param

    def get_learnable_params(self):
        return (
            list(self.generation.parameters()) + list(self.align.parameters())
            if self.align
            else list(self.generation.parameters())
        )


class MGD(Distiller):
    def __init__(
        self,
        teacher,
        student,
        ce_weight=1.0,
        alpha=0.00007,
        beta=0.15,
        combined_KD=False,
        temperature=None,
        single_stage=False,
    ):
        super(MGD, self).__init__(teacher=teacher, student=student)
        """
            Args:
                alpha: weight of mgd loss.
                beta: masked ratio.
        """
        self.ce_weight = ce_weight
        self.alpha = alpha
        self.beta = beta
        self.combined_KD = combined_KD
        self.temperature = temperature
        self.single_stage = single_stage

        # for other dataset, it needs to be rewritten
        self.teacher_shapes, self.student_shapes = get_feature_shapes(teacher, student, input_size=(32, 32))

        if self.single_stage:
            self.teacher_channels = self.teacher_shapes[-1][1]
            self.student_channels = self.student_shapes[-1][1]
        else:
            self.teacher_channels = []
            self.student_channels = []

            for shape in self.teacher_shapes:
                self.teacher_channels.append(shape[1])

            for shape in self.student_shapes:
                self.student_channels.append(shape[1])

        self.num_params = 0
        self.learnable_params = None

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_mgd, self.num_params, self.learnable_params = compute_mgd_loss(
            teacher_channels=self.teacher_channels,
            student_channels=self.student_channels,
            teacher_features=teacher_feature["features"][-1] if self.single_stage else teacher_feature["features"],
            student_features=student_feature["features"][-1] if self.single_stage else student_feature["features"],
            single_stage=self.single_stage,
        )

        loss_dict = {"loss_ce": loss_ce, "loss_mgd": loss_mgd}

        total_loss = loss_ce + loss_mgd

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict["loss_kd"] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss

    def get_extra_parameters(self):
        return self.num_params

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.learnable_params
