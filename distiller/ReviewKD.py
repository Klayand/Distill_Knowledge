"""
    Original author: https://github.com/dvlab-research/ReviewKD
    Verified by Zikai Zhou

    Still existing some bugs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb

from .__base_distiller import Distiller
from .utils import hcl_loss, ABF, get_feature_shapes


class ReviewKD(Distiller):
    """ Distilling Knowledge via Knowledge Review"""
    def __init__(self, teacher, student, combine_KD=False,
                 ce_weight=1.0, reviwekd_weight=1.0,
                 input_size=(32, 32), temperature=None):
        super(ReviewKD, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.reviewkd_weight = reviwekd_weight
        self.temperature = temperature
        self.combine_KD = combine_KD

        self.teacher_features_shapes, self.student_features_shapes = get_feature_shapes(
                                                                        teacher=self.teacher,
                                                                        student=self.student,
                                                                        input_size=input_size,
                                                                        pooling=True)

        self.in_channels = []
        self.out_channels = []
        self.shapes = []
        self.out_shapes = []

        for i in range(len(self.student_features_shapes)):
            if i != len(self.student_features_shapes) - 1:
                self.in_channels.append(self.student_features_shapes[i][1])
            if i != 0:
                self.shapes.append(self.student_features_shapes[i][1])

        for i in range(len(self.teacher_features_shapes)):
            if i != len(self.teacher_features_shapes) - 1:
                self.out_channels.append(self.teacher_features_shapes[i][1])
            if i != 0:
                self.out_shapes.append(self.teacher_features_shapes[i][1])



        abfs = nn.ModuleList()

        mid_channel = min(512, self.in_channels[-1])
        for idx, in_channel in enumerate(self.in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    self.out_channels[idx],
                    idx < len(self.in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_param = 0
        for param in self.abfs.parameters():
            num_param += param.numel()
        return num_param

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)


        x = student_feature['features'] + [student_feature['avgpool_feature'].unsqueeze(-1).unsqueeze(-1)]

        # for the last layer, it doesn`t need to get across ABF module
        x = x[::-1]

        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)

        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)

        features_teacher = teacher_feature["features"][1:] + [teacher_feature["avgpool_feature"].unsqueeze(-1).unsqueeze(-1)]

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_reviewkd = (
            self.reviewkd_weight
            * hcl_loss(results, features_teacher)
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_reviewkd': loss_reviewkd
        }

        total_loss = loss_ce + loss_reviewkd

        if self.combine_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss
