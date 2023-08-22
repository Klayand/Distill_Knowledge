"""
    From https://github.com/megvii-research/mdistiller/distillers/_base.py

    Changes created by Zikai Zhou
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

        self.device = device

    def init(self):
        self.student.to(self.device)
        self.teacher.to(self.device)

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-implement this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        _, pred = torch.max(self.student(image.to(self.device)), dim=1)
        return pred



