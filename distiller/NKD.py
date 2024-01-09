import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


class NKDLoss(Distiller):
    """ PyTorch version of NKD """

    def __init__(self,
                 teacher,
                 student,
                 ce_weight=1.0,
                 temp=1.0,
                 gamma=1.5,
                 ):
        super(NKDLoss, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.temp = temp
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward_train(self, image, target, **kwargs):

        gt_label = target

        logit_s, _ = self.student(image)

        with torch.no_grad():
            logit_t, _ = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logit_s, target)

        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        N, c = logit_s.shape
        s_i = self.log_softmax(logit_s)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        loss_t = - (t_t * s_t).mean()

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)

        # N*class
        S_i = self.log_softmax(logit_s / self.temp)
        T_i = F.softmax(logit_t / self.temp, dim=1)

        loss_non = (T_i * S_i).sum(dim=1).mean()
        loss_non = - self.gamma * (self.temp ** 2) * loss_non

        loss_nkd = loss_t + loss_non

        loss = loss_nkd + loss_ce

        loss_dict = {
                        "loss_ce": loss_ce,
                        "loss_nkd": loss_nkd
                     }

        return logits_s, loss_dict, loss
