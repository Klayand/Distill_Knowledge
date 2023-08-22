import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base_distiller import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1).mean()

    # follow the paper, the gradient is scaled 1/T^2, multiply T^2 to recover its gradients
    loss_kd = loss_kd * temperature**2

    return loss_kd


class KD(Distiller):
    """ Distilling the Knowledge in a Neural Network"""

    def __init__(self, teacher, student, temperature, ce_weight, kd_weight):
        super(KD, self).__init__(student=student, teacher=teacher)
        self.temperature = temperature
        self.ce_loss_weight = ce_weight
        self.kd_loss_weight = kd_weight

    # calculate a mini-batch data
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )

        losses_dict = {
            'loss_ce': loss_ce,
            'loss_kd': loss_kd
        }

        loss = loss_kd + loss_ce

        return loss, losses_dict


















