import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller


def sp_loss(teacher_feature, student_feature):
    batch_size = teacher_feature.shape[0]
    b_chw_teacher = teacher_feature.reshape(batch_size, -1)
    b_chw_student = student_feature.reshape(batch_size, -1)

    G_teacher = torch.mm(b_chw_teacher, torch.t(b_chw_student))
    G_student = torch.mm(b_chw_student, torch.t(b_chw_student))

    G_teacher_normalized = torch.nn.functional.normalize(G_teacher)
    G_student_noramlized = torch.nn.functional.normalize(G_student)

    square_loss = (G_teacher_normalized - G_student_noramlized)**2
    sp_loss = square_loss.view(-1, 1).sum(0) / (batch_size * batch_size)

    return sp_loss


class SP(Distiller):
    """ Similarity-Preserving Knowledge Distillation, ICCV2019"""
    def __init__(self, teacher, student, ce_weight, feature_weight, temperature=None):
        super(SP, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.feature_weight = feature_weight
        self.temperature = temperature

    def forward_train(self, image, target,
                      combined_kd=False, **kwargs):

        logits_student, student_features = self.student(image)

        with torch.no_grad():
            if combined_kd:
                if self.temperature:
                    logits_teacher, teacher_features = self.teacher(image, target)
                else:
                    print("Please input temperature")
            else:
                _, teacher_features = self.teacher(image, target)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)
        loss_sp = self.feature_weight * sp_loss(
            teacher_features['features'][-1], student_features['features'][-1]
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_sp': loss_sp
        }

        total_loss = loss_ce + loss_sp

        if combined_kd:
            from KD import kd_loss
            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd

            total_loss += loss_kd

        return logits_student, loss_dict, total_loss