"""
    Probabilistic Knowledge Transfer for deep representation learning
    Using Conditional KDE like cosine KDE to get the prob distribution of features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller


def pkt_loss(teacher_feature, student_feature, eps=1e-7):
    # Normalize each vector by its norm, get unit vector
    # student feature [N, -1]
    output_net_norm = torch.sqrt(torch.sum(student_feature ** 2, dim=1, keepdim=True))  # [N, 1]
    student_feature = student_feature / (output_net_norm + eps)  # [N, -1]
    student_feature[student_feature != student_feature] = 0  # replace NaN

    target_net_norm = torch.sqrt(torch.sum(teacher_feature ** 2, dim=1, keepdim=True))
    teacher_feature = teacher_feature / (target_net_norm + eps)
    teacher_feature[teacher_feature != teacher_feature] = 0  # replace NaN

    # Calculate the cosine similarity
    model_similarity = torch.mm(student_feature, student_feature.transpose(0, 1))  # [N, N]
    target_similarity = torch.mm(teacher_feature, teacher_feature.transpose(0, 1))  # [N, N]

    # Scale cosine similarity to 0-1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)  # [N, N]
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss


class PKD(Distiller):
    """
        We down/up sample the features of the student model to the spatial size of those
        of the teacher model if their spatial sizes are different by default.
    """

    def __init__(
            self,
            teacher,
            student,
            ce_weight=1.0,
            feature_weight=3000.0,
            combined_KD=False,
            temperature=4,
    ):
        super(PKD, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.feature_weight = feature_weight
        self.combined_KD = combined_KD
        self.temperature = temperature

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_pkt = self.feature_weight * pkt_loss(
            teacher_feature=teacher_feature['avgpool_feature'],
            student_feature=student_feature['avgpool_feature']
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_pkt': loss_pkt
        }

        total_loss = loss_ce + loss_pkt

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict["loss_kd"] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss
