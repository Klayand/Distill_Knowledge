import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .__base_distiller import Distiller


def CenterKernelAlignment(X, Y, with_l2_norm):
    """Compute the CKA similarity betweem samples"""
    # Compute Gram matrix
    gram_X = torch.matmul(X, X.t())
    gram_Y = torch.matmul(Y, Y.t())

    # l2 norm or not
    if with_l2_norm:
        gram_X = gram_X / torch.sqrt(torch.diag(gram_X)[:, None])
        gram_Y = gram_Y / torch.sqrt(torch.diag(gram_Y)[:, None])

    # compute cka
    cka = torch.trace(torch.matmul(gram_X, gram_Y.t())) / torch.sqrt(
        torch.trace(torch.matmul(gram_X, gram_X.t())) * torch.trace(torch.matmul(gram_Y, gram_Y.t()))
    )

    return cka


def cka_loss(teacher_logits, student_logits, with_l2_norm):
    """Compute the CKA similarity between samples
    CKA computes similarity between batches
    input: (N, P) ----> output: (N, N) similarity matrix
    """
    N_t = teacher_logits.shape[0]
    N_s = student_logits.shape[0]
    assert N_s == N_t  # when use cka, you need to make sure N the same

    # get a similarity score between teacher and student
    similarity_martix = CenterKernelAlignment(teacher_logits, student_logits, with_l2_norm)

    # maximize the likelihood of it
    return -similarity_martix


class CenterKernelAlignmentRKD(Distiller):
    """Center Kernel Alignment for Relational Knowledge Distillation"""

    def __init__(
        self,
        teacher,
        student,
        ce_weight=1.0,
        intra_weight=15,
        inter_weight=15,
        combined_KD=False,
        temperature=None,
        with_l2_norm=True,
        soften=False,
    ):
        super(CenterKernelAlignmentRKD, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD
        self.soften = soften

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        if self.soften:
            teacher_logits = (logits_teacher / self.temperature).softmax(dim=1)
            studnet_logits = (logits_student / self.temperature).softmax(dim=1)

            loss_cka_inter = self.inter_weight * cka_loss(
                teacher_logits=teacher_logits, student_logits=studnet_logits, with_l2_norm=self.with_l2_norm
            )

        else:
            loss_cka_inter = self.inter_weight * cka_loss(
                teacher_logits=logits_teacher, student_logits=logits_student, with_l2_norm=self.with_l2_norm
            )

        loss_cka_intra = self.intra_weight * cka_loss(
            teacher_logits=logits_teacher.transpose(0, 1),
            student_logits=logits_student.transpose(0, 1),
            with_l2_norm=self.with_l2_norm
        )

        loss_cka = loss_cka_intra + loss_cka_inter

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_cka_inter": loss_cka_inter,
            "loss_cka_intra": loss_cka_intra,
            "loss_cka": loss_cka,
        }

        total_loss = loss_ce + loss_cka

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict["loss_kd"] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss
