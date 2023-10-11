"""
    Proposer: Zikai Zhou
        Combine DIST ,SP and CKA.
"""

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


def compute_matrix(teacher_feature, student_feature):
    batch_size = teacher_feature.shape[0]

    b_chw_teacher = teacher_feature.reshape(batch_size, -1)
    b_chw_student = student_feature.reshape(batch_size, -1)

    G_teacher = torch.mm(b_chw_teacher, torch.t(b_chw_teacher))
    G_student = torch.mm(b_chw_student, torch.t(b_chw_student))

    G_teacher_normalized = torch.nn.functional.normalize(G_teacher)
    G_student_noramlized = torch.nn.functional.normalize(G_student)

    return G_teacher_normalized, G_student_noramlized


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(teacher_logits, student_logits):
    return 1 - pearson_correlation(student_logits, teacher_logits).mean()


def intra_class_relation(teacher_logits, student_logits):
    return inter_class_relation(teacher_logits.transpose(0, 1), student_logits.transpose(0, 1))


class SPDISTCKA(Distiller):

    def __init__(
            self,
            teacher,
            student,
            ce_weight=1.0,
            beta=1.0,
            gamma=1.0,
            temperature=4.0,
            cka_weight=25,
            dist_weight=5,
            is_cka=False,
            with_l2_norm=True
    ):
        super(SPDISTCKA, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.cka_weight = cka_weight
        self.is_cka = is_cka
        self.with_l2_norm = with_l2_norm
        self.dist_weight = dist_weight

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        teacher_feature, student_feature = compute_matrix(
            teacher_feature=teacher_feature["features"][-1],
            student_feature=student_feature["features"][-1],
        )

        loss_inter = self.beta * (self.temperature**2) * inter_class_relation(teacher_feature, student_feature)
        loss_intra = self.gamma * (self.temperature**2) * intra_class_relation(teacher_feature, student_feature)

        loss_dist = self.dist_weight * (loss_intra + loss_inter)

        loss_dict = {"loss_ce": loss_ce, "loss_inter": loss_inter, "loss_intra": loss_intra, "loss_dist": loss_dist}

        total_loss = loss_dist + loss_ce

        if self.is_cka:
            loss_cka = self.cka_weight * cka_loss(
                teacher_logits=logits_teacher, student_logits=logits_student, with_l2_norm=self.with_l2_norm
            )
            loss_dict['cka'] = loss_cka
            total_loss += loss_cka

        return logits_student, loss_dict, total_loss
