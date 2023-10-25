"""
    Combined SP and CKA
    proposer: Zikai Zhou
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller
from .SP import compute_sp_loss, sp_loss
from .CKA import cka_loss, CenterKernelAlignment


class CKASP(Distiller):

    def __init__(
        self,
        teacher,
        student,
        ce_weight=1.0,
        sp_weight=3000.0,
        rkd_weight=15,
        combined_KD=False,
        temperature=None,
        with_l2_norm=True,
        single_stage=False,
        soften=False,
    ):
        super(CKASP, self).__init__(teacher=teacher, student=student)
        self.sp_weight = sp_weight
        self.ce_weight = ce_weight
        self.rkd_weight = rkd_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD
        self.single_stage = single_stage
        self.soften = soften

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_sp = self.sp_weight * sp_loss(
            teacher_feature=teacher_feature["features"][-1] if self.single_stage else teacher_feature["features"][1:],
            student_feature=student_feature["features"][-1] if self.single_stage else student_feature["features"][1:],
            single_stage=self.single_stage,
        )

        if self.soften:
            teacher_logits = (logits_teacher / self.temperature).softmax(dim=1)
            studnet_logits = (logits_student / self.temperature).softmax(dim=1)

            loss_cka = self.rkd_weight * cka_loss(
                teacher_logits=teacher_logits, student_logits=studnet_logits, with_l2_norm=self.with_l2_norm
            )
        else:
            loss_cka = self.rkd_weight * cka_loss(
                teacher_logits=logits_teacher, student_logits=logits_student, with_l2_norm=self.with_l2_norm
            )

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_sp": loss_sp,
            "loss_cka": loss_cka,
        }

        total_loss = loss_ce + loss_cka + loss_sp

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict["loss_kd"] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss


def compute_spcka_loss(teacher_feature, student_feature):
    batch_size = teacher_feature.shape[0]
    b_chw_teacher = teacher_feature.reshape(batch_size, -1)
    b_chw_student = student_feature.reshape(batch_size, -1)

    G_teacher = torch.mm(b_chw_teacher, torch.t(b_chw_teacher))
    G_student = torch.mm(b_chw_student, torch.t(b_chw_student))

    G_teacher_normalized = torch.nn.functional.normalize(G_teacher)
    G_student_noramlized = torch.nn.functional.normalize(G_student)

    loss = cka_loss(G_teacher_normalized, G_student_noramlized, with_l2_norm=False)

    return loss


def spcka_loss(teacher_feature, student_feature, single_stage=False):
    if single_stage:
        return compute_spcka_loss(teacher_feature, student_feature)
    else:
        return sum([compute_spcka_loss(f_t, f_s) for f_t, f_s in zip(teacher_feature, student_feature)])


class CKASP_bmvc(Distiller):
    """
        Distilling Representational Similarity using Centered Kernel Alignment ---- 2022 BMVC
        combine CKA and SP, using it only in feature dimension.
    """

    def __init__(
        self,
        teacher,
        student,
        ce_weight=1.0,
        cka_weight=1.0,
        combined_KD=False,
        temperature=None,
        with_l2_norm=True,
        single_stage=False,

    ):
        super(CKASP_bmvc, self).__init__(teacher=teacher, student=student)
        self.cka_weight = cka_weight
        self.ce_weight = ce_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD
        self.single_stage = single_stage

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_cka = self.cka_weight * spcka_loss(
            teacher_feature=teacher_feature["features"][-1] if self.single_stage else teacher_feature["features"][1:],
            student_feature=student_feature["features"][-1] if self.single_stage else student_feature["features"][1:],
            single_stage=self.single_stage,
        )

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_cka": loss_cka,
        }

        total_loss = loss_ce + loss_cka

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict["loss_kd"] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss


class DKCKA(Distiller):
    """
        Combine CKA in logits and CKA-SP in feature.
        Proposer: Zikai Zhou
    """

    def __init__(
        self,
        teacher,
        student,
        ce_weight=1.0,
        spcka_weight=1.0,
        cka_weight=15.0,
        combined_KD=False,
        temperature=None,
        with_l2_norm=True,
        single_stage=False,
        soften=False,
    ):
        super(DKCKA, self).__init__(teacher=teacher, student=student)
        self.cka_weight = cka_weight
        self.spcka_weight = spcka_weight
        self.ce_weight = ce_weight
        self.temperature = temperature
        self.with_l2_norm = with_l2_norm
        self.combined_KD = combined_KD
        self.single_stage = single_stage
        self.soften = soften

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_spcka = self.spcka_weight * spcka_loss(
            teacher_feature=teacher_feature["features"][-1] if self.single_stage else teacher_feature["features"][1:],
            student_feature=student_feature["features"][-1] if self.single_stage else student_feature["features"][1:],
            single_stage=self.single_stage,
        )

        if self.soften:
            teacher_logits = (logits_teacher / self.temperature).softmax(dim=1)
            studnet_logits = (logits_student / self.temperature).softmax(dim=1)

            loss_cka = self.cka_weight * cka_loss(
                teacher_logits=teacher_logits, student_logits=studnet_logits, with_l2_norm=self.with_l2_norm
            )
        else:
            loss_cka = self.cka_weight * cka_loss(
                teacher_logits=logits_teacher, student_logits=logits_student, with_l2_norm=self.with_l2_norm
            )

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_spcka": loss_spcka,
            "loss_cka": loss_cka
        }

        total_loss = loss_ce + loss_cka + loss_spcka

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict["loss_kd"] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss

