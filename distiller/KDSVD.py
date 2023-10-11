import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller


def svd_loss(teacher_features, student_features, K):
    v_sb = None
    v_tb = None
    losses = []

    for i, student_feature, teacher_feature in zip(range(student_features), student_features, teacher_features):
        u_t, s_t, v_t = svd(teacher_feature, K)
        u_s, s_s, v_s = svd(student_feature, K + 3)
        v_s, v_t = align_rsv(v_s, v_t)
        s_t = s_t.unsqueeze(1)
        v_t = v_t * s_t
        v_s = v_s * s_t

        if i > 0:
            s_rbf = torch.exp(-(v_s.unsqueeze(2) - v_sb.unsqueeze(1)).pow(2) / 8)
            t_rbf = torch.exp(-(v_t.unsqueeze(2) - v_tb.unsqueeze(1)).pow(2) / 8)

            l2loss = (s_rbf - t_rbf.detach()).pow(2)
            l2loss = torch.where(
                torch.isfinite(l2loss), l2loss, torch.zeros_like(l2loss)
            )
            losses.append(l2loss.sum())

        v_tb = v_t
        v_sb = v_s

    batch_size = student_features[0].shape[0]
    losses = [loss / batch_size for loss in losses]
    return sum(losses)


def svd(feature, n=1):
    size = feature.shape
    assert len(size) == 4

    x = feature.view(size[0], size[1] * size[2], size[3]).float()
    u, s, v = torch.svd(x)

    u = remove_nan(u)
    s = remove_nan(s)
    v = remove_nan(v)

    if n > 0:
        u = F.normalize(u[:, :, :n], dim=1)
        s = F.normalize(s[:, :n], dim=1)
        v = F.normalize(v[:, :, :n], dim=1)

    return u, s, v


def remove_nan(x):
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x


def align_rsv(a, b):
    cosine = torch.matmul(a.transpose(-2, -1), b)
    max_abs_cosine, _ = torch.max(torch.abs(cosine), 1, keepdim=True)
    mask = torch.where(
        torch.eq(max_abs_cosine, torch.abs(cosine)),
        torch.sign(cosine),
        torch.zeros_like(cosine),
    )
    a = torch.matmul(a, mask)
    return a, b


class KDSVD(Distiller):
    """
    Self-supervised Knowledge Distillation using Singular Value Decomposition
    original Tensorflow code: https://github.com/sseung0703/SSKD_SVD
    """

    def __init__(
            self,
            teacher,
            student,
            combined_KD=False,
            ce_weight=1.0,
            feature_weight=1.0,
            temperature=4,
            K=1
    ):
        super(KDSVD, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.combined_KD = combined_KD
        self.temperature = temperature
        self.K = K

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_svd = self.feature_weight * svd_loss(
            teacher_features=teacher_feature['features'][1:],
            student_features=student_feature['features'][1:],
            K=self.K
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_svd': loss_svd
        }

        total_loss = loss_ce + loss_svd

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict["loss_kd"] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss
