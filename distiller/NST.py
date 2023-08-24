import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller


def nst_loss(teacher_feature, student_feature, single_stage=False, kernel_function: str = 'poly'):
    # N, C, H, W ---> N, C, H * W
    if single_stage:
        return single_stage_loss(teacher_feature, student_feature, kernel_function)
    else:
        return sum([single_stage_loss(f_t, f_s, kernel_function) for f_t, f_s in zip(teacher_feature, student_feature)])


def single_stage_loss(f_t, f_s, kernel_function: str = 'poly'):
    # keep the spatial dimensions of F_t and F_s the same.
    # the number of channels may be the same.

    s_H, s_W, t_H, t_W = f_s.shape[2], f_s.shape[3], f_t.shape[2], f_t.shape[3]

    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    else:
        if s_W > t_W:
            min_size = min(t_W, t_H)
            f_s = F.adaptive_avg_pool2d(f_s, (min_size, min_size))
        else:
            min_size = min(s_W, s_H)
            f_t = F.adaptive_avg_pool2d(f_t, (min_size, min_size))

    # N, C, H, W ---> N, C, H * W
    f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
    f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)

    normalized_f_s = F.normalize(f_s, dim=2)
    normalized_f_t = F.normalize(f_t, dim=2)

    if kernel_function == 'poly':
        # d=2, c=0
        # detach() create a copy of original tensor, but have no gradient, cannot be optimized
        return (
                poly_kernel(normalized_f_t, normalized_f_t).mean().detach()
            + poly_kernel(normalized_f_s, normalized_f_s).mean()
            - 2 * poly_kernel(normalized_f_t, normalized_f_s).mean()
        )

    elif kernel_function == 'linear':
        # behave like AT
        return linear_kernel(normalized_f_t, normalized_f_s)

    elif kernel_function == 'gaussian':
        return guassian_kernel(normalized_f_t, normalized_f_s)


def poly_kernel(a, b):
    # a: [N, C, HW] b: [N, C^, HW]
    # third-order matrix multiplication: C[i][j][k] = Σ(A[i][j][l] * B[l][j][k])
    a = a.unsqueeze(1)
    b = b.unsqueeze(2)
    # a: [N, 1, C, HW] b: [N, C^, 1, HW]
    # should be (a * b + c)^d
    result = (a * b).sum(-1).pow(2)

    return result


def linear_kernel(a, b):
    # should be (a * b)
    result = (sum(a).mean() - sum(b).mean()).pow(2)
    return result


def guassian_kernel(a, b):
    import numpy as np
    distance = torch.linalg.norm(a - b) # calculate Euclidean Distance
    square_distance = distance ** 2
    result = torch.exp(-square_distance / (2 * square_distance.mean()))

    return result


class NST(Distiller):
    """ Like What You Like: Knowledge Distill via Neuron Selectivity Transfer"""

    def __init__(self, teacher, student,
                 ce_weight=1.0, feature_weight=50.0, temperature=None,
                 single_stage=False, kernel_function: str='poly'):
        super(NST, self).__init__(student=student, teacher=teacher)
        self.ce_weight = ce_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.single_stage = single_stage
        self.kernel_function = kernel_function

    def forward_train(self, image, target,
                      combined_KD=False, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            if combined_KD:
                if self.temperature:
                    logits_teacher, teacher_feature = self.teacher(image)
                else:
                    print("Please input temperature")
            else:
                _, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_nst = self.feature_weight * nst_loss(
            teacher_feature=teacher_feature['features'][1:],
            student_feature=student_feature['features'][1:],
            single_stage=self.single_stage,
            kernel_function=self.kernel_function
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_nst': loss_nst
        }

        total_loss = loss_ce + loss_nst

        if combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss