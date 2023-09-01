import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller


def activation_max(f_t, f_s, p):
    # N, C, H, W ---> N, H, W ---> N,H*W*C
    f_t_max = torch.max(torch.abs(f_t).pow(p), dim=1).values.reshape(f_t.shape[0], -1)
    f_s_max = torch.max(torch.abs(f_s).pow(p), dim=1).values.reshape(f_s.shape[0], -1)

    return f_t_max, f_s_max


def activation_sum(f_t, f_s, p):
    # N, C, H, W ---> N, H, W
    f_t_sum = torch.sum(torch.abs(f_t).pow(p), dim=1).reshape(f_t.shape[0], -1)
    f_s_sum = torch.sum(torch.abs(f_s).pow(p), dim=1).reshape(f_s.shape[0], -1)

    return f_t_sum, f_s_sum


def at_loss(input_data, target, teacher_feature, student_feature, teacher_net, student_net, single_stage=False, at_method: str='activation_sum', p=2):
    # N, C, H, W ---> N, C, H * W
    method = at_method
    if method.split('_')[0] == 'gradient':
        return gradient_based_loss(input_data, target, teacher_net, student_net, at_method)
    else:
        if single_stage:
            return single_stage_loss(teacher_feature, student_feature, at_method, p)
        else:
            return sum([single_stage_loss(f_t, f_s, at_method, p) for f_t, f_s in zip(teacher_feature, student_feature)])


def single_stage_loss(f_t, f_s, at_method, p):
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

    if at_method == 'activation_sum':
        assert p >= 1
        f_t_q, f_s_q = activation_sum(f_t, f_s, p)
    elif at_method == 'activation_max':
        assert p >= 1
        f_t_q, f_s_q = activation_max(f_t, f_s, p)

    f_t_sum_normalized = F.normalize(f_t_q)
    f_s_sum_normalized = F.normalize(f_s_q)

    result = (f_t_sum_normalized - f_s_sum_normalized).pow(2).mean()

    return result


def gradient_based_loss(input_data, target, teacher_net, student_net, at_method):
    # N, C, H, W
    input_data.requires_grad_()

    if at_method == 'gradient_flip':
        teacher_logits, _ = teacher_net(torch.flip(input_data, dims=[3])) # horizontal flip
    elif at_method == 'gradient':
        teacher_logits, _ = teacher_net(input_data)

    student_logits, _ = student_net(input_data)

    teacher_loss = F.cross_entropy(teacher_logits, target)
    student_loss = F.cross_entropy(student_logits, target)

    teacher_gradients = torch.autograd.grad(teacher_loss, input_data,
                                            create_graph=True)[0]

    student_gradients = torch.autograd.grad(student_loss, input_data,
                                            create_graph=True)[0]

    result = (teacher_gradients - student_gradients).pow(2).sum()
    return result


class AT(Distiller):
    """  Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer """

    def __init__(self, teacher, student, combined_KD=False,
                ce_weight=1.0, feature_weight=1000.0, temperature=None, p=2,
                single_stage=False, at_method: str= 'activation_sum'):
        super(AT, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.single_stage = single_stage
        self.at_method = at_method
        self.p = p
        self.combined_KD = combined_KD

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_at = self.feature_weight * at_loss(
            input_data = image,
            target = target,
            teacher_feature=teacher_feature['features'][1:],
            student_feature=student_feature['features'][1:],
            teacher_net = self.teacher,
            student_net = self.student,
            single_stage=self.single_stage,
            at_method=self.at_method,
            p=self.p
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_at': loss_at
        }

        total_loss = loss_ce + loss_at

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss