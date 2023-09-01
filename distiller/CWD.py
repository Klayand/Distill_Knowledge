import torch
import torch.nn as nn
import torch.nn.functional as F

from .__base_distiller import Distiller


CONV_LAYER = []


def cwd_loss(teacher_features, student_features, temperature, single_stage=False):
    if single_stage:
        return single_stage_loss(teacher_features, student_features, temperature)
    else:
        return sum([single_stage_loss(f_t, f_s, temperature) for f_t, f_s in zip(teacher_features, student_features)])


def single_stage_loss(teacher_feature, student_feature, temperature):
    N_s, C_s, H_s, W_s = student_feature.shape
    N_t, C_t, H_t, W_t = teacher_feature.shape

    teacher_feature_norm = F.softmax(teacher_feature.reshape(N_t, C_t, -1) / temperature, dim=2)
    student_feature_norm = F.log_softmax(student_feature.reshape(N_s, C_s, -1) / temperature, dim=2)

    if C_t != C_s:
        conv_layer = nn.Conv2d(C_t, C_s, kernel_size=(1, 1))
        teacher_feature_norm = conv_layer(teacher_feature_norm)
        CONV_LAYER.append(conv_layer)

    loss = F.kl_div(student_feature_norm, teacher_feature_norm, reduction='none').sum(2).mean()

    return loss


class ChannelWiseDivergence(Distiller):
    """ Channel-wise Knowledge Distillation for Dense Prediction

        For this work, it is noted that, the original paper says the activation maps involve
        inner feature maps and final score logits.
        So if just use the score logits, it behaves like KD.
        But if you combine them together, you have two options:
        1. using inner feature map and score logits separately.
        2. fusing inner feature map and score logits together.

        Here I just choose the former. And I think the activation maps only involve the
        inner feature maps, you can set combie_KD=True to introduce the final score logits.
    """

    def __init__(self, teacher, student, combined_KD=False,
                 ce_weight=1.0, cwd_weight=3.0,
                 temperature=4, single_stage=False):
        super(ChannelWiseDivergence, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.cwd_weight = cwd_weight
        self.temperature = temperature
        self.single_stage = single_stage
        self.combined_KD = combined_KD

    def forward_train(self, image, target, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)
        # loss_kd = self.cwd_weight * kd_loss(logits_student, logits_teacher, self.temperature)

        loss_cwd = self.cwd_weight * cwd_loss(
                teacher_features=teacher_feature["features"],
                student_features=student_feature['features'],
                temperature=self.temperature,
        )

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_cwd': loss_cwd
        }

        total_loss = loss_ce + loss_cwd

        if self.combined_KD:
            from .KD import kd_loss
            loss_kd = self.cwd_weight * kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss

    def get_learnable_parameters(self):
        if len(self.conv_layers) != 0:
            return super(ChannelWiseDivergence, self).get_learnable_parameters() + [layer.parameters for layer in CONV_LAYER]
        else:
            return super(ChannelWiseDivergence, self).get_learnable_parameters()

    def get_extra_parameters(self):
        if len(CONV_LAYER) != 0:
            num_param = 0
            for i in range(len(CONV_LAYER)):
                num_param += CONV_LAYER[i].parameters.numel()
            return num_param
        else:
            return 0