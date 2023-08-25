"""
    ref from megvii-research
    verified by Zikai Zhou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_feature_shapes(teacher, student, input_size, pooling=False):
    """
    Get the shape of feature for each layer of teacher model or student model

    Args:
        teacher: teacher model
        student: student model
        input_size: the input data size, using it to get the shapes of the features
    """
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        # fix the parameters of student model and teacher model
        _, teacher_features = teacher(data)
        _, student_features = student(data)

    teacher_feature_shapes = [feature.shape for feature in teacher_features['features']]
    student_feature_shapes = [feature.shape for feature in student_features['features']]

    if pooling:
        teacher_feature_shapes.append(teacher_features['avgpool_feature'].shape)
        student_feature_shapes.append(student_features['avgpool_feature'].shape)

    return teacher_feature_shapes, student_feature_shapes


class ConvReg(nn.Module):
    """ Convolution Regression module

        Todo: to make the guided layer shape equals the hint layer shape
    """
    def __init__(self, teacher_shape, student_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu

        s_N, s_C, s_H, s_W = student_shape
        t_N, t_C, t_H, t_W = teacher_shape

        # make sure the shape of teacher equals the shape of student
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_H - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))

        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all