"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>

adding hyperparameter norm_layer: Huanran Chen
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "mobilenetv2_T_w",
    "mobilenetV2",
]

BN = None


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            norm_layer(inp * expand_ratio),
            nn.ReLU(),
            # dw
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False,
            ),
            norm_layer(inp * expand_ratio),
            nn.ReLU(),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ["0", "1", "2", "3", "4", "5", "6", "7"]

    def forward(self, x):
        t = x

        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """mobilenetV2"""

    def __init__(self, T, feature_dim, input_size=32, width_mult=1.0, remove_avg=False,
                 norm_layer=nn.BatchNorm2d):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 2)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(InvertedResidual(input_channel, output_channel, stride, t,
                                               norm_layer=norm_layer))
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(self.last_channel, feature_dim)
        H = input_size // (32 // 2)
        self.avgpool = nn.AvgPool2d(H, ceil_mode=True)

        self._initialize_weights()
        print(T, width_mult)
        self.stage_channels = [32, 24, 32, 96, 320]
        self.stage_channels = [int(c * width_mult) for c in self.stage_channels]


    def get_bn_before_relu(self):
        bn1 = self.blocks[1][-1].conv[-1]
        bn2 = self.blocks[2][-1].conv[-1]
        bn3 = self.blocks[4][-1].conv[-1]
        bn4 = self.blocks[6][-1].conv[-1]
        return [bn1, bn2, bn3, bn4]

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def forward(self, x, is_feat=False, preact=False, is_srrl=None):

        if not is_srrl is None:
            if not self.remove_avg:
                out = self.avgpool(is_srrl)
            out = out.view(out.size(0), -1)

            out = self.classifier(out)

            return out

        else:

            out = self.conv1(x)
            f0 = out
            out = self.blocks[0](out)
            out = self.blocks[1](out)
            f1 = out

            out = self.blocks[2](out)
            f2 = out

            out = self.blocks[3](out)
            out = self.blocks[4](out)
            f3 = out

            out = self.blocks[5](out)
            out = self.blocks[6](out)
            f4 = out

            out = self.conv2(out)

            if not self.remove_avg:
                out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            avg = out

            out = self.classifier(out)

            features = {}
            features['features'] = [f0, f1, f2, f3, f4]
            features['avgpool_feature'] = avg

            return out, features

    def get_stage_channels(self):
        return self.stage_channels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2_T_w(T, W, feature_dim=100, **kwargs):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W, **kwargs)
    return model


def mobilenetV2(num_classes, **kwargs):
    return mobilenetv2_T_w(6, 0.5, num_classes, **kwargs)

