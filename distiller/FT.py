import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import optim

from torch.autograd import Variable
from torch.autograd import Function
from tqdm import tqdm

from .__base_distiller import Distiller
from .utils import get_feature_shapes


def ft_loss(factor_teacher, factor_student, method: str = "l1"):
    factor_t = F.normalize(factor_teacher.view(factor_teacher.size(0), -1))
    factor_s = F.normalize(factor_student.view(factor_student.size(0), -1))

    if method == "l1":
        loss = F.l1_loss(factor_s, factor_t)
    elif method == "mse":
        loss = F.mse_loss(factor_s, factor_t)

    return loss


def train_paraphraser(model, module, data_loader, epochs=300, save=True, fp16=False):
    from torch.cuda.amp import autocast, GradScaler

    print("Paraphraser training begin...")

    optimizer = optim.SGD(module.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    criterion = nn.L1Loss()

    model.to("cuda")
    module.to("cuda")

    model.eval()

    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        train_loss = 0

        module.train()
        # train
        pbar = tqdm(data_loader)
        for step, (x, y) in enumerate(pbar, 1):
            x, y = x.to("cuda"), y.to("cuda")
            if fp16:
                with autocast():
                    with torch.no_grad():
                        out, features = model(x)

                    input_feature = features["features"][-1]
                    out_feature = module(input_feature, mode=0)

                    loss = criterion(out_feature, input_feature)

            else:
                with torch.no_grad():
                    out, features = model(x)

                input_feature = features["features"][-1]
                out_feature = module(input_feature, mode=0)

                loss = criterion(out_feature, input_feature)

            train_loss += loss.item()
            optimizer.zero_grad()

            if fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # nn.utils.clip_grad_value_(self.teacher.parameters(), 0.1)
                # nn.utils.clip_grad_norm(self.teacher.parameters(), max_norm=10)
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                # nn.utils.clip_grad_value_(self.teacher.parameters(), 0.1)
                # nn.utils.clip_grad_norm(self.teacher.parameters(), max_norm=10)
                optimizer.step()

            if step % 10 == 0:
                pbar.set_postfix_str(f"loss={train_loss / step}")

        train_loss /= len(data_loader)

        print(f"epoch {epoch}, train_loss = {train_loss}")

        if (not os.path.exists("paraphraser.pth")) and save:
            torch.save(module.state_dict(), "paraphraser.pth")

    print("Paraphraser training completed...")
    return module


class FT(Distiller):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""

    def __init__(self, teacher, student, ce_weight=1.0, k_rate=0.5, beta=250, combined_KD=False, temperature=None):
        super(FT, self).__init__(teacher=teacher, student=student)

        self.ce_weight = ce_weight
        self.k_rate = k_rate
        self.beta = beta
        self.combined_KD = combined_KD
        self.temperature = temperature

        # for other dataset, it needs to be rewritten
        self.teacher_shapes, self.student_shapes = get_feature_shapes(teacher, student, input_size=(32, 32))
        self.teacher_shape, self.student_shape = self.teacher_shapes[-1][1], self.student_shapes[-1][1]

        # you need first to train the paraphraser.
        # for the translator, it should be trained jointly with student model.
        self.paraphraser = Paraphraser(
            in_planes=self.teacher_shape, planes=int(round(self.teacher_shape * self.k_rate))
        ).to("cuda")
        self.translator = Translator(
            in_planes=self.student_shape, planes=int(round(self.teacher_shape * self.k_rate))
        ).to("cuda")

    def forward_train(self, image, target, **kwargs):
        self.translator.train()
        self.paraphraser.eval()

        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        teacher_featuremap = teacher_feature["features"][-1]
        student_featuremap = student_feature["features"][-1]

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        with torch.no_grad():
            factor_teacher = self.paraphraser(teacher_featuremap, mode=1)
        factor_student = self.translator(student_featuremap)

        loss_ft = self.beta * ft_loss(factor_teacher=factor_teacher, factor_student=factor_student, method="l1")

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_ft": loss_ft,
        }

        total_loss = loss_ce + loss_ft

        if self.combined_KD:
            from .KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict["loss_kd"] = loss_kd
            total_loss += loss_kd

        return logits_student, loss_dict, total_loss

    def get_learnable_parameters(self):
        return (
            super().get_learnable_parameters() + list(self.paraphraser.parameters()) + list(self.translator.parameters())
        )

    def get_extra_parameters(self):
        num_param = 0
        for param in self.paraphraser.parameters():
            num_param += param.numel()
        for param in self.translator.parameters():
            num_param += param.numel()
        return num_param


class Paraphraser(nn.Module):
    """From original paper, the reconstruction loss should use l1 loss, if you use l2 loss, you should adjust the learning rate
    to avoid gradient explosion.
    For using BN or not, it is noted in original paper, we do not use BN.
    Also, the architecture of Paraphraser follows the original one.
    """

    def __int__(self, in_planes, planes, stride=1):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU(0.1)
        # self.bn = nn.BatchNorm2d(in_planes)

        self.conv0 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.deconv0 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn_de0 = nn.BatchNorm2d(planes)

        self.deconv1 = nn.ConvTranspose2d(planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn_de1 = nn.BatchNorm2d(in_planes)

        self.deconv2 = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, mode):
        """
        Mode 0: train the encoder and decoder (reconstruction)
        Mode 1: extracting teacher factors (encoder)
        Mode 2: only use decoder
        """
        if mode == 0:
            # encoder
            out = self.leakyrelu(self.conv0(x))
            out = self.leakyrelu(self.conv1(out))
            out = self.leakyrelu(self.conv2(out))

            # decoder
            out = self.leakyrelu(self.deconv0(out))
            out = self.leakyrelu(self.deconv1(out))
            out = self.leakyrelu(self.deconv2(out))

        if mode == 1:
            out = self.leakyrelu(self.conv0(x))
            out = self.leakyrelu(self.conv1(out))
            out = self.leakyrelu(self.conv2(out))

        if mode == 2:
            out = self.leakyrelu(self.deconv0(x))
            out = self.leakyrelu(self.deconv1(out))
            out = self.leakyrelu(self.deconv2(out))

        return out

    def get_learnable_parameters(self):
        return list(self.parameters())

    def get_num_parameters(self):
        num_param = 0
        for param in self.parameters():
            num_param += param.numel()
        return num_param


class Translator(nn.Module):
    def __int__(self, in_planes, planes, stride=1):
        super(Translator, self).__init__()

        self.leakyrelu = nn.LeakyReLU(0.1)
        # self.bn = nn.BatchNorm2d(in_planes)

        self.conv0 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = self.leakyrelu(self.conv0(x))
        out = self.leakyrelu(self.conv1(out))
        out = self.leakyrelu(self.conv2(out))

        return out

    def get_learnable_parameters(self):
        return list(self.parameters())

    def get_num_parameters(self):
        num_param = 0
        for param in self.parameters():
            num_param += param.numel()
        return num_param
