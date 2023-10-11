import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from .__base_distiller import Distiller
from .utils import get_feature_shapes


def normalize(x):
    return F.normalize(x, dim=1)


class CRD(Distiller):
    """ Contrastive Representation Distillation """

    def __init__(self,
                 teacher,
                 student,
                 ce_weight=1.0,
                 feature_weight=1.0,
                 temperature=0.07,
                 neg_num=16384,
                 sample_n=50000,
                 dim_out=128,
                 momentum=0.5,
                 eps=1e-7,
                 combined_KD=False
                 ):
        super(CRD, self).__init__(teacher=teacher, student=student)
        self.ce_weight = ce_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.neg_num = neg_num
        self.sample_n = sample_n
        self.dim_out = dim_out
        self.momentum = momentum
        self.eps = eps
        self.combined_KD = combined_KD

        # for cifar, input size ---> 32x32
        teacher_feature_shapes, student_feature_shapes = get_feature_shapes(self.teacher,
                                                                            self.student,
                                                                            input_size=(32, 32),
                                                                            pooling=True)

        self.teacher_dim, self.student_dim = teacher_feature_shapes[-1][1], student_feature_shapes[-1][1]

        self.init_crd_modules(
            student_channel=self.student_dim,
            teacher_channel=self.teacher_dim,
            out_dim=self.dim_out,
            sample_n=self.sample_n,
            neg_num=self.neg_num,
            momentum=self.momentum,
            temperature=self.temperature
        )

    def init_crd_modules(
            self,
            student_channel,
            teacher_channel,
            out_dim,
            sample_n,
            neg_num,
            momentum,
            temperature
    ):
        self.embed_student = Embed(student_channel, out_dim)
        self.embed_teacher = Embed(teacher_channel, out_dim)
        self.contrast = ContrastMemory(out_dim, sample_n, neg_num, temperature, momentum)
        self.student_criterion = ContrastLoss(sample_n)
        self.teacher_criterion = ContrastLoss(sample_n)

    def forward_train(self, image, target, index, contrastive_index, **kwargs):
        logits_student, student_feature = self.student(image)

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(image)

        # Compute loss
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)

        loss_crd = self.feature_weight * self.crd_loss(
            student_feature=student_feature['avgpool_feature'],
            teacher_feature=teacher_feature['avgpool_feature'],
            idx=index,
            contrast_idx=contrastive_index
        )

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_crd": loss_crd
        }

        total_loss = loss_ce + loss_crd

        if self.combined_KD:
            from KD import kd_loss

            loss_kd = kd_loss(logits_student, logits_teacher, self.temperature)
            loss_dict['loss_kd'] = loss_kd

            total_loss += loss_kd

        return logits_student, loss_dict, total_loss

    def get_learnable_parameters(self):
        return (
                super().get_learnable_parameters()
                + list(self.embed_teacher.parameters())
                + list(self.embed_student.parameters())
        )

    def get_extra_parameters(self):
        params = (
                list(self.embed_student.parameters())
                + list(self.embed_teacher.parameters())
                + list(self.contrast.buffers())
        )
        num_p = 0

        for p in params:
            num_p += p.numel()
        return num_p

    # img, target, index, sample_idx from train_loader
    def crd_loss(self, student_feature, teacher_feature, idx, contrast_idx):
        teacher_feature = self.embed_teacher(teacher_feature)
        student_feature = self.embed_student(student_feature)

        student_out, teacher_out = self.contrast(student_feature, teacher_feature, idx, contrast_idx)
        student_loss = self.student_criterion(student_out)
        teacher_loss = self.teacher_criterion(teacher_out)

        return student_loss + teacher_loss


class Embed(nn.Module):
    """Embedding Module"""

    def __init__(self, dim_in, dim_out):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = normalize(x)

        return x


class ContrastLoss(nn.Module):
    """ contrastive loss """

    def __init__(self, num_data):
        super(ContrastLoss, self).__init__()
        self.num_data = num_data

    def forward(self, x):
        eps = 1e-7
        batch_size = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.num_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class ContrastMemory(nn.Module):
    """ memory buffer that supplies large amount of negative samples"""

    def __init__(
            self,
            input_size,
            output_size,
            neg_num,
            temperature,
            momentum,
    ):
        super(ContrastMemory, self).__init__()
        self.nLem = output_size
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()

        self.neg_num = neg_num

        self.register_buffer('params', torch.tensor([neg_num, temperature, -1, -1, momentum]))
        stdv = 1. / math.sqrt(input_size / 3)
        self.register_buffer('memory_v1', torch.randn(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.randn(output_size, input_size).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample

        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
     From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())

        neg_num = len(probs)
        self.prob = torch.zeros(neg_num)
        self.alias = torch.LongTensor([0] * neg_num)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1 / K

        smaller = []
        larger = []

        for kk, prob in enumerate(probs):
            self.prob[kk] = neg_num * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop through and create little binary mixtures that
        # appropriately allocate the larger outcomes over the overall uniform mixture

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, sample_n):
        """ Draw N samples from multinomial """
        neg_num = self.alias.size(0)

        kk = torch.zeros(sample_n, dtype=torch.long, device=self.prob.device).random_(0, neg_num)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)

        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj
