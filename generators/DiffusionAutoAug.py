import random

import torch
from torch import nn
import torch.nn.functional as F
from backbone.Diffusion import EDMStochasticSampler, get_edm_cifar_uncond

p = lambda x: nn.Parameter(torch.tensor(x))


def default_generator_loss(student_out, teacher_out, label, config):
    t_loss = F.cross_entropy(teacher_out, label)
    s_loss = F.cross_entropy(student_out, label)

    if s_loss > config.student_max:
        s_loss = s_loss / s_loss.item() * config.student_max

    loss_dict = {
        'teacher_loss': config.generator_alpha * t_loss,
        'student_loss': config.generator_beta * s_loss,
        'total_loss': config.generator_alpha * t_loss - config.generator_beta * s_loss
    }

    return config.generator_alpha * t_loss - config.generator_beta * s_loss, loss_dict


class DiffusionAutoAug(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(DiffusionAutoAug, self).__init__()
        self.device = device

        self.config = config
        self.ori_p = 0.5

        unet = get_edm_cifar_uncond()
        unet.load_state_dict(torch.load("./resources/checkpoints/edm_cifar_uncond_vp.pt"))

        # self.model = EDMStochasticSampler(unet=unet, grad_checkpoint=True)
        self.model = EDMStochasticSampler(unet=unet)

        self.student = student
        self.teacher = teacher

        # self.optimizer = torch.optim.Adam(self.aug.parameters(), lr=config["lr"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.generator_learning_rate)

        self.count = 0
        self.accum_iter = config.accum_iter

    def forward(self, x, y):

        loss_dict = {}

        self.count += 1

        if random.random() < self.ori_p:
            return x, y, loss_dict

        self.model.eval()
        self.model.requires_grad_(True)

        x = self.model.sample(batch_size=x.shape[0])

        with torch.no_grad():
            logits_teacher, teacher_feature = self.teacher(x)
            _, y = torch.max(logits_teacher, dim=1)

        loss, loss_dict = default_generator_loss(self.student(x)[0], self.teacher(x)[0], y, self.config)

        loss = loss / self.accum_iter
        loss.backward()

        if self.count % self.accum_iter == 0:
            self.optimizer.zero_grad()
            self.optimizer.step()

        self.model.requires_grad_(False)

        # return torch.cat([x.detach(), original_x], dim=0), torch.cat([y.detach(), original_y], dim=0)
        return x.detach(), y.detach(), loss_dict