import torch
from kornia import augmentation as KA
from generators.AutoAug import AutoAugment
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

p = lambda x: nn.Parameter(torch.tensor(x))


def default_generator_loss(student_out, teacher_out, label, alpha=1, beta=1):
    t_loss = F.cross_entropy(teacher_out, label)
    s_loss = F.cross_entropy(student_out, label)
    return alpha * t_loss - beta * s_loss


def default_generating_configuration():
    x = {
        "iter_step": 0,
        "lr": 0.1e-4,
        "criterion": default_generator_loss,
    }
    print("generating config:")
    print(x)
    print("-" * 100)
    return x


class SimpleAug(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config=default_generating_configuration(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(SimpleAug, self).__init__()
        self.device = device
        self.aug = KA.AugmentationSequential(AutoAugment(policy="cifar10"))
        self.optimizer = torch.optim.Adam(self.aug.parameters(), lr=config["lr"])
        self.student = student
        self.teacher = teacher
        self.config = config

    def forward(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.aug.requires_grad_(True)
        self.student.eval()
        self.student.requires_grad_(False)
        original_x = x.clone()
        original_y = y.clone()

        for _ in range(self.config["iter_step"]):
            x = self.normalize_back(original_x.clone())
            x = self.aug(x)
            loss = self.config["criterion"](self.student(x), self.teacher(x), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # give back
        self.aug.requires_grad_(False)
        self.student.train()
        self.student.requires_grad_(True)

        # prepare for final
        x = original_x.clone()
        # with torch.no_grad():
        #     x = self.normalize_back(original_x.clone())
        #     x = self.aug(x).detach()

        return torch.cat([x.detach(), original_x], dim=0), torch.cat([y.detach(), original_y], dim=0)
