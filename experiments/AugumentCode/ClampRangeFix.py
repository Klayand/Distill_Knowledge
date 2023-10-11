import torch
from kornia import augmentation as KA
from torch.utils.tensorboard import SummaryWriter

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
        "iter_step": 1,
        "lr": 0.1e-4,
        "criterion": default_generator_loss,
    }
    print("generating config:")
    print(x)
    print("-" * 100)
    return x


class DifferentiableAutoAug(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config=default_generating_configuration(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(DifferentiableAutoAug, self).__init__()
        self.device = device
        self.aug = KA.AugmentationSequential(AutoAugment(policy="cifar10"))
        self.optimizer = torch.optim.Adam(self.aug.parameters(), lr=config["lr"])
        self.student = student
        self.teacher = teacher
        self.config = config

    @torch.no_grad()
    def clamp(self):
        for name, param in self.aug.named_parameters():
            postfix = name.split('.')[-1]
            if postfix == 'magnitude_range':
                if 'ShearX' in name:
                    param = param.data.clamp_(min=-54.0, max=54.0)
                # elif 'ShearY' in name:
                #     param = param.data.clamp_(min=0.0, max=0.3)
                elif 'TranslateX' in name:
                    param = param.data.clamp_(min=-0.5, max=0.5)
                elif 'TranslateY' in name:
                    param = param.data.clamp_(min=-0.5, max=0.5)
                elif 'Rotate' in name:
                    param = param.data.clamp_(min=-30.0, max=30.0)
                # elif 'Solarize' in name:
                #     param = param.data.clamp_(min=0.0, max=1.0)
                # elif 'Posterize' in name:
                #     param = param.data.clamp_(min=1.0, max=8.0)
                elif 'Contrast' in name:
                    param = param.data.clamp_(min=0.1, max=1.9)
                elif 'Brightness' in name:
                    param = param.data.clamp_(min=0.1, max=1.9)
                elif 'Sharpness' in name:
                    param = param.data.clamp_(min=0.1, max=1.9)
                elif 'Color' in name:
                    param = param.data.clamp_(min=0.1, max=1.9)

    def forward(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.aug.requires_grad_(True)
        self.student.eval()

        original_x = x.clone()
        original_y = y.clone()

        for _ in range(self.config["iter_step"]):
            x = original_x
            x = self.aug(x)
            loss = self.config["criterion"](self.student(x)[0], self.teacher(x)[0], y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.clamp()

        # give back
        self.aug.requires_grad_(False)
        self.student.train()
        self.student.requires_grad_(True)

        # prepare for final
        # with torch.no_grad():
        #     x = self.normalize_back(original_x.clone())
        #     x = self.aug(x).detach()

        return torch.cat([x.detach(), original_x], dim=0), torch.cat([y.detach(), original_y], dim=0)
