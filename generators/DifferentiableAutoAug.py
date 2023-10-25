import torch
from kornia import augmentation as KA
from torch.utils.tensorboard import SummaryWriter

from generators.AutoAug import AutoAugment
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

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


# def default_generating_configuration(config):
#     x = {
#         "iter_step": config.iter_step,
#         "lr": config.generator_learning_rate,
#         "criterion": default_generator_loss(config),
#     }
#     print("generating config:")
#     print(x)
#     print("-" * 100)
#     return x


class DifferentiableAutoAug(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(DifferentiableAutoAug, self).__init__()
        self.device = device
        # self.aug = KA.AugmentationSequential(AutoAugment(policy="cifar10"))

        # self.aug = KA.AugmentationSequential(
        #     AutoAugment(policy="cifar10"),
        #     AutoAugment(policy="cifar10"),
        #     AutoAugment(policy="cifar10")
        # )
        self.model_list = nn.Sequential()

        self.config = config

        for i in range(self.config.num_ka):
            self.model_list.add_module(f'model{i + 1}', KA.AugmentationSequential(AutoAugment(policy='cifar10')))

        self.student = student
        self.teacher = teacher

        # self.optimizer = torch.optim.Adam(self.aug.parameters(), lr=config["lr"])
        self.optimizer = torch.optim.Adam(self.model_list.parameters(), lr=self.config.generator_learning_rate)

    @torch.no_grad()
    def clamp(self, model):
        for name, param in model.named_parameters():
            postfix = name.split('.')[-1]
            if postfix == 'magnitude_range':
                if 'ShearX' in name:
                    param.data.clamp_(min=-54.0, max=54.0)
                # elif 'ShearY' in name:
                #     param.data.clamp_(min=0.0, max=0.3)
                elif 'TranslateX' in name:
                    param.data.clamp_(min=-0.5, max=0.5)
                elif 'TranslateY' in name:
                    param = param.data.clamp_(min=-0.5, max=0.5)
                elif 'Rotate' in name:
                    param = param.data.clamp_(min=-30.0, max=30.0)
                # elif 'Solarize' in name:
                #     param.data.clamp_(min=0.0, max=1.0)
                # elif 'Posterize' in name:
                #     param.data.clamp_(min=1.0, max=8.0)
                elif 'Contrast' in name:
                    param.data.clamp_(min=0.1, max=1.9)
                elif 'Brightness' in name:
                    param.data.clamp_(min=0.1, max=1.9)
                elif 'Sharpness' in name:
                    param.data.clamp_(min=0.1, max=1.9)
                elif 'Color' in name:
                    param.data.clamp_(min=0.1, max=1.9)

    def forward(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        #self.aug.requires_grad_(True)
        self.model_list.requires_grad_(True)

        self.student.eval()

        original_x = x.clone()
        original_y = y.clone()

        for _ in range(self.config.iter_step):
            x = original_x

            # 最好不要使用clamp_，这个是inplace操作，会导致错误。
            for model in self.model_list:
                x = model(x).clamp(min=0, max=1)
                self.clamp(model)

            loss, loss_dict = default_generator_loss(self.student(x)[0], self.teacher(x)[0], y, self.config)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        # give back
        self.model_list.requires_grad_(False)
        self.student.train()
        self.student.requires_grad_(True)

        # prepare for final
        # with torch.no_grad():
        #     x = self.normalize_back(original_x.clone())
        #     x = self.aug(x).detach()

        # return torch.cat([x.detach(), original_x], dim=0), torch.cat([y.detach(), original_y], dim=0)
        return x.detach(), y.detach(), loss_dict