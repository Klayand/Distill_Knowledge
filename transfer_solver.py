import os

import torch
from torch import nn
from torch import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from optimizer import SGD, Adam
from scheduler import ALRS, CosineLRS, Lambda_ImageNet
import cv2


def ce_loss(x, y):
    cross_entropy = F.cross_entropy(x, y)
    return cross_entropy


class Solver:
    def __init__(
            self,
            model: nn.Module,
            loss_function: Callable or None = None,
            optimizer: torch.optim.Optimizer or None = None,
            scheduler: Callable or None = None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            local_rank=None
    ):
        self.local_rank = local_rank

        self.teacher = model

        self.criterion = loss_function if loss_function is not None else ce_loss

        self.teacher_optimizer = optimizer if optimizer is not None else SGD(self.teacher)

        self.teacher_scheduler = scheduler if scheduler is not None else ALRS(self.teacher_optimizer)

        self.device = device

        self.teacher_path = None

        # initialization
        self.init()

    def init(self):
        # change device
        self.teacher.to(self.device)

    def train(
            self, train_loader: DataLoader, validation_loader: DataLoader, total_epoch=350, fp16=False, is_student=False
    ):
        from torch.cuda.amp import autocast, GradScaler

        scaler = GradScaler()

        for epoch in range(1, total_epoch + 1):
            train_loss, train_acc, validation_loss, validation_acc = 0, 0, 0, 0

            # train teacher model
            self.teacher.train()
            # train
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                if x.shape[1] == 1:
                    x = x.expand(-1, 3, -1, -1)
                x, y = x.to(self.device), y.to(self.device)

                if fp16:
                    with autocast():
                        teacher_out, _ = self.teacher(x)
                        _, pre = torch.max(teacher_out, dim=1)
                        loss = self.criterion(teacher_out, y)

                else:
                    teacher_out, _ = self.teacher(x)
                    _, pre = torch.max(teacher_out, dim=1)
                    loss = self.criterion(teacher_out, y)

                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()

                self.teacher_optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.teacher_optimizer)
                    # nn.utils.clip_grad_value_(self.teacher.parameters(), 0.1)
                    # nn.utils.clip_grad_norm(self.teacher.parameters(), max_norm=10)
                    scaler.step(self.teacher_optimizer)
                    scaler.update()

                else:
                    loss.backward()
                    # nn.utils.clip_grad_value_(self.teacher.parameters(), 0.1)
                    # nn.utils.clip_grad_norm(self.teacher.parameters(), max_norm=10)
                    self.teacher_optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix_str(f"loss={train_loss / step}, acc={train_acc / step}")

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            # validation
            vbar = tqdm(validation_loader, colour="yellow")
            self.teacher.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(vbar, 1):
                    if x.shape[1] == 1:
                        x = x.expand(-1, 3, -1, -1)
                    x, y = x.to(self.device), y.to(self.device)
                    teacher_out, _ = self.teacher(x)
                    _, pre = torch.max(teacher_out, dim=1)
                    loss = self.criterion(teacher_out, y)

                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)
                    validation_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    validation_loss += loss.item()

                    if step % 10 == 0:
                        vbar.set_postfix_str(f"loss={validation_loss / step}, acc={validation_acc / step}")

                validation_loss /= len(validation_loader)
                validation_acc /= len(validation_loader)

            self.teacher_scheduler.step(train_loss, epoch)

            print(f"epoch {epoch}, train_loss = {train_loss}, train_acc = {train_acc}")
            print(f"epoch {epoch}, validation_loss = {validation_loss}, validation_acc = {validation_acc}")
            print("*" * 100)

            if is_student:
                if os.path.exists('student_baseline.pth'):
                    torch.save(self.teacher.state_dict(), 'student_baseline.pth')
            else:
                torch.save(self.teacher.state_dict(), 'teacher.pth')
                self.teacher_path = 'teacher.pth'

        return self.teacher


if __name__ == '__main__':
    import torchvision
    from Normalizations import ASRNormBN2d, ASRNormIN, build_ASRNormIN, build_ASRNormBN2d, ASRNormLN, build_ASRNormLN
    # from torchvision.models import resnet18
    from data import get_mnist_train, get_svhn_test, get_usps_test, get_svhn_train, get_mnist_test, get_usps_train, get_mnist_m_test
    from backbone import convnet

    a = convnet(dim=1, norm_layer=nn.BatchNorm2d, num_classes=10)

    # freeze_weights(a, nn.BatchNorm2d)
    # build_ASRNormLN(a, True)
    # build_ASRNormBN2d(a, True)

    # train_loader = get_CIFAR10_train(batch_size=128, augment=True)
    # test_loader = get_CIFAR10_test(batch_size=256)

    train_loader = get_mnist_train(batch_size=64, num_workers=16)
    test_loader = get_usps_test(batch_size=64, num_workers=16)

    w = Solver(a)
    w.train(train_loader, test_loader, total_epoch=100)
