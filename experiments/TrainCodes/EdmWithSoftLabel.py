import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from optimizer import SGD, Adam
from scheduler import ALRS


def ce_loss(x, y):
    cross_entropy = F.cross_entropy(x, y)
    return cross_entropy


class LearnWhatYouDontKnow:
    def __init__(
            self,
            teacher: nn.Module,
            student: nn.Module,
            distiller: nn.Module or None = None,
            generator=None,
            loss_function: Callable or None = None,
            optimizer: torch.optim.Optimizer or None = None,
            scheduler=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.teacher = teacher
        self.student = student

        self.distiller = distiller
        self.generator = generator

        self.criterion = loss_function if loss_function is not None else ce_loss
        self.optimizer = optimizer if optimizer is not None else SGD(self.student)
        self.scheduler = scheduler if scheduler is not None else ALRS(self.optimizer)
        self.device = device

        # initialization
        self.init()

    def init(self):
        # ban teacher gradient
        self.teacher.requires_grad_(False)

        # change device
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.teacher.eval()

        # tensorboard
        self.writer = SummaryWriter(log_dir="runs/baseline/edm_with_distillation")

    def train(
            self,
            train_loader: DataLoader,
            validation_loader: DataLoader,
            total_epoch=500,
            fp16=False,
    ):
        """

        :param total_epoch:
        :param step_each_epoch: this 2 parameters is just a convention, for when output loss and acc, etc.
        :param fp16:
        :param generating_data_configuration:
        :return:
        """
        from torch.cuda.amp import autocast, GradScaler

        BEST_ACC = -999

        scaler = GradScaler()
        self.teacher.eval()

        # train
        for epoch in range(1, total_epoch + 1):
            train_loss = 0

            pbar = tqdm(train_loader)
            self.student.train()
            for step, x in enumerate(pbar, 1):
                x = x.to(self.device)

                if fp16:
                    with autocast():
                        # distillation part
                        student_logits, losses_dict, loss = self.distiller.forward_train(image=x)
                else:
                    # distillation part
                    student_logits, losses_dict, loss = self.distiller.forward_train(image=x)

                train_loss += loss.item()
                self.optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    # nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    self.optimizer.step()

            train_loss /= len(train_loader)

            self.scheduler.step(train_loss, epoch)

            # tensorboard
            self.writer.add_scalar("train/loss", train_loss, epoch)

            # validation
            vbar = tqdm(validation_loader, colour="yellow")

            test_loss, test_acc = 0, 0

            self.student.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(vbar, 1):
                    x, y = x.to(self.device), y.to(self.device)

                    student_out, student_feature = self.student(x)
                    _, pre = torch.max(student_out, dim=1)

                    loss = self.criterion(student_out, y)

                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)

                    test_acc += (torch.sum(pre == y)).item() / y.shape[0]
                    test_loss += loss.item()

                    if step % 10 == 0:
                        vbar.set_postfix_str(f"loss={test_loss / step}, acc={test_acc / step}")

                test_loss /= len(validation_loader)
                test_acc /= len(validation_loader)
                self.writer.add_scalar("test/loss", test_loss, epoch)
                self.writer.add_scalar("test/acc", test_acc, epoch)

                if test_acc > BEST_ACC:
                    BEST_ACC = test_acc

            print(f"epoch {epoch}, distilling_loss = {train_loss}")
            print(f"epoch {epoch}, validation_loss = {test_loss}, validation_acc = {test_acc}")
            print("*" * 100)

        print(f"student with distillation best acc {BEST_ACC}")
        torch.save(self.student.state_dict(), 'student_with_distillation.pth')

        return self.student, BEST_ACC
