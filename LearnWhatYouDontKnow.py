import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from optimizer import SGD, Adam
from scheduler import ALRS, LambdaLR, Lambda_EMD
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import CLASS_NAME


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
            config=None,
            local_rank=None
    ):
        self.config = config

        self.visual_path = self.config.name
        self.ddp_mode = self.config.ddp_mode
        self.local_rank = local_rank

        self.teacher = teacher
        self.student = student

        self.distiller = distiller
        self.generator = generator

        self.criterion = loss_function if loss_function is not None else ce_loss

        if 'mobilenet' in self.config.student:
            self.optimizer = optimizer if optimizer is not None else SGD(self.student, lr=0.01)
        else:
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

        if self.ddp_mode:
            # self.teacher = DDP(self.teacher, device_ids=[self.local_rank], output_device=self.local_rank)
            self.student = DDP(self.student, device_ids=[self.local_rank], output_device=self.local_rank)

        self.teacher.eval()

        # tensorboard
        self.writer = SummaryWriter(log_dir=f"runs/baseline/{self.visual_path}")

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
        # self.teacher.train()

        # train
        for epoch in range(1, total_epoch + 1):

            if self.ddp_mode:
                train_loader.sampler.set_epoch(epoch)

            train_loss, train_acc = 0, 0
            train_ce_loss, train_distill_loss = 0, 0
            student_confidence, teacher_confidence = 0, 0

            generator_teacher_loss, generator_student_loss, generator_total_loss = 0, 0, 0

            pbar = tqdm(train_loader)
            self.student.train()
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)
                print(f"GPU id {self.local_rank}, Batch Size {x.shape[0]}")
                x, y, generator_loss_dict = self.generator(x, y)

                if generator_loss_dict:
                    generator_teacher_loss += generator_loss_dict['teacher_loss'].item()
                    generator_student_loss += generator_loss_dict['student_loss'].item()
                    generator_total_loss += generator_loss_dict['total_loss'].item()

                with torch.no_grad():
                    teacher_out, teacher_feature = self.teacher(x)

                    now_teacher_confidence = torch.mean(
                        F.softmax(teacher_out, dim=1)[torch.arange(y.shape[0] // 2), y[: y.shape[0] // 2]]
                    ).item()
                    teacher_confidence += now_teacher_confidence

                if fp16:
                    with autocast():
                        with torch.no_grad():
                            student_out, student_feature = self.student(x)  # N, 60
                        _, pre = torch.max(student_out, dim=1)

                        self.student.train()

                        # distillation part
                        student_logits, losses_dict, loss = self.distiller.forward_train(image=x, target=y)
                else:
                    with torch.no_grad():
                        student_out, student_feature = self.student(x)  # N, 60
                    _, pre = torch.max(student_out, dim=1)

                    self.student.train()

                    # distillation part
                    student_logits, losses_dict, loss = self.distiller.forward_train(image=x, target=y)

                    now_student_confidence = torch.mean(
                        F.softmax(student_out, dim=1)[torch.arange(y.shape[0] // 2), y[: y.shape[0] // 2]]
                    ).item()
                    student_confidence += now_student_confidence

                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]

                train_ce_loss += losses_dict['loss_ce'].item()
                train_distill_loss += losses_dict['loss_kd'].item()

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
                    
                if step % 10 == 0:
                    pbar.set_postfix_str(f"loss={train_loss / step}, acc={train_acc / step}")

            train_loss /= len(train_loader)
            train_ce_loss /= len(train_loader)
            train_distill_loss /= len(train_loader)

            if generator_loss_dict:
                generator_teacher_loss /= len(train_loader)
                generator_student_loss /= len(train_loader)
                generator_total_loss /= len(train_loader)

            train_acc /= len(train_loader)

            self.scheduler.step(train_loss, epoch)

            # tensorboard
            self.writer.add_scalar("confidence/teacher_confidence", teacher_confidence / len(train_loader), epoch)
            self.writer.add_scalar("confidence/student_confidence", student_confidence / len(train_loader), epoch)

            self.writer.add_scalar("train/loss/total_loss", train_loss, epoch)
            self.writer.add_scalar("train/loss/ce_loss", train_ce_loss, epoch)
            self.writer.add_scalar("train/loss/distill_loss", train_distill_loss, epoch)
            
            if generator_loss_dict:
                self.writer.add_scalar("generator/loss/total_loss", generator_total_loss, epoch)
                self.writer.add_scalar("generator/loss/teacher_loss", generator_teacher_loss, epoch)
                self.writer.add_scalar("generator/loss/student_loss", generator_student_loss, epoch)

            self.writer.add_scalar("train/acc", train_acc, epoch)

            self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

            ## tensorboard add image
            if self.config.dataset != 'ImageNet':
                image = x[: x.size(0) // 2][0].squeeze()
                label = CLASS_NAME[y[: y.size(0) // 2][0].squeeze().item()]
                self.writer.add_image(f"generator/images/{label}", image, epoch)

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

            print(f"epoch {epoch}, distilling_loss = {train_loss}, distilling_acc = {train_acc}")
            print(f"epoch {epoch}, validation_loss = {test_loss}, validation_acc = {test_acc}")
            print("*" * 100)

        print(f"student with distillation best acc {BEST_ACC}")
        torch.save(self.student.state_dict(), 'student_with_distillation.pth')

        return self.student, BEST_ACC
