import torch
from torch import nn
from torch import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from optimizer import SGD, Adam
from scheduler import ALRS, CosineLRS

BEST_ACC_DICT = {
            'teacher_acc': -999,
            'student_acc': -999,
            'distillation_acc': -999
        }


def ce_loss(x, y):
    cross_entropy = F.cross_entropy(x, y)
    return cross_entropy


class Solver():
    def __init__(self, teacher: nn.Module, student: nn.Module or None = None,
                 distiller: nn.Module or None = None,
                 loss_function: Callable or None = None,
                 optimizer: torch.optim.Optimizer or None = None,
                 scheduler: Callable or None = None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 ):
        self.student = student
        self.teacher = teacher

        self.distiller = distiller

        self.criterion = loss_function if loss_function is not None else ce_loss
        self.student_optimizer = optimizer if optimizer is not None else SGD(self.student)
        self.teacher_optimizer = optimizer if optimizer is not None else SGD(self.teacher)

        self.student_scheduler = scheduler if scheduler is not None else ALRS(self.student_optimizer)
        self.teacher_scheduler = scheduler if scheduler is not None else ALRS(self.teacher_optimizer)

        self.device = device

        self.teacher_path = None

        # initialization
        self.init()

    def init(self):
        # change device
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.distiller.to(self.device)

        # module parameters initialization

    def train(self,
              train_loader: DataLoader,
              validation_loader: DataLoader,
              total_epoch=500,
              fp16=False,
              is_student=False
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
            vbar = tqdm(validation_loader, colour='yellow')
            self.teacher.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(vbar, 1):
                    x, y = x.to(self.device), y.to(self.device)
                    teacher_out, _ = self.teacher(x)
                    _, pre = torch.max(teacher_out, dim=1)
                    loss = self.criterion(teacher_out, y)

                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)
                    validation_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    validation_loss += loss.item()

                    if step % 10 == 0:
                        vbar.set_postfix_str(f'loss={validation_loss / step}, acc={validation_acc / step}')

                validation_loss /= len(validation_loader)
                validation_acc /= len(validation_loader)

                if is_student:
                    if validation_acc > BEST_ACC_DICT['student_acc']:
                        BEST_ACC_DICT['student_acc'] = validation_acc
                else:
                    BEST_ACC_DICT['teacher_acc'] = validation_acc

            self.teacher_scheduler.step(train_loss, epoch)

            print(f'epoch {epoch}, train_loss = {train_loss}, train_acc = {train_acc}')
            print(f'epoch {epoch}, validation_loss = {validation_loss}, validation_acc = {validation_acc}')
            print('*' * 100)

            if is_student:
                torch.save(self.teacher.state_dict(), 'student_baseline.pth')
            else:
                torch.save(self.teacher.state_dict(), 'teacher.pth')
                self.teacher_path = 'teacher.pth'

        return self.teacher

    def distill(self,
                train_loader: DataLoader,
                validation_loader: DataLoader,
                total_epoch=500,
                fp16=False,
                ):

        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

        for epoch in range(1, total_epoch + 1):
            train_loss, validation_loss, validation_acc = 0, 0, 0

            if self.teacher_path:
                self.teacher.load_state_dict(torch.load(self.teacher_path))

            # train teacher model
            self.teacher.eval()
            self.student.train()
            # train
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)
                if fp16:
                    with autocast():
                        student_logits, losses_dict, loss = distiller.forward_train(image=x, target=y)

                else:
                    student_logits, losses_dict, loss = distiller.forward_train(image=x, target=y)

                train_loss += loss.item()

                self.student_optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.student_optimizer)
                    # nn.utils.clip_grad_value_(self.teacher.parameters(), 0.1)
                    # nn.utils.clip_grad_norm(self.teacher.parameters(), max_norm=10)
                    scaler.step(self.student_optimizer)
                    scaler.update()

                else:
                    loss.backward()
                    # nn.utils.clip_grad_value_(self.teacher.parameters(), 0.1)
                    # nn.utils.clip_grad_norm(self.teacher.parameters(), max_norm=10)
                    self.student_optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix_str(f"distill loss={train_loss / step}")

            train_loss /= len(train_loader)

            # validation
            vbar = tqdm(validation_loader, colour='yellow')
            self.student.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(vbar, 1):
                    x, y = x.to(self.device), y.to(self.device)
                    student_out, _, = self.student(x)
                    _, pre = torch.max(student_out, dim=1)
                    loss = self.criterion(student_out, y)

                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)
                    validation_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    validation_loss += loss.item()

                    if step % 10 == 0:
                        vbar.set_postfix_str(f'loss={validation_loss / step}, acc={validation_acc / step}')

                validation_loss /= len(validation_loader)
                validation_acc /= len(validation_loader)

                if validation_acc > BEST_ACC_DICT['distillation_acc']:
                    BEST_ACC_DICT['distillation_acc'] = validation_acc

            self.student_scheduler.step(train_loss, epoch)

            print(f'epoch {epoch}, distill_loss = {train_loss}')
            print(f'epoch {epoch}, validation_loss = {validation_loss}, validation_acc = {validation_acc}')
            print('*' * 100)

            torch.save(self.student.state_dict(), 'student.pth')

        return self.student


if __name__ == '__main__':
    import torchvision
    from backbone import resnet32, resnet14
    from distiller import CenterKernelAnalysisRKD, DistanceWiseRKD
    from data import get_CIFAR100_train, get_CIFAR100_test

    student_model = resnet14(num_classes=100)
    teacher_model = resnet32(num_classes=100)

    distiller = CenterKernelAnalysisRKD(teacher=teacher_model, student=student_model).to('cuda')

    train_loader = get_CIFAR100_train(batch_size=128, num_workers=8, augment=True)
    test_loader = get_CIFAR100_test(batch_size=128, num_workers=8)

    w = Solver(teacher=teacher_model, student=student_model, distiller=distiller)
    w.train(train_loader, test_loader, total_epoch=200)

    print()
    print("Teacher model training completed!")

    # for student baseline
    s = Solver(teacher=student_model)
    s.train(train_loader, test_loader, total_epoch=200)
    print()
    print("Student model without distillation training completed!")

    w.distill(train_loader, test_loader, total_epoch=200)

    print()
    print("Student model with distillation training completed!")

    print('-'*100)
    print(f"teahcer acc: {BEST_ACC_DICT['teacher']}, student acc: {BEST_ACC_DICT['student']}, "
          f"distillation acc: {BEST_ACC_DICT['distillation_acc']}")

