import torch
import torch.distributed as dist
import argparse

from Solver import Solver, BEST_ACC_DICT
from torchvision import transforms
from backbone import model_dict, CIFARNormModel, ImageNetNormModel
from distiller import DKCKA, CKASP, CenterKernelAlignmentRKD
from data import get_CIFAR100_train, get_CIFAR100_test, get_imagenet_loader


# load teacher checkpoint and train student baseline
parser = argparse.ArgumentParser(description="hyper-parameters")

parser.add_argument('--ddp_mode', type=bool, default=False, help='Distributed DataParallel Training?')
parser.add_argument('--teacher', type=str)
parser.add_argument('--student', type=str)
parser.add_argument('--name', type=str, help='Experiment name')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='CIFAR')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--ckpt', type=str, default='./resources/checkpoints/')

parser = parser.parse_args()
print("generating config:")
print(f"Config: {parser}")
print('-'*100)


# -------- initialize model ----------------
student_baseline = model_dict[parser.student](num_classes=parser.num_classes)

student_model = model_dict[parser.student](num_classes=parser.num_classes)
teacher_model = model_dict[parser.teacher](num_classes=parser.num_classes)

# ------- DDP -----------------
if parser.ddp_mode:
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    student_baseline = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_baseline)

    student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
    teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)

# ------- Normalized Model ----------------
if parser.dataset == 'CIFAR':
    # student_baseline = CIFARNormModel(student_baseline)
    student_model = CIFARNormModel(student_model)

    teacher_model = CIFARNormModel(teacher_model)

elif parser.dataset == 'ImageNet':
    # student_baseline = ImageNetNormModel(student_baseline)
    student_model = ImageNetNormModel(student_model)

    teacher_model = ImageNetNormModel(teacher_model)

# ---------- load ckpt --------------- --
if parser.pretrained:
    if parser.dataset == 'CIFAR':
        ckpt = torch.load(f"{parser.ckpt}/{parser.teacher}.pth")
        teacher_model.model.load_state_dict(ckpt["model"])
    elif parser.dataset == 'ImageNet':
        # ckpt = torch.load(f"{parser.ckpt}/{parser.teacher}_imagenet.pth")
        ckpt = torch.load(f"{parser.ckpt}/{parser.teacher}.pth")
        teacher_model.model.load_state_dict(ckpt)
    print("finished loading pretrained model")

distiller = CenterKernelAlignmentRKD(teacher=teacher_model, student=student_model, rkd_weight=15).to("cuda")

# ----------------------------------------------------------------------------------------------------------------------
# Get dataloader
if parser.dataset == 'CIFAR':
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            # transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]
    )
    train_loader = get_CIFAR100_train(batch_size=128, num_workers=16, transform=transform, ddp=parser.ddp_mode)
    test_loader = get_CIFAR100_test(batch_size=128, num_workers=16, ddp=parser.ddp_mode)

elif parser.dataset == 'ImageNet':
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_loader = get_imagenet_loader(split='train', batch_size=256, pin_memory=True, transform=transform, ddp=parser.ddp_mode)
    test_loader = get_imagenet_loader(split='val', batch_size=256, pin_memory=True, ddp=parser.ddp_mode)

# train teacher baseline
if parser.ddp_mode:
    w = Solver(
        teacher=teacher_model,
        student=student_model,
        distiller=distiller,
        config=parser,
        local_rank=local_rank
    )
else:
    w = Solver(
        teacher=teacher_model,
        student=student_model,
        distiller=distiller,
        config=parser,
    )
# w.train(train_loader, test_loader, total_epoch=1)
# print()
# print("Teacher model training completed!")
# print()

# for student baseline
# s = Solver(teacher=student_baseline, student=teacher_model, distiller=distiller)
# s.train(train_loader, test_loader, total_epoch=500, is_student=True)
print()
print("Student model without distillation training completed!")
print()


# ----------------------------------------------------------------------------------------------------------------------
# distillation

w.distill(train_loader, test_loader, total_epoch=300)

if parser.ddp_mode:
    dist.destroy_process_group()

print()
print("Student model with distillation training completed!")

print("-" * 100)
print(
    f"teahcer acc: {BEST_ACC_DICT['teacher_acc']}, student acc: {BEST_ACC_DICT['student_acc']}, "
    f"distillation acc: {BEST_ACC_DICT['distillation_acc']}"
)
