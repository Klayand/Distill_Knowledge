import torch
import argparse

from Solver import Solver, BEST_ACC_DICT
from torchvision import transforms
from backbone import model_dict, CIFARNormModel, ImageNetNormModel
from distiller import KD
from data import get_CIFAR100_train, get_CIFAR100_test, get_imagenet_loader
from LearnWhatYouDontKnow import LearnWhatYouDontKnow
from generators import DifferentiableAutoAug
import torch.distributed as dist

# load teacher checkpoint and train student baseline
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
parser.add_argument('--student_max', type=float, default=1.5)
parser.add_argument('--iter_step', type=int, default=1)
parser.add_argument('--num_ka', type=int, default=3)
parser.add_argument('--generator_alpha', type=int, default=2)
parser.add_argument('--generator_beta', type=int, default=1)
parser.add_argument('--generator_learning_rate', type=float, default=1e-3)

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
    student_baseline = CIFARNormModel(student_baseline)
    student_model = CIFARNormModel(student_model)

    teacher_model = CIFARNormModel(teacher_model)

elif parser.dataset == 'ImageNet':
    student_baseline = ImageNetNormModel(student_baseline)
    student_model = ImageNetNormModel(student_model)

    teacher_model = ImageNetNormModel(teacher_model)

# ---------- load ckpt ---------------
if parser.pretrained:
    if parser.dataset == 'CIFAR':
        ckpt = torch.load(f"{parser.ckpt}/{parser.teacher}.pth")
        teacher_model.model.load_state_dict(ckpt["model"])
    elif parser.dataset == 'ImageNet':
        # ckpt = torch.load(f"{parser.ckpt}/{parser.teacher}_imagenet.pth")
        ckpt = torch.load(f"{parser.ckpt}/{parser.teacher}.pth")
        teacher_model.model.load_state_dict(ckpt)
    print("finished loading pretrained model")

distiller = KD(teacher=teacher_model, student=student_model, ce_weight=1.0, kd_weight=1.0, temperature=4).to("cuda")

# ----------------------------------------------------------------------------------------------------------------------
# Training baseline loader

# transform = transforms.Compose(
#     [
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
#         transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
#         transforms.RandomRotation(5),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
#     ]
# )
# train_loader = get_CIFAR100_train(batch_size=128, num_workers=16, transform=transform)

# train teacher baseline
# w = Solver(teacher=teacher_model, student=student_model, distiller=distiller)
# w.train(train_loader, test_loader, total_epoch=1)
# print()
# print("Teacher model training completed!")
# print()

# for student baseline
# s = Solver(teacher=student_baseline, student=teacher_model, distiller=distiller)
# s.train(train_loader, test_loader, total_epoch=1, is_student=True)
print()
print("Student model without distillation training completed!")
print()

# ----------------------------------------------------------------------------------------------------------------------
# distillation

# w.distill(train_loader, test_loader, total_epoch=1)

# --------- train model generator ---------------

if parser.dataset == 'CIFAR':
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_loader = get_imagenet_loader(split='train', batch_size=512, pin_memory=True, transform=transform, ddp=parser.ddp_mode)
    test_loader = get_imagenet_loader(split='val', batch_size=512, pin_memory=True, ddp=parser.ddp_mode)

generator = DifferentiableAutoAug(student=student_model, teacher=teacher_model, config=parser)

if parser.ddp_mode:
    learn_what_you_dont_konw = LearnWhatYouDontKnow(
        teacher=teacher_model,
        student=student_model,
        distiller=distiller,
        generator=generator,
        config=parser,
        local_rank=local_rank
    )
else:
    learn_what_you_dont_konw = LearnWhatYouDontKnow(
        teacher=teacher_model,
        student=student_model,
        distiller=distiller,
        generator=generator,
        config=parser,
    )

_, distillation_acc = learn_what_you_dont_konw.train(
    train_loader=train_loader, validation_loader=test_loader, total_epoch=600
)

BEST_ACC_DICT['distillation_acc'] = distillation_acc

if parser.ddp_mode:
    dist.destroy_process_group()

print()
print("Student model with distillation training completed!")

print("-" * 100)
print(
    f"teahcer acc: {BEST_ACC_DICT['teacher_acc']}, student acc: {BEST_ACC_DICT['student_acc']},"
    f"distillation acc: {BEST_ACC_DICT['distillation_acc']}"
)
