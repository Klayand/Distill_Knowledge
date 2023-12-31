import torch

from Solver import Solver, BEST_ACC_DICT
from torchvision import transforms
from backbone import wrn_16_2, wrn_40_2, CIFARNormModel, pyramidnet272
from distiller import KD
from data import get_CIFAR100_train, get_CIFAR100_test, get_edm_cifar10_loader
from LearnWhatYouDontKnow import LearnWhatYouDontKnow
from generators import DifferentiableAutoAug
import torch.nn as nn

# load teacher checkpoint and train student baseline

student_model = CIFARNormModel(wrn_16_2(num_classes=100))
# teacher_model = CIFARNormModel(resnet32x4(num_classes=100))
# ckpt = torch.load("./resources/checkpoints/resnet32x4.pth", map_location=torch.device("cuda"))
# teacher_model.model.load_state_dict(ckpt['model'])

teacher_model = CIFARNormModel(pyramidnet272(num_classes=100))
ckpt = torch.load("./resources/checkpoints/cifar100_pyramid272_top1_11.74.pth", map_location=torch.device("cuda"))
teacher_model.model.load_state_dict(ckpt["model"], strict=False)

distiller = KD(teacher=teacher_model, student=student_model, ce_weight=1.0, kd_weight=1.0, temperature=4).to("cuda")

# ----------------------------------------------------------------------------------------------------------------------
# Training baseline loader

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
train_loader = get_CIFAR100_train(batch_size=128, num_workers=16, transform=transform)
test_loader = get_CIFAR100_test(batch_size=128, num_workers=16)

# train teacher baseline
# w = Solver(teacher=teacher_model, student=student_model, distiller=distiller)
# w.train(train_loader, test_loader, total_epoch=1)
# print()
# print("Teacher model training completed!")
# print()

# for student baseline
student_baseline = CIFARNormModel(wrn_16_2(num_classes=100))
s = Solver(teacher=student_baseline, student=teacher_model, distiller=distiller)
# s.train(train_loader, test_loader, total_epoch=1, is_student=True)
print()
print("Student model without distillation training completed!")
print()


# ----------------------------------------------------------------------------------------------------------------------
# distillation

# w.distill(train_loader, test_loader, total_epoch=1)

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
train_loader = get_edm_cifar10_loader(batch_size=128, num_workers=16, shuffle=True)

generator = DifferentiableAutoAug(student=student_model, teacher=teacher_model)

learn_what_you_dont_konw = LearnWhatYouDontKnow(
    teacher=teacher_model, student=student_model, distiller=distiller, generator=generator
)

_, distillation_acc = learn_what_you_dont_konw.train(
    train_loader=train_loader, validation_loader=test_loader, total_epoch=150
)

print()
print("Student model with distillation training completed!")

print("-" * 100)
print(
    f"teahcer acc: {BEST_ACC_DICT['teacher_acc']}, student acc: {BEST_ACC_DICT['student_acc']}, "
    f"distillation acc: {distillation_acc}"
)
