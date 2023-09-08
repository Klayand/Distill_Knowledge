import torch

from Solver import Solver, BEST_ACC_DICT
from torchvision import transforms
from backbone import wrn_16_1, wrn_40_2, vgg11_bn, CIFARNormModel
from distiller import KD
from data import get_CIFAR100_train, get_CIFAR100_test
from LearnWhatYouDontKnow import LearnWhatYouDontKnow
from generators import DifferentiableAutoAug

# train teacher model and student baseline model
student_model = CIFARNormModel(wrn_16_1(num_classes=100))
teacher_model = CIFARNormModel(wrn_40_2(num_classes=100))
ckpt = torch.load("./resources/checkpoints/wrn_40_2.pth", map_location=torch.device("cpu"))
teacher_model.model.load_state_dict(ckpt["model"])

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
train_loader = get_CIFAR100_train(batch_size=128, num_workers=1, transform=transform)
test_loader = get_CIFAR100_test(batch_size=128, num_workers=1)

w = Solver(teacher=teacher_model, student=student_model, distiller=distiller)
w.train(train_loader, test_loader, total_epoch=1)
print()
print("Teacher model training completed!")
print()

# for student baseline
student_baseline = CIFARNormModel(wrn_16_1(num_classes=100))
s = Solver(teacher=student_baseline, student=teacher_model, distiller=distiller)
s.train(train_loader, test_loader, total_epoch=1, is_student=True)
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
        # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ]
)
train_loader = get_CIFAR100_train(batch_size=128, num_workers=1, transform=transform)

generator = DifferentiableAutoAug(student=student_model, teacher=teacher_model)

leanrn_what_you_dont_konw = LearnWhatYouDontKnow(
    teacher=teacher_model, student=student_model, distiller=distiller, generator=generator
)

_, distillation_acc = leanrn_what_you_dont_konw.train(
    train_loader=train_loader, validation_loader=test_loader, total_epoch=1
)

print()
print("Student model with distillation training completed!")

print("-" * 100)
print(
    f"teahcer acc: {BEST_ACC_DICT['teacher_acc']}, student acc: {BEST_ACC_DICT['student_acc']}, "
    f"distillation acc: {distillation_acc}"
)
