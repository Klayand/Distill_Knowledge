import torch

from Solver import Solver, BEST_ACC_DICT
from torchvision import transforms
from backbone import resnet34_imagenet, resnet18_imagenet, ImageNetNormModel
from distiller import KD
from data import get_imagenet_loader
from LearnWhatYouDontKnow import LearnWhatYouDontKnow
from generators import DifferentiableAutoAug

# load teacher checkpoint and train student baseline

student_model = ImageNetNormModel(resnet18_imagenet(num_classes=1000))
teacher_model = ImageNetNormModel(resnet34_imagenet(num_classes=1000))
ckpt = torch.load("./resources/checkpoints/resnet34_imagenet.pth")
teacher_model.model.load_state_dict(ckpt)

distiller = KD(teacher=teacher_model, student=student_model, ce_weight=1.0, kd_weight=1.0, temperature=4).to("cuda")

# ----------------------------------------------------------------------------------------------------------------------
# Training baseline loader

transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
)
train_loader = get_imagenet_loader(batch_size=128, split='train', num_workers=16, shuffle=True, pin_memory=True, transform=transform)
test_loader = get_imagenet_loader(batch_size=128, split='val', num_workers=16)

# train teacher baseline
# w = Solver(teacher=teacher_model, student=student_model, distiller=distiller)
# w.train(train_loader, test_loader, total_epoch=1)
# print()
# print("Teacher model training completed!")
# print()

# for student baseline
student_baseline = ImageNetNormModel(resnet18_imagenet(num_classes=1000))
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
            transforms.Resize((256, 256)),
            #transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
)
train_loader = get_imagenet_loader(batch_size=128, split='train', num_workers=16, shuffle=True, pin_memory=True, transform=transform)
generator = DifferentiableAutoAug(student=student_model, teacher=teacher_model)

learn_what_you_dont_konw = LearnWhatYouDontKnow(
    teacher=teacher_model, student=student_model, distiller=distiller, generator=generator
)

_, distillation_acc = learn_what_you_dont_konw.train(
    train_loader=train_loader, validation_loader=test_loader, total_epoch=600
)

print()
print("Student model with distillation training completed!")

print("-" * 100)
print(
    f"teahcer acc: {BEST_ACC_DICT['teacher_acc']}, student acc: {BEST_ACC_DICT['student_acc']},"
    f"distillation acc: {BEST_ACC_DICT[distillation_acc]}"
)
