import torch
import torch.distributed as dist
import argparse

from afn_solver import Solver, BEST_ACC_DICT
from Normalizations import build_ASRNormBN2d, build_ASRNormIN
from torchvision import transforms
from backbone import model_dict, CIFARNormModel, ImageNetNormModel, MnistNormModel, MnistmNormModel, SvhnNormModel, UspsNormModel
from data import (
            get_CIFAR100_train,
            get_CIFAR100_test,
            get_imagenet_loader,
            get_CIFAR10_train,
            get_CIFAR10_test,
            get_fashion_mnist_train,
            get_fashion_mnist_test,
            get_mnist_train,
            get_mnist_test,
            get_mnist_m_test,
            get_mnist_m_train,
            get_usps_train,
            get_usps_test,
            get_svhn_train,
            get_svhn_test
)


# load teacher checkpoint and train student baseline
parser = argparse.ArgumentParser(description="hyper-parameters")

parser.add_argument('--ddp_mode', type=bool, default=False, help='Distributed DataParallel Training?')
parser.add_argument('--sync_bn', type=bool, default=False)
parser.add_argument('--norm_layer', type=str, default='BN')
parser.add_argument('--model', type=str)
parser.add_argument('--name', type=str, help='Experiment name')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='CIFAR100')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--ckpt', type=str, default='./resources/checkpoints/')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=128)

parser = parser.parse_args()
print("generating config:")

print(f"Config: {parser}")
print('-'*100)


# -------- initialize model ----------------
model = model_dict[parser.model](num_classes=parser.num_classes)

# -------- transfer learning or not -----------
if parser.norm_layer != 'BN':
    if parser.norm_layer == 'AFN':
        build_ASRNormBN2d(model, True)
    elif parser.norm_layer == 'ASRNorm':
        build_ASRNormIN(model, True)

# ------- DDP -----------------
if parser.ddp_mode:
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    if parser.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

# ------- Normalized Model ----------------
if parser.dataset == 'CIFAR10':
    model = CIFARNormModel(model)
elif parser.dataset == 'CIFAR100':
    model = CIFARNormModel(model)
elif parser.dataset == 'ImageNet':
    model = ImageNetNormModel(model)
elif parser.dataset == 'MNIST':
    model = MnistNormModel(model)
elif parser.dataset == 'M-MNIST':
    model = MnistmNormModel(model)
elif parser.dataset == 'SVHN':
    model = SvhnNormModel(model)
elif parser.dataset == 'USPS':
    model = UspsNormModel(model)


# ----------------------------------------------------------------------------------------------------------------------
# Get dataloader
if parser.dataset == 'CIFAR10':
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
    train_loader = get_CIFAR10_train(batch_size=parser.batch_size, num_workers=16, transform=transform, ddp=parser.ddp_mode)
    test_loader = get_CIFAR10_test(batch_size=parser.batch_size, num_workers=16, ddp=parser.ddp_mode)

elif parser.dataset == 'CIFAR100':
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
    train_loader = get_CIFAR100_train(batch_size=parser.batch_size, num_workers=16, transform=transform, ddp=parser.ddp_mode)
    test_loader = get_CIFAR100_test(batch_size=parser.batch_size, num_workers=16, ddp=parser.ddp_mode)

elif parser.dataset == 'ImageNet':
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_loader = get_imagenet_loader(split='train', batch_size=parser.batch_size, pin_memory=True, transform=transform, ddp=parser.ddp_mode)
    test_loader = get_imagenet_loader(split='val', batch_size=parser.batch_size, pin_memory=True, ddp=parser.ddp_mode)
elif parser.dataset == 'ImageNet':
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_loader = get_imagenet_loader(split='train', batch_size=parser.batch_size, pin_memory=True, transform=transform, ddp=parser.ddp_mode)
    test_loader = get_imagenet_loader(split='val', batch_size=parser.batch_size, pin_memory=True, ddp=parser.ddp_mode)

elif parser.dataset == 'Fashion-MNIST':

    train_loader = get_fashion_mnist_train(batch_size=parser.batch_size, pin_memory=True)
    test_loader = get_fashion_mnist_test(batch_size=parser.batch_size, pin_memory=True)

elif parser.dataset == 'MNIST':

    train_loader = get_mnist_train(batch_size=parser.batch_size, pin_memory=True)
    test_loader = get_mnist_test(batch_size=parser.batch_size, pin_memory=True)

elif parser.dataset == 'M-MNIST':

    train_loader = get_mnist_m_train(batch_size=parser.batch_size, pin_memory=True)
    test_loader = get_mnist_m_test(batch_size=parser.batch_size, pin_memory=True)

elif parser.dataset == 'SVHN':

    train_loader = get_svhn_train(batch_size=parser.batch_size, pin_memory=True)
    test_loader = get_svhn_test(batch_size=parser.batch_size, pin_memory=True)

elif parser.dataset == 'USPS':

    train_loader = get_usps_train(batch_size=parser.batch_size, pin_memory=True)
    test_loader = get_svhn_test(batch_size=parser.batch_size, pin_memory=True)

# train teacher baseline
if parser.ddp_mode:
    w = Solver(
        model=model,
        config=parser,
        local_rank=local_rank
    )
else:
    w = Solver(
        model=model,
        config=parser,
    )

print()
print("Student model without distillation training completed!")
print()

# ----------------------------------------------------------------------------------------------------------------------
w.train(train_loader, test_loader, total_epoch=parser.epochs)

if parser.ddp_mode:
    dist.destroy_process_group()

print("-" * 100)
print(
    f"model acc: {BEST_ACC_DICT['teacher_acc']} "
)
