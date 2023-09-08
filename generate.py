from backbone.Diffusion import get_edm_cifar_uncond, EDMStochasticSampler
import torch

uncond_edm = get_edm_cifar_uncond()
uncond_edm.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
sampler = EDMStochasticSampler(uncond_edm)
total_images = 1000000
batch_size = 64
num_classes = 10
device = torch.device("cuda")
for batch in range(total_images // batch_size):
    y = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)
    x = sampler(y=y)
    # TODO: 你自己存一下x和y，存成数据集，存成someset的格式，参考data/someset.py
