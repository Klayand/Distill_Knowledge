from backbone.Diffusion import get_edm_cifar_uncond, EDMStochasticSampler
import torch
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert len(input_tensor.shape) == 3

    to_image = transforms.ToPILImage()
    image = to_image(input_tensor)
    image.save(filename)


uncond_edm = get_edm_cifar_uncond()
uncond_edm.load_state_dict(torch.load("resources/checkpoints/edm_cifar_uncond_vp.pt"))
sampler = EDMStochasticSampler(uncond_edm).eval()
total_images = 1000000
batch_size = 64
num_classes = 10
device = torch.device("cuda")
count = 1
for batch in tqdm(range(total_images // batch_size)):
    x = sampler(batch_size=batch_size)
    for i in range(x.shape[0]):
        tensor = x[i]
        save_image_tensor2cv2(filename=f"resources/data/edm-cifar10/{count}.png", input_tensor=tensor)
        count += 1
