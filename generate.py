from backbone.Diffusion import get_edm_cifar_uncond, EDMStochasticSampler
import torch
from PIL import Image
from tqdm import tqdm

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert len(input_tensor.shape) == 3

    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()

    image = Image.fromarray(input_tensor)
    image.save(filename)


uncond_edm = get_edm_cifar_uncond()
uncond_edm.load_state_dict(torch.load("resources/checkpoints/edm_cifar_uncond_vp.pt"))
sampler = EDMStochasticSampler(uncond_edm)
total_images = 1000000
batch_size = 64
num_classes = 10
device = torch.device("cuda")
count = 1
for batch in tqdm(range(total_images // batch_size)):
    x = sampler(batch_size=batch_size)
    for i in range(x.shape[0]):
        tensor = x[i]
        save_image_tensor2cv2(filename=f"resources/data/edm/{count}.jpg", input_tensor=tensor)
        count += 1
