import numpy as np
from PIL import Image

data = np.load("resources/data/edm-cifar100.npz")
images, targets = data['image'], data['label']

image = Image.fromarray(images[0])
print(images[0])
image.show()
print(targets[0])

