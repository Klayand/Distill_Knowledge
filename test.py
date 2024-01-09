from PIL import Image
import matplotlib.pyplot as plt

# 读取多个图片
image_paths = [
    'pic/resnet110-renset20.png',
    'pic/resnet110-renset32.png',
    'pic/VGG13-MobileNetV2.png',
    'pic/VGG13-VGG8.png',
    'pic/WRN40-2-WRN16-2.png',
    'pic/WRN40-2-WRN40-1.png'
]
images = [Image.open(path) for path in image_paths]

# 同时显示这些图片
for i in range(6):
    plt.subplot(3, 2, i+1)
    plt.imshow(images[i])
plt.show()