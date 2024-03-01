import numpy as np
from PIL import Image
import os

# 定义图像数据文件夹路径
data_dir = "/home/classlab2/16T/datasets/boreas./boreas-2021-01-15-12-17/radar/cart_original"

# 获取所有图像文件路径
image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

# 初始化均值和标准差
mean = np.zeros(3)
std = np.zeros(3)

# 遍历所有图像
for image_file in image_files:
    # 读取图像
    image = np.array(Image.open(image_file))
    # print(image.shape)
    # 计算每个通道的均值和标准差
    mean += np.mean(image, axis=(0, 1))
    std += np.std(image, axis=(0, 1))

# 计算总体均值和标准差
mean /= len(image_files)
std /= len(image_files)

print("Mean:", mean)
print("Std:", std)
