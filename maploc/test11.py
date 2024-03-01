# import math

# # def signed_angle_to_degree(signed_angle_rad):
# #     # 将弧度转换为角度
# #     angle_deg = math.degrees(signed_angle_rad)
    
# #     # 将角度映射到 0 到 360 之间
# #     angle_deg = (angle_deg + 360) % 360
    
# #     return angle_deg

# # # 例如，假设有一个有符号弧度为 -π/4
# # signed_angle_rad = -math.pi / 4

# # # 转换为按逆时针定义的 0-360 度的角度值
# # angle_deg = signed_angle_to_degree(signed_angle_rad)
# # print(f"有符号弧度 {signed_angle_rad} 对应的角度是 {angle_deg} 度")
# # angle_deg = 0
# # num_bins = 64  # 划分的份数
# # angle_index = int((angle_deg / 360.0) * num_bins) % num_bins
# # print(angle_index)
# import torch
# def rotate_images(images, num_rotations):
#     rotated_images = []
#     for i in range(num_rotations):
#         # 计算旋转的角度
#         angle = i * (360 / num_rotations)
#         rotated_image = rotate_image(images, angle)
#         rotated_images.append(rotated_image)
#     return torch.stack(rotated_images, dim=1)

# # 定义一个函数，用于旋转单个图像
# def rotate_image(image, angle):
#     # 根据 angle 旋转单个图像
#     #按逆时针方向旋转图像 角度的定义是0-360度
#     # 这里只是一个示例，具体实现取决于你使用的库
#     rotated_image = torch.rot90(image, k=int(angle / 90), dims=(2, 3))
#     return rotated_image

# import torch

# # # 示例的权重和旋转后的图像集合
# # x = torch.rand(1, 64)
# # rotated_images = torch.rand(1, 64, 3, 256, 256)

# # # 调整权重的形状
# # x = x.view(1, 64, 1, 1, 1)

# # # 对图像集合进行加权
# # weighted_images = (rotated_images * x).sum(dim=1)

# # print(weighted_images.shape)
# # yaw = 0.1
# # angle_deg = math.degrees(yaw) 
# # # 将角度映射到 0 到 360 之间
# # angle_deg = (angle_deg + 360) % 360
# # yaw_angle = angle_deg
# # print(yaw_angle)
# import numpy
# from PIL import Image
# from PIL import Image, ImageStat
# from torchvision import transforms
# import cv2
# def calculate_single_channel_mean(image, channel_index):
#     # 打开图像
    
    
#     # 获取图像统计信息
#     stat = ImageStat.Stat(image)
    
#     # 获取指定通道的均值
#     mean_value = stat.mean[channel_index]
    
#     return mean_value
# def get_bin_table(threshold=10):
#     table = []
#     for i in range(256):

#         if i < threshold:

#             table.append(0)

#         else:

#             table.append(i)

#     return table

# import numpy as np

# radar_image_path = "/home/classlab2/16T/datasets/boreas./boreas-2021-01-15-12-17/radar/cart_rotated/1610731110203625.png"
# radar_image = Image.open(radar_image_path)
# radar_image.save("test_radar.png")
# imgry = radar_image.convert('L')
# from torchvision.transforms import ToPILImage
# channel_index = 0  # 替换为你需要的通道索引
# # 计算指定通道的均值
# mean_value = calculate_single_channel_mean(imgry, channel_index)
# table = get_bin_table(threshold=mean_value+10)
# # binary = imgry.point(table, '1')
# binary = imgry.point(table)
# # binary[binary > 0] = 1
# binary.save('binary.png')

# numpy_array = np.array(binary)
# tensor_image = torch.from_numpy(numpy_array)
# tensor_image = tensor_image.float()
# tensor_image[tensor_image > 0] = 1


# print(torch.max(tensor_image))#直方图均衡化
# target_pil = ToPILImage()(tensor_image.cpu())
# target_pil.save("target_image1111.png")

# radar_image_path1 = "/home/classlab2/16T/datasets/boreas./boreas-2021-01-15-12-17/radar/cart_rotated/1610731110203625.png"
# print(radar_image_path1)
# radar_image1 = Image.open(radar_image_path1)

# # tensor_radar_image1 = transform(radar_image1)
# # from torchvision.transforms import ToPILImage
# # import torch.nn.functional as F



# # target = tensor_radar_image1#【1，3，256，256】
# # target = torch.mean(target.float(), dim=1, keepdim=True)#[1,1,256,256]
# # # target_gray = torch.mean(target.float(), dim=-1, keepdim=False)
# # # target_resized = F.interpolate(target_gray.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)
# # # target_resized = target_resized.squeeze(1)
# # orign = tensor_radar_image
# # origin = torch.mean(orign.float(), dim=1, keepdim=True)
# # # 保存目标图像
# # target_pil = ToPILImage()(target[0].cpu())
# # # target_pil.save("target_image.png")
# # # 保存原始图像
# # origin_pil = ToPILImage()(orign[0].cpu())
# # origin_pil.save("origin_image.png")

# # # 获取旋转的图像集合
# # rotated_images_set = rotate_images(orign, self.num_classes)#[B,N,C,H,W]【1，64，1，256，256】
# # # print(rotated_images_set.shape)
# # batch_size  = rotated_images_set.shape[0]
# # # 对图像集合进行加权
# # # weighted_images = torch.einsum("bncwh,bn->bwh",rotated_images, x)#【1，1，256，256】
# # weighted_images = (rotated_images_set * x.view(batch_size, self.num_classes, 1, 1, 1)).sum(dim=1)
# # weigthed_pil = ToPILImage()(weighted_images[0].cpu())
# # # weigthed_pil.save("weighted_image.png")
# # target_label = data["label"]
# # 计算加权得到的预测图像和目标图像之间的 L1 损失
# # loss = F.l1_loss(orign, target,reduction='sum')
# # print(loss)

import torch
import torch.nn as nn

# 模型定义
class YourModel(nn.Module):
    def __init__(self, input_channels, num_concatenations):
        super(YourModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 更多你的模型层...
        )
        self.fc_layers = nn.Linear(128 * num_concatenations * H * W, 10)  # 假设输出大小是10

    def forward(self, x):
        # 卷积层
        conv_output = self.conv_layers(x)
        
        # 将张量展平，以便输入全连接层
        flattened_output = conv_output.view(x.size(0), -1)

        # 全连接层
        final_output = self.fc_layers(flattened_output)

        return final_output

# 示例使用
B, C, num_concatenations, H, W = 1, 64, 2, 256, 256

# 创建模型实例
model = YourModel(input_channels=C, num_concatenations=num_concatenations)

# 创建拼接后的向量
concatenated_tensor = torch.rand((B, C, num_concatenations, H, W))

# 将拼接后的向量输入到模型中
outputs = model(concatenated_tensor)

# 输出的形状
print("Model Outputs Shape:", outputs.shape)
