import torch
import torch.nn as nn

from .base import BaseModel
import torch.nn.functional as F
from .metrics import AngleError,AngleRecall_rotation,AngleError_rotation
import math
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# from skimage.metrics import structural_similarity as ssim
class RotationSelector(BaseModel):
    default_conf = {
        "matching_dim": "???",
        "output_dim": None,
        "num_classes": 64,
        "backbone": "???",
        "unary_prior": False,
    }
    def _init(self, conf):
        # 在这里进行模型的初始化，根据配置设置模型的各种参数和层
        self.num_classes = conf.num_classes
        # 创建模型的卷积部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=conf.matching_dim, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # 创建模型的 ResNet 块
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(1024, 1024),
            ResNetBlock(1024, 1024),
            ResNetBlock(1024, 1024),
            ResNetBlock(1024, 1024),
            ResNetBlock(1024, 1024),
            ResNetBlock(1024, 1024)
        ])

        # 创建最后的全连接层
        self.fc = nn.Linear(1024, conf.num_classes)#最后一个维度输出预测的yaw角度

    def _forward(self, data):
        pred={}
        radar = data["image"]
        radar= radar.unsqueeze(1)#[1,1,256,256]
        target_pil = ToPILImage()(radar[0].cpu())
        # target_pil.save("target_image11.png")
        map = data["map_viz"]
        map = torch.mean(map.float(), dim=1, keepdim=True)
        map[map == 1] = 0
        map[map == 255] = 1#[1,1,256,256]
        target_pil = ToPILImage()(map[0].cpu())
        # target_pil.save("target_image1.png")
        # 获取输入数据
        #radar 256*256
        #map [1，3，256，256】
        stacked_tensors = contact_tensor(radar,map,self.num_classes)#[1,64,2,256,256]
        
        #拼接N次
        x = stacked_tensors
        # x = torch.cat([radar, map], dim=1)
        # 通过卷积部分
        x = self.conv_layers(x)

        # 通过 ResNet 块
        for block in self.resnet_blocks:
            x = block(x)

        # 对 H、W、C 进行求和，得到向量
        x = x.sum(dim=[2, 3])
    
        # 通过全连接层
        x = self.fc(x)
        
        # # # 应用 softmax
        x = nn.functional.softmax(x, dim=1)
        # 分离 x 和 yaw
        # print(x)

        return {
            **pred,
            "x": x,
        }
    
    def loss(self, pred, data):

        # 获取目标图像（地面实际旋转图像）
        target = data["rotated_image"]#【1，256，256】二值化过了
        target= target.unsqueeze(1)#【1，1, 256，256】
        # target_gray = torch.mean(target.float(), dim=-1, keepdim=False)
        # target_resized = F.interpolate(target_gray.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)
        # target_resized = target_resized.squeeze(1)
        orign = data["image"]
        orign= orign.unsqueeze(1)
        # 保存目标图像
        target_pil = ToPILImage()(target[0].cpu())
        target_pil.save("target_image1.png")
        # 保存原始图像
        origin_pil = ToPILImage()(orign[0].cpu())
        origin_pil.save("origin_image1.png")
        x = pred["x"]
        x = x
        # 获取旋转的图像集合
        rotated_images_set = rotate_images(orign, self.num_classes)#[B,N,C,H,W]【1，64，1，256，256】
        # print(rotated_images_set.shape)
        batch_size  = rotated_images_set.shape[0]
        # 对图像集合进行加权
        # weighted_images = torch.einsum("bncwh,bn->bwh",rotated_images, x)#【1，1，256，256】
        weighted_images = (rotated_images_set * x.view(batch_size, self.num_classes, 1, 1, 1)).sum(dim=1)
        weigthed_pil = ToPILImage()(weighted_images[0].cpu())
        weigthed_pil.save("weighted_image1.png")
        target_label = data["label"]
        # print(target_label)
        criterion = nn.CrossEntropyLoss()
        loss_cross= criterion(x, target_label)
        # 计算加权得到的预测图像和目标图像之间的 L1 损失
        loss = F.l1_loss(weighted_images, target, reduction='mean')
        # loss = loss_cross
        # # lossL1 = lossL1.to("cuda:1")
        # loss = loss_cross + lossL1
        loss = {"total": loss, "nll": loss}
        # 返回总损失
        return loss
    def metrics(self):
        return {
            "yaw_error": AngleError_rotation("x"),
            "yaw_recall_2°": AngleRecall_rotation(2.0, "x"),
            "yaw_recall_5°": AngleRecall_rotation(5.0, "x"),
        }
    


# 定义 ResNet 块
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += identity  # 残差连接
        x = self.relu(x)
        return x

# 定义一个函数，用于旋转图像集合
def rotate_images(images, num_rotations):
    rotated_images = []
    for i in range(num_rotations):
        # 计算旋转的角度 (度)
        angle = i * (360 / num_rotations)
        rotated_image = rotate_tensor(images, angle)
        rotated_images.append(rotated_image)
    return torch.stack(rotated_images, dim=1)


def rotate_tensor(input_tensor, angle_degrees):
    """
    旋转图像张量

    参数：
    - input_tensor: 输入的四维张量 [批次大小, 通道数, 高度, 宽度]
    - angle_degrees: 旋转角度（以度为单位）

    返回：
    旋转后的张量
    """
    # 将角度转换为弧度
    angle_radians = torch.deg2rad(torch.tensor(angle_degrees))

    # 逐通道旋转每个图像
    rotated_tensor = torch.stack([TF.rotate(input_tensor[i], angle_degrees) for i in range(input_tensor.size(0))])
    # rotated_pil = ToPILImage()(rotated_tensor[0].cpu())
    # rotated_pil.save("rotated_image.png")
    return rotated_tensor

def contact_tensor(radar, map,num_rotations):
    concatenated_tensors = []
    for i in range(num_rotations):
        # 计算旋转的角度 (度)
        angle = i * (360 / num_rotations)
        rotated_image = rotate_tensor(radar, angle)
        rotated_image = rotated_image + map 
        concatenated_tensors.append(rotated_image)
    # 将列表中的所有张量堆叠起来，形成一个新的张量，维度为(N, C, H, W)
    stacked_tensors = torch.stack(concatenated_tensors, dim=1)
    stacked_tensors = stacked_tensors.squeeze(dim=2)
    return stacked_tensors