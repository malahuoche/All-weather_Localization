import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

from .base import BaseModel


class Fusion_net(BaseModel):
    default_conf = {
        "in_channels": 128,
    }

    def _init(self, conf):
        # # 空间注意力
        # self.spatial_attention = nn.Sequential(
        #     nn.Conv2d(conf.in_channels * 2, conf.in_channels, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveMaxPool2d(1),
        #     nn.Conv2d(conf.in_channels, conf.in_channels, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        # # 通道注意力
        # self.channel_attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(conf.in_channels * 2, conf.in_channels, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(conf.in_channels, conf.in_channels, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )
        #简化板
        self.conv = nn.Conv2d(conf.in_channels * 2, conf.in_channels, kernel_size=1)

    def _forward(self, data):
        radar_feature = data["f_radar"]
        image_feature = data["f_image"]
        # pred = {}
        # radar_feature = data["f_radar"]
        # image_feature = data["f_image"]
        # spatial_concat = torch.cat([radar_feature, image_feature], dim=1)
        # spatial_attention_map = self.spatial_attention(spatial_concat)
        # spatial_weighted_feature = spatial_attention_map * radar_feature + (1 - spatial_attention_map) * image_feature

        # # 通道注意力融合
        # channel_concat = torch.cat([radar_feature, image_feature], dim=1)
        # channel_attention_map = self.channel_attention(channel_concat)
        # fused_feature = channel_attention_map * spatial_weighted_feature

        # return fused_feature
        #lidar专用
        # radar_feature = radar_feature.unsqueeze(1)
        # print(radar_feature.shape)
        # avg_pooling = nn.AdaptiveAvgPool2d((64, 129))
        # # 使用平均池化对张量进行压缩
        # radar_feature = avg_pooling(radar_feature)
        # print(radar_feature.shape)
        # concatenated_features = torch.cat([radar_feature, image_feature], dim=1)
        # concatenated_features = concatenated_features.to(torch.float32)
        # concatenated_features = concatenated_features.view(1, 128, 64, -1)
        # concatenated_features = concatenated_features.narrow(3, 0, 129)
        # 简化板
        concatenated_features = torch.cat([radar_feature, image_feature], dim=1)
        # 通过1x1卷积层进行融合
       
        fused_feature = self.conv(concatenated_features)
        return fused_feature


#test_demo

# # 假设雷达和图像特征的通道数都是64
# in_channels = 128
# radar_feature = torch.randn(1, in_channels, 64, 129)
# image_feature_size = (128, 128)
# # 调整radar_feature的空间大小
# radar_feature = F.interpolate(radar_feature, size=image_feature_size, mode='bilinear', align_corners=False)
# image_feature = torch.randn(1, in_channels, 128, 128)

# # 创建AttentionFusionModule实例
# attention_fusion_module = Fusion_net(in_channels)

# # 进行融合
# result_feature = attention_fusion_module(radar_feature, image_feature)
# print(result_feature.shape())
    


#最简单的拼接
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FusionModule, self).__init__()
        # 1x1卷积层，用于拼接和融合雷达和图像BEV特征
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, radar_feature, image_feature):
        # 在通道维度上拼接雷达和图像BEV特征
        concatenated_features = torch.cat([radar_feature, image_feature], dim=1)
        # 通过1x1卷积层进行融合
        fused_feature = self.conv(concatenated_features)
        return fused_feature
