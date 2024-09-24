import torch.nn as nn
import torch
from osgeo import gdal
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UNet(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(input_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.center = self.conv_block(256, 512)
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        center = self.center(self.pool(enc4))

        dec4 = self.dec4(torch.cat([enc4, self.crop_and_concat(self.up(center), enc4)], 1))
        dec3 = self.dec3(torch.cat([enc3, self.crop_and_concat(self.up(dec4), enc3)], 1))
        dec2 = self.dec2(torch.cat([enc2, self.crop_and_concat(self.up(dec3), enc2)], 1))
        dec1 = self.dec1(torch.cat([enc1, self.crop_and_concat(self.up(dec2), enc1)], 1))
        final = self.final(dec1)

        return torch.sigmoid(final).squeeze(1)
    def crop_and_concat(self, upsampled, bypass):
        # 计算要裁剪的边界
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]

        # 裁剪输入张量
        upsampled = torch.nn.functional.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        return upsampled
        


# 创建模型实例
input_channels = 3  # 输入通道，例如 RGB 图像
output_channels = 1  # 输出通道，例如单通道分割图
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(input_channels, output_channels).to(device)

# 创建随机输入数据 (例如：batch_size=1, height=256, width=256)
input_data = torch.randn(1,input_channels ,256, 256).to(device)

# 将输入数据传入模型并获取输出
output = model(input_data)

# 输出结果的形状
print(output.shape)