# import torch.nn as nn
# from osgeo import gdal
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import cv2
# class UNet(nn.Module):
#     def __init__(self, input_channels, out_channels):
#         super(UNet, self).__init__()

#         self.enc1 = self.conv_block(input_channels, 32)
#         self.enc2 = self.conv_block(32, 64)
#         self.enc3 = self.conv_block(64, 128)
#         self.enc4 = self.conv_block(128, 256)
#         self.center = self.conv_block(256, 512)
#         self.dec4 = self.conv_block(512 + 256, 256)
#         self.dec3 = self.conv_block(256 + 128, 128)
#         self.dec2 = self.conv_block(128 + 64, 64)
#         self.dec1 = self.conv_block(64 + 32, 32)
#         self.final = nn.Conv2d(32, out_channels, kernel_size=1)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_channels)
#         )

#     def forward(self, x):
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))
#         enc4 = self.enc4(self.pool(enc3))

#         center = self.center(self.pool(enc4))

#         dec4 = self.dec4(torch.cat([enc4, self.crop_and_concat(self.up(center), enc4)], 1))
#         dec3 = self.dec3(torch.cat([enc3, self.crop_and_concat(self.up(dec4), enc3)], 1))
#         dec2 = self.dec2(torch.cat([enc2, self.crop_and_concat(self.up(dec3), enc2)], 1))
#         dec1 = self.dec1(torch.cat([enc1, self.crop_and_concat(self.up(dec2), enc1)], 1))
#         final = self.final(dec1).squeeze()

#         return torch.sigmoid(final)
#     def crop_and_concat(self, upsampled, bypass):
#         # 计算要裁剪的边界
#         diffY = bypass.size()[2] - upsampled.size()[2]
#         diffX = bypass.size()[3] - upsampled.size()[3]

#         # 裁剪输入张量
#         upsampled = torch.nn.functional.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

#         return upsampled

# model = UNet(3, 1)
# model.load_state_dict(torch.load('E:\PythonProject\modelearning\models_building_52.pth'))
# model.eval()

# image_file='E:\数据集\山体滑坡数据集\landslide\image\js023.png'
# rsdataset = gdal.Open(image_file)
# images=(np.stack([rsdataset.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0))
# test_images = torch.tensor(images).float().unsqueeze(0)

# outputs = model(test_images)
# outputs = (outputs > 0.8).float()

# cv2.imshow('Prediction', outputs.numpy())
# cv2.waitKey(0)
# # from matplotlib import pyplot as plt

# # plt.imshow(outputs.numpy())
# # plt.title('Prediction')
# # plt.show()


import torch.nn as nn
from osgeo import gdal
import numpy as np
import torch
import cv2
# from unet import UNet
from model import UNet
# model = UNet(3, 1)
# model.load_state_dict(torch.load('E:\\repository\\weights\\models_building_500.pth'))
# model.eval()
model = UNet(img_channels=7, output_channels=1)  
model.load_state_dict(torch.load('E:\\repository\\weights\\model_save.pth'))
model.eval()

image_file='E:\数据集\山体滑坡数据集\landslide\image\js021.png'
rsdataset = gdal.Open(image_file)
image_data = np.stack([rsdataset.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0)

# 将图像数据转换为 PyTorch 张量，并增加 batch 维度
test_images = torch.tensor(image_data).float().unsqueeze(0)

# 模型预测
outputs = model(test_images)

# 将预测结果二值化
predicted_mask = (outputs > 0.7).float().squeeze().numpy()

# 将预测结果转换为 8 位掩码图像（0-255）
predicted_mask = (predicted_mask * 255).astype(np.uint8)

# 转换原始图像为 8 位三通道图像
original_image = np.transpose(image_data, (1, 2, 0))  # 转换为 [H, W, C] 格式
original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

# 创建一个三通道的彩色掩码，红色通道显示预测掩码
colored_mask = np.zeros_like(original_image)
colored_mask[:, :, 2] = predicted_mask  # 将掩码设置为红色

# 将彩色掩码叠加到原始图像上
result_image = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)

# 显示带有预测结果的图像
cv2.imshow('Prediction on Original Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(predicted_mask)
# 如果你想保存带有预测结果的图像，可以使用以下代码：
# cv2.imwrite('output_with_prediction.png', result_image)