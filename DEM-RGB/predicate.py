# import torch.nn as nn
# from osgeo import gdal
# import numpy as np
# import torch
# import cv2
# from model import UNet
# from data import train_loader,valid_loader
# from PIL import Image
# import torchvision.transforms as transforms





# model = UNet(img_channels=7, output_channels=1)  
# model.load_state_dict(torch.load('E:\\repository\\model_save.pth'))
# model.eval()


# predictions = []
# with torch.no_grad():  # 禁用梯度计算
#     for images in valid_loader:
#         outputs = model(images[0])  # 获取模型输出
#         predictions.append(outputs)  # 收集输出
        
# predictions = torch.cat(predictions).numpy()








# import torch.nn as nn
# import numpy as np
# import torch
# from model import UNet
# from data import train_loader, valid_loader
# from PIL import Image
# import os

# # 加载模型
# model = UNet(img_channels=7, output_channels=1)
# model.load_state_dict(torch.load('E:\\repository\\model_save.pth'))
# model.eval()

# predictions = []
# with torch.no_grad():  # 禁用梯度计算
#     for images in valid_loader:
#         outputs = model(images[0])  # 获取模型输出
#         predictions.append(outputs)  # 收集输出

# # 将预测结果合并为一个张量
# predictions = torch.cat(predictions).cpu().numpy()

# # 创建输出文件夹
# output_folder = 'output_images'  # 输出文件夹
# os.makedirs(output_folder, exist_ok=True)

# # 保存输出结果为黑白图像
# for i in range(predictions.shape[0]):
#     # 取出每个预测样本
#     output_image = predictions[i, 0]  # 取出第一个通道，假设是单通道输出
    
#     # 打印输出值的范围
#     print(f"输出的最小值和最大值 (样本 {i})：", output_image.min(), output_image.max())
    
#     # 确保输出值在0到1之间
#     output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # 归一化
#     output_image = (output_image > 0.5).astype(np.uint8)  # 二值化处理

#     # 转换为PIL图像，模式为'1'表示黑白图像
#     output_pil_image = Image.fromarray(output_image * 255, mode='1')  # 将0和1映射为黑白

#     # 保存输出图像
#     output_pil_image.save(os.path.join(output_folder, f'output_image_{i}.png'))  # 保存每个输出图像

import torch
import numpy as np
import os
from PIL import Image
from model import UNet
from data import train_loader, valid_loader

# 加载模型
model = UNet(img_channels=7, output_channels=1)
model.load_state_dict(torch.load('E:\\repository\\model_save.pth'))
model.eval()

predictions = []
with torch.no_grad():  # 禁用梯度计算
    for images in valid_loader:
        outputs = model(images[0])  # 获取模型输出
        predictions.append(outputs)  # 收集输出

# 将预测结果合并为一个张量
predictions = torch.cat(predictions).cpu().numpy()

# 创建输出文件夹
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# 保存输出结果为黑白图像
for i in range(predictions.shape[0]):
    output_image = predictions[i, 0]  # 取出第一个通道

    # 打印输出值的范围
    min_val, max_val = output_image.min(), output_image.max()
    print(f"输出的最小值和最大值 (样本 {i})：", min_val, max_val)

    # 归一化处理，将范围调整到0到1之间
    output_image = (output_image - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(output_image)

    # 二值化处理
    output_image = (output_image > 0.5).astype(np.uint8)  # 调整阈值为0.5

    # 转换为PIL图像，模式为'1'表示黑白图像
    output_pil_image = Image.fromarray(output_image * 255, mode='1')

    # 保存输出图像
    output_pil_image.save(os.path.join(output_folder, f'output_image_{i}.png'))

  # 合并批次输

# image_file='E:\数据集\山体滑坡数据集\landslide\image\js021.png'
# rsdataset = gdal.Open(image_file)
# image_data = np.stack([rsdataset.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0)








# # 将图像数据转换为 PyTorch 张量，并增加 batch 维度
# test_images = torch.tensor(image_data).float().unsqueeze(0)

# # 模型预测
# outputs = model(test_images)

# # 将预测结果二值化
# predicted_mask = (outputs > 0.8).float().squeeze().numpy()

# # 将预测结果转换为 8 位掩码图像（0-255）
# predicted_mask = (predicted_mask * 255).astype(np.uint8)

# # 转换原始图像为 8 位三通道图像
# original_image = np.transpose(image_data, (1, 2, 0))  # 转换为 [H, W, C] 格式
# original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

# # 创建一个三通道的彩色掩码，红色通道显示预测掩码
# colored_mask = np.zeros_like(original_image)
# colored_mask[:, :, 2] = predicted_mask  # 将掩码设置为红色

# # 将彩色掩码叠加到原始图像上
# result_image = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)

# # 显示带有预测结果的图像
# cv2.imshow('Prediction on Original Image', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 如果你想保存带有预测结果的图像，可以使用以下代码：
# # cv2.imwrite('output_with_prediction.png', result_image)