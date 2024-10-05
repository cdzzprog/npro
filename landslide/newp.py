import torch.nn as nn
from osgeo import gdal
import numpy as np
import torch
import cv2
import os
from unet import UNet

def predict_and_save(image_file, model, output_dir, threshold=0.8):
    # 加载和预处理图像
    rsdataset = gdal.Open(image_file)
    image_data = np.stack([rsdataset.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0)
    test_images = torch.tensor(image_data).float().unsqueeze(0)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(test_images)
    
    # 将预测结果二值化
    predicted_mask = (outputs > threshold).float().squeeze().numpy()
    
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
    
    # 保存结果图像
    base_name = os.path.basename(image_file)
    result_file = os.path.join(output_dir, f"predicted_{base_name}")
    cv2.imwrite(result_file, result_image)
    print(f"Saved prediction for {image_file} to {result_file}")

def batch_predict_and_save(image_dir, output_dir, model, threshold=0.8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.png'):  # 根据实际情况过滤文件类型
            full_image_path = os.path.join(image_dir, image_file)
            predict_and_save(full_image_path, model, output_dir, threshold)

# 初始化模型
model = UNet(3, 1)
model.load_state_dict(torch.load('E:\\repository\\weights\\models_building_500.pth'))
model.eval()

# 图像输入目录和预测输出目录
image_dir = 'E:\\数据集\\landslide4sense2022\\train\\images'
output_dir = 'predictions2'

# 批量预测和保存
batch_predict_and_save(image_dir, output_dir, model, threshold=0.8)