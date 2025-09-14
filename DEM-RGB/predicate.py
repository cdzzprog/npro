# import torch.nn as nn
# from osgeo import gdal
# import numpy as np
# import torch
# import cv2
# from model import UNet
# from data import X_tensor 
# from dataset0 import TRAIN_XX, TRAIN_YY

# model = UNet(img_channels=7, output_channels=1)  
# model.load_state_dict(torch.load('E:\\repository\\weights\\model_save.pth'))
# model.eval()
# test_image = X_tensor[15].unsqueeze(0)


# img = 15
# image_data=TRAIN_XX[img, 0:3,:, :] * 255
# outputs = model(test_image)


# predicted_mask = (outputs > 0.8).float().squeeze().numpy()


# predicted_mask = (predicted_mask * 255).astype(np.uint8)


# original_image = np.transpose(image_data, (1, 2, 0))  # 转换为 [H, W, C] 格式
# if original_image.dtype != np.uint8:
#     original_image = (original_image* 255).astype(np.uint8)
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


# import os
# import numpy as np
# import torch
# import cv2
# from model import UNet
# from data import X_tensor
# from dataset0 import TRAIN_XX, TRAIN_YY

# # 设置模型和加载权重
# model = UNet(img_channels=7, output_channels=1)  
# model.load_state_dict(torch.load('E:\\repository\\weights\\model_save.pth'))
# model.eval()

# # 创建保存结果的文件夹
# output_folder = 'E:\\repository\\predictions'
# os.makedirs(output_folder, exist_ok=True)

# # 批量预测
# for img in range(len(X_tensor)):
#     test_image = X_tensor[img].unsqueeze(0)  # 添加批次维度

#     # 获取输入图像数据
#     image_data = TRAIN_XX[img, 0:3, :, :] * 255
#     outputs = model(test_image)

#     predicted_mask = (outputs > 0.8).float().squeeze().numpy()
#     predicted_mask = (predicted_mask * 255).astype(np.uint8)

#     # 转换图像数据格式
#     original_image = np.transpose(image_data, (1, 2, 0))  # 转换为 [H, W, C] 格式
#     if original_image.dtype != np.uint8:
#         original_image = (original_image * 255).astype(np.uint8)
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

#     # 创建一个三通道的彩色掩码
#     colored_mask = np.zeros_like(original_image)
#     colored_mask[:, :, 1] = predicted_mask  # 将掩码设置为红色

#     # 叠加掩码到原始图像上
#     result_image = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)

#     # 保存结果图像
#     output_path = os.path.join(output_folder, f'prediction_{img}.png')
#     cv2.imwrite(output_path, result_image)

# print("所有预测结果已保存！")


import os
import numpy as np
import torch
import cv2
from model import UNet
from data import X_tensor,x_valid
from dataset0 import TRAIN_XX, TRAIN_YY

# 设置模型和加载权重
# model = UNet(img_channels=7, output_channels=1) 
model = UNet(3, 1)  
model.load_state_dict(torch.load('E:\\repository\\weights\\models_building_500.pth'))
model.eval()

# 创建保存结果的文件夹
output_folder = 'E:\\repository\\predictions'
os.makedirs(output_folder, exist_ok=True)


# 批量预测
for img in range(20):
    test_image = X_tensor[img].unsqueeze(0)  # 添加批次维度
    test_image = (test_image - 0.5) / 0.5
    # 获取输入图像数据
    # image_data = TRAIN_XX[img, 0:3, :, :] * 255
    # outputs = model(test_image)
    outputs = torch.sigmoid(model(test_image))

    predicted_mask = (outputs > 0.5).float().squeeze().numpy()
    predicted_mask = (predicted_mask).astype(np.uint8)

    # 保存预测掩码
    mask_output_path = os.path.join(output_folder, f'mask_{img}.png')
    cv2.imwrite(mask_output_path, predicted_mask)
    
    # 转换图像数据格式
    # original_image = np.transpose(image_data, (1, 2, 0))  # 转换为 [H, W, C] 格式
    # if original_image.dtype != np.uint8:
    #     original_image = (original_image * 255).astype(np.uint8)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # # 可选：也可以单独保存原图
    # original_output_path = os.path.join(output_folder, f'original_{img}.png')
    # cv2.imwrite(original_output_path, original_image)

    import matplotlib.pyplot as plt

    plt.imshow(predicted_mask, cmap='gray')  # 以灰度图显示掩码
    plt.show()
    print(np.unique(predicted_mask))  # 输出预测掩码的唯一值

    # print(outputs)
    # print(predicted_mask)

print("所有预测掩码已保存！")
