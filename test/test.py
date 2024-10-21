# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取低分辨率图像
# low_res_image = cv2.imread(r'E:\dataset1\bijie\landlside\dem\df022.png')  # 替换为你的低分辨率图像路径
# low_res_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)  # 转换颜色通道

# # 获取原始图像的大小
# original_size = low_res_image.shape[:2]
# print(f"Original size: {original_size}")

# # 使用 OpenCV 的 resize 函数进行图像插值
# high_res_image = cv2.resize(low_res_image, (1024, 1024), interpolation=cv2.INTER_LINEAR)  # 将图像重建为 512x512

# # 显示结果
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title('Low Resolution Image')
# plt.imshow(low_res_image)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('High Resolution Image')
# plt.imshow(high_res_image)
# plt.axis('off')

# plt.show()

# # 保存高分辨率图像
# cv2.imwrite('high_res_image.jpg', cv2.cvtColor(high_res_image, cv2.COLOR_RGB2BGR))  # 保存为高分辨率图像
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # 输入和输出文件夹路径
# input_folder = r'E:\dataset1\bijie\landlside\image'
# output_folder = r'E:\dataset1\bijie\landlside\images'

# # 如果输出文件夹不存在，创建它
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 获取所有图片文件名
# image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # 批量处理每张图像
# for image_file in image_files:
#     # 构造图像的完整路径
#     image_path = os.path.join(input_folder, image_file)
    
#     # 读取低分辨率图像
#     low_res_image = cv2.imread(image_path)
#     if low_res_image is None:
#         print(f"Error reading {image_file}")
#         continue  # 如果图像读取失败，跳过此图像
    
#     # 转换颜色通道（OpenCV 默认为 BGR，转换为 RGB）
#     low_res_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)
    
#     # 获取原始图像的大小
#     original_size = low_res_image.shape[:2]
#     print(f"Processing {image_file}, Original size: {original_size}")
    
#     # 使用 OpenCV 的 resize 函数进行图像插值
#     high_res_image = cv2.resize(low_res_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    
#     # 显示结果（你可以选择在此不显示图像，以加快批量处理速度）
#     # plt.figure(figsize=(10, 5))

#     # plt.subplot(1, 2, 1)
#     # plt.title('Low Resolution Image')
#     # plt.imshow(low_res_image)
#     # plt.axis('off')

#     # plt.subplot(1, 2, 2)
#     # plt.title('High Resolution Image')
#     # plt.imshow(high_res_image)
#     # plt.axis('off')

#     # plt.show()

#     # 保存高分辨率图像
#     high_res_image_path = os.path.join(output_folder, f"{image_file}")
#     cv2.imwrite(high_res_image_path, cv2.cvtColor(high_res_image, cv2.COLOR_RGB2BGR))  # 保存为高分辨率图像
#     print(f"Saved high resolution image: {high_res_image_path}")

# print("Batch processing completed.")



######分辨率转化

import cv2
import os

# 输入和输出文件夹路径
input_folder = r'E:\dataset1\bijie\landlside\dems'  # 替换为你的输入文件夹路径
output_folder = r'E:\dataset1\bijie\landlside\demss'  # 输出文件夹路径

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有图片文件名
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 批量处理每张图像
for image_file in image_files:
    # 构造图像的完整路径
    image_path = os.path.join(input_folder, image_file)
    
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading {image_file}")
        continue  # 如果图像读取失败，跳过此图像
    
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 保存灰度图像
    gray_image_path = os.path.join(output_folder, f"{image_file}")
    cv2.imwrite(gray_image_path, gray_image)  # 保存为灰度图像
    print(f"Saved gray image: {gray_image_path}")

print("Batch conversion to grayscale completed.")
