import cv2

# 读取三通道图像（例如 RGB）
image = cv2.imread(r'E:\data001\test\label\insar.png')

# 将三通道图像转换为单通道灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 保存转换后的单通道图像
cv2.imwrite('insar.png', gray_image)
# import cv2
# import os

# # 图像文件夹路径
# image_folder = r'E:\papers\zip\AEKAN-for-MCD-main\data'

# # 遍历文件夹中的所有文件
# for filename in os.listdir(image_folder):
#     # 只处理图片文件（例如 .png, .jpg 等）
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(image_folder, filename)
        
#         # 读取图像
#         image = cv2.imread(image_path)

#         # 检查图像是否正确读取
#         if image is not None:
#             # 将三通道图像转换为单通道灰度图像
#             gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
#             # 保存转换后的单通道图像，文件名加上 "_gray" 后缀
#             new_image_path = os.path.join(image_folder, f"{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}")
#             cv2.imwrite(new_image_path, gray_image)
#             print(f"Converted {filename} to grayscale and saved as {new_image_path}")
#         else:
#             print(f"Failed to read {filename}, skipping.")
