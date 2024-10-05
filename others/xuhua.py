from PIL import Image, ImageFilter

# 打开图片
image = Image.open('C:\\Users\\龙儿璨\\Desktop\\images\\虚化\\微信图片_20240923151018.jpg')  # 确保图片格式正确

# 应用模糊效果
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=45))

# 将图片转换为RGB模式
blurred_image = blurred_image.convert('RGB')

# 保存模糊后的图片
blurred_image.save('blurred_image.jpg', format='JPEG')
# import os
# from PIL import Image, ImageFilter

# # 指定图片文件夹路径
# input_folder = 'C:\\Users\\龙儿璨\\Desktop\\images'  # 替换为你的文件夹路径
# output_folder = 'C:\\Users\\龙儿璨\\Desktop\\blurred_images'  # 输出文件夹

# # 如果输出文件夹不存在，则创建
# os.makedirs(output_folder, exist_ok=True)

# # 遍历文件夹中的所有图片
# for filename in os.listdir(input_folder):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):  # 只处理这些格式
#         # 打开图片
#         image_path = os.path.join(input_folder, filename)
#         image = Image.open(image_path)

#         # 应用模糊效果
#         blurred_image = image.filter(ImageFilter.GaussianBlur(radius=65))

#         # 将图片转换为RGB模式
#         blurred_image = blurred_image.convert('RGB')

#         # 保存模糊后的图片
#         output_path = os.path.join(output_folder, f'blurred_{filename}')
#         blurred_image.save(output_path, format='JPEG')

# print("所有图片处理完成！")


# import os
# from PIL import Image, ImageFilter

# # 指定图片文件夹路径
# input_folder = 'C:\\Users\\龙儿璨\\Desktop\\images\虚化'  # 替换为你的文件夹路径
# output_folder = 'C:\\Users\\龙儿璨\\Desktop\\transparent_images1'  # 输出文件夹

# # 如果输出文件夹不存在，则创建
# os.makedirs(output_folder, exist_ok=True)

# # 遍历文件夹中的所有图片
# for filename in os.listdir(input_folder):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):  # 只处理这些格式
#         # 打开图片
#         image_path = os.path.join(input_folder, filename)
#         image = Image.open(image_path).convert('RGBA')

#         # 应用模糊效果
#         blurred_image = image.filter(ImageFilter.GaussianBlur(radius=40))

#         # 创建透明背景
#         datas = blurred_image.getdata()
#         new_data = []
#         for item in datas:
#             # 将白色背景转换为透明
#             if item[0] > 200 and item[1] > 200 and item[2] > 200:
#                 new_data.append((255, 255, 255, 0))  # 变为透明
#             else:
#                 new_data.append(item)

#         blurred_image.putdata(new_data)

#         # 保存带透明背景的图片
#         output_path = os.path.join(output_folder, f'transparent_{filename}.png')
#         blurred_image.save(output_path, format='PNG')

# print("所有图片处理完成！")
