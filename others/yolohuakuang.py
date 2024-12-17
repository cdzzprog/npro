# import cv2

# # 图像路径
# image_path = r'E:\papers\model\Data_Trans-main\VOC_To_YOLO\JPEGImages_filtered\00010.jpg'

# # 标注文件路径 (YOLO格式)
# txt_path = r'E:\papers\model\Data_Trans-main\VOC_To_YOLO\JPEGImages_filtered\00010.txt'

# # 读取图像
# image = cv2.imread(image_path)
# h, w, _ = image.shape  # 获取图像的高度和宽度

# # 打开YOLO格式的txt文件，读取标注信息
# with open(txt_path, 'r') as file:
#     for line in file:
#         # 解析每一行，获取标注的类和边界框的相对坐标
#         class_id, center_x, center_y, box_width, box_height = map(float, line.strip().split())

#         # 将相对坐标转换为绝对坐标
#         xmin = int((center_x - box_width / 2) * w)
#         ymin = int((center_y - box_height / 2) * h)
#         xmax = int((center_x + box_width / 2) * w)
#         ymax = int((center_y + box_height / 2) * h)

#         # 绘制矩形框
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 绿色框，线宽为2

# # 显示带框的图像
# cv2.imshow('Image with bounding boxes', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 可选：保存带框的图像
# cv2.imwrite('annotated_image_1.jpg', image)
import cv2
import os

# 图像和标注文件的目录路径
image_dir = r'E:\dataset1\landslides100'
txt_dir = image_dir  # 假设标注文件与图像文件在同一目录

# 输出结果保存的目录
output_dir = r'E:\dataset1\landslides100\100_label'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，则创建它

# 获取目录中所有图像文件（假设是.jpg文件）
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# 遍历每个图像文件
for image_file in image_files:
    # 获取对应的txt标注文件
    txt_file = image_file.replace('.png', '.txt')
    
    # 图像路径和标注文件路径
    image_path = os.path.join(image_dir, image_file)
    txt_path = os.path.join(txt_dir, txt_file)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}")
        continue

    h, w, _ = image.shape  # 获取图像的高度和宽度

    # 打开YOLO格式的txt文件，读取标注信息
    with open(txt_path, 'r') as file:
        for line in file:
            # 解析每一行，获取标注的类和边界框的相对坐标
            class_id, center_x, center_y, box_width, box_height = map(float, line.strip().split())

            # 将相对坐标转换为绝对坐标
            xmin = int((center_x - box_width / 2) * w)
            ymin = int((center_y - box_height / 2) * h)
            xmax = int((center_x + box_width / 2) * w)
            ymax = int((center_y + box_height / 2) * h)

            # 绘制矩形框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 绿色框，线宽为2

    # 显示带框的图像
    # cv2.imshow('Image with bounding boxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存带框的图像到指定目录
    output_image_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_image_path, image)
    print(f"Annotated image saved at: {output_image_path}")
