import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as transforms
import torch

# 数据集目录
image_dir = 'E:\数据集\山体滑坡数据集\landslide\image/'
label_dir = 'E:\数据集\山体滑坡数据集\landslide\mask/'

# 设定图像尺寸
target_size = (256, 256)  # 根据需要设置目标尺寸

# 定义数据增强操作
def get_augmentation_pipeline():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

def preprocess_image(image_path, transform=None):
    """加载、调整尺寸和归一化图像"""
    img = Image.open(image_path).convert('RGB')  # 转换为RGB模式
    img = img.resize(target_size)  # 调整尺寸
    
    if transform:
        img = transform(img)
    
    return img

def preprocess_label(label_path, transform=None):
    """加载、调整尺寸和处理标注图像"""
    label = Image.open(label_path).convert('L')  # 转换为灰度模式
    label = label.resize(target_size)  # 调整尺寸
    label = np.array(label)
    
    # 将滑坡区域标记为1，其余为0
    label_binary = (label > 0).astype(np.int32)
    
    if transform:
        # 将numpy数组转换为PIL图像，然后应用转换
        label_image = Image.fromarray(label_binary.astype(np.uint8) * 255)  # 将二进制标签转为图像
        label_image = transform(label_image)
        label_binary = np.array(label_image) / 255.0  # 归一化
    
    return label_binary

def load_and_preprocess_data(image_dir, label_dir, augmentations):
    """加载和预处理所有图像和标注图像"""
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    images = []
    labels = []

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        # 定义增强操作
        transform = augmentations

        img = preprocess_image(image_path, transform)
        label = preprocess_label(label_path, transform)

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# 创建数据增强流水线
augmentation_pipeline = get_augmentation_pipeline()

# 使用函数加载和预处理数据
images, labels = load_and_preprocess_data(image_dir, label_dir, augmentation_pipeline)

print(f'Loaded and preprocessed {len(images)} images and {len(labels)} labels with augmentation.')
print(f'Image shape: {images[0].shape}')
print(f'Label shape: {labels[0].shape}')