import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class LandslideDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images=os.listdir(image_dir)

        # self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        # self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png') or f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])
        
        image= Image.open(image_path).convert('RGB')  # 转换为RGB模式
        mask = Image.open(label_path).convert('L')  # 转换为灰度模式
        
        if self.transform is not None:
            augmentation=self.transform(iamge=image,mask=mask)
            image=augmentation("image")
            mask=augmentation("mask")
            
        return image,mask

#         img = img.resize(target_size)
#         label = label.resize(target_size)
        
#         # 将灰度标签图像转换为二进制
#         label = np.array(label)
#         label_binary = (label > 0).astype(np.int32)  # 将滑坡区域标记为1，其余为0
        
#         if self.transform:
#             img = self.transform(img)
#             label_image = Image.fromarray(label_binary.astype(np.uint8) * 255)  # 将二进制标签转为图像
#             label_image = self.transform(label_image)
#             label_binary = np.array(label_image) / 255.0  # 归一化

#         return img, torch.tensor(label_binary, dtype=torch.float32)

# # 数据集目录
# image_dir = 'E:/数据集/山体滑坡数据集/landslide/image/'
# label_dir = 'E:/数据集/山体滑坡数据集/landslide/mask/'

# # 设定图像尺寸
# target_size = (256, 256)  # 根据需要设置目标尺寸

# # 创建数据增强流水线
# augmentation_pipeline = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=30),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.Resize(target_size),
#     transforms.ToTensor()
# ])

# # 创建数据集实例
# dataset = LandslideDataset(image_dir=image_dir, label_dir=label_dir, transform=augmentation_pipeline)

# # 创建数据加载器
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# # 打印数据集信息
# print(f'Loaded and preprocessed {len(dataset)} images and labels with augmentation.')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import models

# # 假设你有一个简单的卷积神经网络模型，或者你可以使用预训练模型
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(16 * 128 * 128, 2)  # 假设目标图像尺寸是128x128

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = x.view(-1, 16 * 128 * 128)
#         x = self.fc1(x)
#         return x

# # 定义模型、损失函数和优化器
# model = SimpleCNN()
# criterion = nn.BCEWithLogitsLoss()  # 二分类问题
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 数据集和数据加载器
# dataset = LandslideDataset(image_dir=image_dir, label_dir=label_dir, transform=augmentation_pipeline)
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# # 训练过程
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in data_loader:
#         # 将数据移动到GPU（如果可用）
#         inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
#         # 前向传播
#         outputs = model(inputs)
#         loss = criterion(outputs.squeeze(), labels)
        
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item() * inputs.size(0)
    
#     epoch_loss = running_loss / len(dataset)
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# print('Training complete.')