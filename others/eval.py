import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unet import UNet  # 假设UNet模型定义在unet.py中
import data  # 你的数据预处理脚本

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 加载和预处理数据
image_dir = 'E:\数据集\山体滑坡数据集\landslide\image/'
label_dir = 'E:\数据集\山体滑坡数据集\landslide\mask/'

augmentation_pipeline = data.get_augmentation_pipeline()
images, labels = data.load_and_preprocess_data(image_dir, label_dir, augmentation_pipeline)

# 创建数据加载器
dataset = TensorDataset(torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=4,num_workers=0, shuffle=True) # 根据需要调整批量大小

# 定义UNet模型并将其移动到GPU
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')
torch.save(model.state_dict(), 'models_building_51.pth')

print('Training completed.')

#  import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from unet import UNet  # 假设UNet模型定义在unet.py中
# import data  # 你的数据预处理脚本

# # 加载和预处理数据
# image_dir = 'E:\数据集\山体滑坡数据集\landslide\image/'
# label_dir = 'E:\数据集\山体滑坡数据集\landslide\mask/'

# augmentation_pipeline = data.get_augmentation_pipeline()
# images, labels = data.load_and_preprocess_data(image_dir, label_dir, augmentation_pipeline)

# # 创建数据加载器
# dataset = TensorDataset(torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # 根据需要调整批量大小

# # 定义UNet模型
# model = UNet(in_channels=3, out_channels=1)  # 这里的输入通道和输出通道数需要根据实际情况设置
# criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失
# optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam优化器

# # 训练循环
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in dataloader:
#         optimizer.zero_grad()
        
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
    
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

# print('Training completed.')