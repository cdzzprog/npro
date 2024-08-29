import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unet import UNet  # 假设UNet模型定义在unet.py中
import data  # 你的数据预处理脚本

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载测试数据
test_image_dir = 'E:\数据集\山体滑坡数据集\landslide\image2/'
test_label_dir = 'E:\数据集\山体滑坡数据集\landslide\mask2/'

# 使用与训练时相同的数据预处理和增强方法
augmentation_pipeline = data.get_augmentation_pipeline()
test_images, test_labels = data.load_and_preprocess_data(test_image_dir, test_label_dir, augmentation_pipeline)

# 创建测试数据加载器
test_dataset = TensorDataset(torch.tensor(test_images, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))
test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=0, shuffle=False)

# 加载训练好的模型
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('models_building_51.pth', weights_only=True))
model.eval()

# 计算测试精度
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        predictions = torch.sigmoid(outputs) > 0.5  # 二值化处理
        labels = labels > 0.5  # 如果标签是0或1，你可以使用这个来做二值化
        
        # 计算正确的预测数量
        correct += (predictions == labels).sum().item()
        total += labels.numel()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')