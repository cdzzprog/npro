import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import VOCSegmentation
from dataset import LandslideDataset

# 1. 定义数据预处理和数据加载
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = VOCSegmentation(root='data', year='2012', image_set='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

# 2. 定义模型
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.conv1x1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv1x1(x)
        x = self.upsample(x)
        return x

num_classes = 21
model = FCN(num_classes=num_classes).cuda()  # 如果使用GPU，调用`.cuda()`

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 保存训练好的模型
torch.save(model.state_dict(), 'fcn_model.pth')