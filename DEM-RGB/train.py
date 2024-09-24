import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import train_loader,valid_loader
from model import UNet


model = UNet(img_channels=7, output_channels=1)  
model.train()
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 根据任务调整损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)



# 训练过程
num_epochs = 100
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)

    # 验证过程
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in valid_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            val_loss += loss.item()

    val_loss /= len(valid_loader)
    val_loss_history.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), "model_save.pth")

# 计算评价指标
# 这里可以根据任务定义准确率、F1等指标

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
ax1.plot(train_loss_history, label='Train Loss')
ax1.plot(val_loss_history, label='Validation Loss')
ax1.set_title('Loss Over Epochs')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()

# 如果有其他指标可以继续添加绘图代码
plt.show()
