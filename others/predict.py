import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unet import UNet  # 假设UNet模型定义在unet.py中
import data  # 你的数据预处理脚本
# 重新加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('models_building_51.pth'))
model.eval()

# 预测函数
def predict_image(image_path, model, device, transform=None):
    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)  # 添加批量维度并移动到设备

    # 进行预测
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output).cpu().numpy().squeeze()  # 去掉批量维度并转到CPU

    return output

# 定义预处理变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 使用预测函数
image_path = 'E:\数据集\山体滑坡数据集\landslide\image2\df022.png'
prediction = predict_image(image_path, model, device, transform)

# 保存或展示预测结果
predicted_image = Image.fromarray((prediction * 255).astype(np.uint8))  # 转换为图像格式
predicted_image.save('prediction_result.png')
predicted_image.show()