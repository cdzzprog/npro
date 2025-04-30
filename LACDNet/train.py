import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model.KAN import ChangeDetectionNetwork
def train_model():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集（这里假设有source_train_loader和target_train_loader）
    source_train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor()])), batch_size=2, shuffle=True)
    target_train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor()])), batch_size=2, shuffle=True)
    

    # 初始化模型
    model = ChangeDetectionNetwork().to(device)
    
    # 损失函数
    criterion = nn.BCELoss()
    
    # 优化器
    optimizer_g = optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 训练
    for epoch in range(10):
        for i, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, _ = source_data
            target_image, _ = target_data
           
            source_image, target_image = source_image.to(device), target_image.to(device)
            print(source_image.shape)
            # 训练生成器和判别器
            optimizer_g.zero_grad()
            generated_image, disc_output = model(source_image, target_image)
            g_loss = criterion(disc_output, torch.ones_like(disc_output))  # 生成器目标是让判别器认为生成的图像来自目标域
            g_loss.backward()
            optimizer_g.step()
            
            optimizer_d.zero_grad()
            real_output = model.discriminator(target_image)
            d_loss_real = criterion(real_output, torch.ones_like(real_output))  # 判别器目标是识别真实图像
            fake_output = model.discriminator(generated_image.detach())
            d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))  # 判别器目标是识别生成的图像
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_d.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{10}], Step [{i}/{len(source_train_loader)}], "
                      f"Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}")
        
        # 可视化每个epoch生成的图像
        if (epoch + 1) % 5 == 0:
            plt.imshow(generated_image[0].cpu().detach().numpy().transpose(1, 2, 0))
            plt.show()

if __name__ == "__main__":
    train_model()