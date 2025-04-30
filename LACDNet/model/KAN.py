import torch
import torch.nn as nn

# 定义生成器（G）和判别器（D）
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128 * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x.view(x.size(0), 128, 8, 8)  # 变为适合卷积的形状
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # 输出图像 [batch_size, 3, 64, 64]
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1)
    
    def forward(self, x):
        print(x.shape)  # 打印输入的形状，检查是否是 [batch_size, 3, 64, 64]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 定义变化检测网络
class ChangeDetectionNetwork(nn.Module):
    def __init__(self):
        super(ChangeDetectionNetwork, self).__init__()
        self.generator = Generator(input_dim=3, output_dim=3)  # 假设输入为RGB图像
        self.discriminator = Discriminator(input_dim=3)

    def forward(self, source_image, target_image):
        # 域适应部分
        generated_image = self.generator(source_image)
        
        # 判别器判断生成图像和目标图像是否属于目标域
        discriminator_output = self.discriminator(generated_image)
        
        return generated_image, discriminator_output
