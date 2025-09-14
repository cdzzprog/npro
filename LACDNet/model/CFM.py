# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CFM(nn.Module):
#     def __init__(self, C):
#         super(CFM, self).__init__()
        
#         # 第一注意力模块：通道注意力
#         self.conv1 = nn.Conv2d(2 * C, C, kernel_size=1)
#         self.mlp1 = nn.Sequential(
#             nn.Linear(C, C),
#             nn.ReLU(),
#             nn.Linear(C, C)
#         )
#         self.mlp2 = nn.Sequential(
#             nn.Linear(C, C),
#             nn.ReLU(),
#             nn.Linear(C, C)
#         )
        
#         # 第二注意力模块：空间注意力
#         self.conv2 = nn.Conv2d(2 * C, C, kernel_size=1)
#         self.conv3 = nn.Conv2d(C, C, kernel_size=1)
#         self.conv4 = nn.Conv2d(C, C, kernel_size=1)

#     def forward(self, FRGB, FSAR):
#         # 步骤1：通道注意力模块
#         cat_features = torch.cat((FRGB, FSAR), dim=1)  # 沿通道维度拼接（C）
#         s = F.adaptive_avg_pool2d(self.conv1(cat_features), (1, 1))  # 全局平均池化
#         s = s.view(s.size(0), -1)  # 将特征向量拉平成2D（batch, C）
        
#         # 将结果输入到两个MLP，分别获取RGB和SAR的通道权重
#         weight_rgb = torch.sigmoid(self.mlp1(s))  # 对RGB应用MLP
#         weight_sar = torch.sigmoid(self.mlp2(s))  # 对SAR应用MLP
        
#         F_r_s = FRGB * weight_rgb.view(-1, 1, 1)  # 通道级别的调制
#         F_s_s = FSAR * weight_sar.view(-1, 1, 1)  # 通道级别的调制

#         # 步骤2：空间注意力模块
#         cat_features2 = torch.cat((F_r_s, F_s_s), dim=1)  # 沿通道维度拼接（C）
#         z = self.conv2(cat_features2)  # 通过卷积融合
      
        
#         # 使用softmax得到空间注意力权重
#         softmax_rgb = F.softmax(self.conv3(z), dim=1)  # 对卷积后的结果使用softmax
#         softmax_sar = F.softmax(self.conv4(z), dim=1)
#         # 应用空间注意力权重
#         F_r_z = F_r_s * softmax_rgb
#         F_s_z = F_s_s * softmax_sar
        
#         # 步骤3：最终融合
#         FCFM = F_r_z + F_s_z  # 按像素加法得到最终特征图

#         return FCFM

# # 示例用法：
# C = 64  # 示例通道数
# H = 128  # 示例高度
# W = 128  # 示例宽度

# # 创建示例RGB和SAR特征图
# FRGB = torch.randn(1, C, H, W)  # 示例RGB特征图
# FSAR = torch.randn(1, C, H, W)  # 示例SAR特征图

# # 初始化并运行CFM模块
# cfm = CFM(C)
# output = cfm(FRGB, FSAR)

# print(output.shape)  # 输出的形状应该是[1, C, H, W]

import torch
import torch.nn as nn
import torch.nn.functional as F

class CFM(nn.Module):
    def __init__(self, C_rgb, C_sar):
        super(CFM, self).__init__()
        
        # 第一注意力模块：通道注意力
        self.conv1 = nn.Conv2d(C_rgb + C_sar, C_rgb, kernel_size=1)  # RGB + SAR
        self.mlp1 = nn.Sequential(
            nn.Linear(C_rgb, C_rgb),  # MLP for RGB
            nn.ReLU(),
            nn.Linear(C_rgb, C_rgb)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(C_sar, C_sar),  # MLP for SAR
            nn.ReLU(),
            nn.Linear(C_sar, C_sar)
        )
        
        # 第二注意力模块：空间注意力
        self.conv2 = nn.Conv2d(C_rgb + C_sar, C_rgb, kernel_size=1)  # 融合后的通道数
        self.conv3 = nn.Conv2d(C_rgb, C_rgb, kernel_size=1)
        self.conv4 = nn.Conv2d(C_sar, C_sar, kernel_size=1)

    def forward(self, FRGB, FSAR):
        # 步骤1：通道注意力模块
        # 分别对RGB和SAR进行处理
        s_rgb = F.adaptive_avg_pool2d(FRGB, (1, 1))  # 对RGB进行全局平均池化
        s_sar = F.adaptive_avg_pool2d(FSAR, (1, 1))  # 对SAR进行全局平均池化

        # 拉平成(batch_size, C)形式
        s_rgb = s_rgb.view(s_rgb.size(0), -1)
        s_sar = s_sar.view(s_sar.size(0), -1)

        # 对RGB通道和SAR通道应用各自的MLP
        weight_rgb = torch.sigmoid(self.mlp1(s_rgb))  # 对RGB应用MLP
        weight_sar = torch.sigmoid(self.mlp2(s_sar))  # 对SAR应用MLP
        
        # 确保通过view操作将输出转化为相应的形状
        weight_rgb = weight_rgb.view(-1, 1, 1)  # 变为(batch_size, 1, 1)
        weight_sar = weight_sar.view(-1, 1, 1)  # 变为(batch_size, 1, 1)
        
        # 对输入特征图进行通道级别的加权调制
        F_r_s = FRGB * weight_rgb  # 对RGB进行调制
        F_s_s = FSAR * weight_sar  # 对SAR进行调制

        # 步骤2：空间注意力模块
        cat_features2 = torch.cat((F_r_s, F_s_s), dim=1)  # 沿通道维度拼接（C_rgb + C_sar）
        z = self.conv2(cat_features2)  # 通过卷积融合
      
        # 使用softmax得到空间注意力权重
        softmax_rgb = F.softmax(self.conv3(z), dim=1)  # 对卷积后的结果使用softmax
        softmax_sar = F.softmax(self.conv4(z), dim=1)
        
        # 应用空间注意力权重
        F_r_z = F_r_s * softmax_rgb
        F_s_z = F_s_s * softmax_sar
        
        # 步骤3：最终融合
        FCFM = F_r_z + F_s_z  # 按像素加法得到最终特征图

        return FCFM

# 示例用法：
C_rgb = 3  # RGB通道数
C_sar = 1  # SAR通道数
H = 128  # 示例高度
W = 128  # 示例宽度

# 创建示例RGB和SAR特征图
FRGB = torch.randn(1, C_rgb, H, W)  # 示例RGB特征图
FSAR = torch.randn(1, C_sar, H, W)  # 示例SAR特征图

# 初始化并运行CFM模块
cfm = CFM(C_rgb, C_sar)
output = cfm(FRGB, FSAR)

print(output.shape)  # 输出的形状应该是[1, C_rgb, H, W]，即[1, 3, H, W]
