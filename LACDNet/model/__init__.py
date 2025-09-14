import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义全局最大池化层
class GlobalMaxPooling(nn.Module):
    def forward(self, x):
        return F.adaptive_max_pool2d(x, (1, 1))

# 定义精炼学习模块
class RefinementLearning(nn.Module):
    def __init__(self, in_channels):
        super(RefinementLearning, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x

# 定义特征图校正模块
class FeatureCorrection(nn.Module):
    def __init__(self):
        super(FeatureCorrection, self).__init__()

    def forward(self, feature_map, refined_weights):
        return feature_map * torch.sigmoid(refined_weights)

# 主模型，包含RGB和SAR差异特征计算以及后续的精炼学习与校正
class FusionNetwork(nn.Module):
    def __init__(self, channels):
        super(FusionNetwork, self).__init__()
        self.global_max_pooling = GlobalMaxPooling()
        self.refinement_rgb = RefinementLearning(channels)
        self.refinement_sar = RefinementLearning(channels)
        self.feature_correction = FeatureCorrection()

    def forward(self, F_rgb, F_sar):
        # Step 1: 计算差异特征
        F_r = F_rgb - F_sar
        F_s = F_sar - F_rgb
        
        # Step 2: 计算全局最大池化
        s_r1 = self.global_max_pooling(F_r)
        s_s1 = self.global_max_pooling(F_s)
        
        # Step 2: 使用卷积层进行精炼学习
        s_r2 = self.refinement_rgb(s_r1)
        s_s2 = self.refinement_sar(s_s1)
        
        # Step 3: 校正特征图
        F_r_sfm = self.feature_correction(F_rgb, s_r2)
        F_s_sfm = self.feature_correction(F_sar, s_s2)
        
        return F_r_sfm, F_s_sfm

# 模型初始化与测试
if __name__ == "__main__":
    # 假设输入的RGB和SAR特征图尺寸为 (batch_size, channels, height, width)
    batch_size = 4
    channels = 64
    height, width = 128, 128

    # 创建示例输入
    F_rgb = torch.randn(batch_size, channels, height, width)
    F_sar = torch.randn(batch_size, channels, height, width)

    # 初始化模型
    model = FusionNetwork(channels=channels)

    # 前向传播
    F_r_sfm, F_s_sfm = model(F_rgb, F_sar)

    # 输出结果的形状
    print("F_r_sfm shape:", F_r_sfm.shape)
    print("F_s_sfm shape:", F_s_sfm.shape)
