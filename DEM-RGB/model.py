import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms.functional
from torch import nn

# class DoubleConvolution(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.act1 = nn.ReLU()
#         self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.act2 = nn.ReLU()

#     def forward(self, x: torch.Tensor):
#         x = self.first(x)
#         x = self.act1(x)
#         x = self.second(x)
#         return self.act2(x)

# class DownSample(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x: torch.Tensor):
#         return self.pool(x)

# class UpSample(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

#     def forward(self, x: torch.Tensor):
#         return self.up(x)

# class CropAndConcat(nn.Module):
#     def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
#         contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
#         x = torch.cat([x, contracting_x], dim=1)
#         return x

# class UNet(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
#                                         [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
#         self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
#         self.middle_conv = DoubleConvolution(512, 1024)
#         self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
#                                         [(1024, 512), (512, 256), (256, 128), (128, 64)]])
#         self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
#                                       [(1024, 512), (512, 256), (256, 128), (128, 64)]])
#         self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
#         self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x: torch.Tensor):
#         pass_through = []
#         for i in range(len(self.down_conv)):
#             x = self.down_conv[i](x)
#             pass_through.append(x)
#             x = self.down_sample[i](x)

#         x = self.middle_conv(x)

#         for i in range(len(self.up_conv)):
#             x = self.up_sample[i](x)
#             x = self.concat[i](x, pass_through.pop())
#             x = self.up_conv[i](x)

#         x = self.final_conv(x)
#         return x

class UNet(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(input_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.center = self.conv_block(256, 512)
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        self.dropout = nn.Dropout(p=0.5)
        enc1 = self.dropout(self.enc1(x))
        enc2 = self.dropout(self.enc2(self.pool(enc1)))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        center = self.center(self.pool(enc4))

        dec4 = self.dec4(torch.cat([enc4, self.crop_and_concat(self.up(center), enc4)], 1))
        dec3 = self.dec3(torch.cat([enc3, self.crop_and_concat(self.up(dec4), enc3)], 1))
        dec2 = self.dec2(torch.cat([enc2, self.crop_and_concat(self.up(dec3), enc2)], 1))
        dec1 = self.dec1(torch.cat([enc1, self.crop_and_concat(self.up(dec2), enc1)], 1))
        final = self.final(dec1).squeeze()

        return final
    def crop_and_concat(self, upsampled, bypass):
        # 计算要裁剪的边界
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]

        # 裁剪输入张量
        upsampled = torch.nn.functional.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        return upsampled









# class UNet(nn.Module):
#     def __init__(self, img_channels, output_channels):
#         super(UNet, self).__init__()
        
#         self.encoder1 = self.contract_block(img_channels, 16)
#         self.encoder2 = self.contract_block(16, 32)
#         self.encoder3 = self.contract_block(32, 64)
#         self.encoder4 = self.contract_block(64, 128)
#         self.bottleneck = self.contract_block(128, 256)
        
#         self.decoder4 = self.expand_block(256 + 128, 128)  # 加上 encoder4 的通道数
#         self.decoder3 = self.expand_block(128 + 64, 64)   # 加上 encoder3 的通道数
#         self.decoder2 = self.expand_block(64 + 32, 32)     # 加上 encoder2 的通道数
#         self.decoder1 = self.expand_block(32 + 16, 16)     # 加上 encoder1 的通道数
        
#         self.final_conv = nn.Conv2d(16, output_channels, kernel_size=(1, 1))

#     def contract_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
    
#     def expand_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(enc1)
#         enc3 = self.encoder3(enc2)
#         enc4 = self.encoder4(enc3)
#         bottleneck = self.bottleneck(enc4)

#         # 打印每一层的输出形状
#         # print(f"enc1 shape: {enc1.shape}")
#         # print(f"enc2 shape: {enc2.shape}")
#         # print(f"enc3 shape: {enc3.shape}")
#         # print(f"enc4 shape: {enc4.shape}")
#         # print(f"bottleneck shape: {bottleneck.shape}")

#         # 上采样到 enc4 的大小
#         bottleneck_up = torch.nn.functional.interpolate(bottleneck, size=(16, 16), mode='bilinear', align_corners=False)
#         dec4 = self.decoder4(torch.cat((bottleneck_up, enc4), dim=1))  # 拼接后传入 decoder4
#         # print(f"dec4 shape: {dec4.shape}")

#         dec3 = self.decoder3(torch.cat((dec4, enc3), dim=1))
#         # print(f"dec3 shape: {dec3.shape}")

#         dec2 = self.decoder2(torch.cat((dec3, enc2), dim=1))
#         # print(f"dec2 shape: {dec2.shape}")

#         dec1 = self.decoder1(torch.cat((dec2, enc1), dim=1))
#         # print(f"dec1 shape: {dec1.shape}")

#         output = self.final_conv(dec1)
   

#         return output
         


# # 测试模型
# if __name__ == "__main__":
# # 创建模型实例
#     input_channels = 7 
#     output_channels = 1  
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = UNet(input_channels, output_channels).to(device)
#     input_data = torch.randn(16,input_channels ,256, 256).to(device)
#     output = model(input_data)
#     print(output.shape)





# import torch
# import torch.nn as nn

# class UNet(nn.Module):
#     def __init__(self, input_channels, out_channels):
#         super(UNet, self).__init__()

#         self.enc1 = self.conv_block(input_channels, 32)
#         self.enc2 = self.conv_block(32, 64)
#         self.enc3 = self.conv_block(64, 128)
#         self.enc4 = self.conv_block(128, 256)
#         self.center = self.conv_block(256, 512)
#         self.dec4 = self.conv_block(512 + 256, 256)
#         self.dec3 = self.conv_block(256 + 128, 128)
#         self.dec2 = self.conv_block(128 + 64, 64)
#         self.dec1 = self.conv_block(64 + 32, 32)
#         self.final = nn.Conv2d(32, out_channels, kernel_size=1)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_channels)
#         )

#     def forward(self, x):
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))
#         enc4 = self.enc4(self.pool(enc3))

#         center = self.center(self.pool(enc4))

#         dec4 = self.dec4(torch.cat([enc4, self.crop_and_concat(self.up(center), enc4)], 1))
#         dec3 = self.dec3(torch.cat([enc3, self.crop_and_concat(self.up(dec4), enc3)], 1))
#         dec2 = self.dec2(torch.cat([enc2, self.crop_and_concat(self.up(dec3), enc2)], 1))
#         dec1 = self.dec1(torch.cat([enc1, self.crop_and_concat(self.up(dec2), enc1)], 1))
#         final = self.final(dec1)

#         return torch.sigmoid(final)  # 保留输出维度

#     def crop_and_concat(self, upsampled, bypass):
#         diffY = bypass.size()[2] - upsampled.size()[2]
#         diffX = bypass.size()[3] - upsampled.size()[3]
#         upsampled = torch.nn.functional.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
#         return upsampled

        


# # 创建模型实例
# # input_channels = 3  # 输入通道，例如 RGB 图像
# # output_channels = 1  # 输出通道，例如单通道分割图
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model = UNet(input_channels, output_channels).to(device)

# # # 创建随机输入数据 (例如：batch_size=1, height=256, width=256)
# # input_data = torch.randn(1,input_channels ,256, 256).to(device)

# # # 将输入数据传入模型并获取输出
# # output = model(input_data)

# # # 输出结果的形状
# # print(output.shape)
# if __name__ == "__main__":
# # 创建模型实例
#     input_channels = 7
#     output_channels = 1  
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = UNet(input_channels, output_channels).to(device)
#     input_data = torch.randn(1,input_channels ,256, 256).to(device)
#     output = model(input_data)
#     print(output.shape)