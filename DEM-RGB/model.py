import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, img_channels, output_channels):
        super(UNet, self).__init__()
        
        self.encoder1 = self.contract_block(img_channels, 16)
        self.encoder2 = self.contract_block(16, 32)
        self.encoder3 = self.contract_block(32, 64)
        self.encoder4 = self.contract_block(64, 128)
        self.bottleneck = self.contract_block(128, 256)
        
        self.decoder4 = self.expand_block(256 + 128, 128)  # 加上 encoder4 的通道数
        self.decoder3 = self.expand_block(128 + 64, 64)   # 加上 encoder3 的通道数
        self.decoder2 = self.expand_block(64 + 32, 32)     # 加上 encoder2 的通道数
        self.decoder1 = self.expand_block(32 + 16, 16)     # 加上 encoder1 的通道数
        
        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=(1, 1))

    def contract_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def expand_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)

        # 打印每一层的输出形状
        # print(f"enc1 shape: {enc1.shape}")
        # print(f"enc2 shape: {enc2.shape}")
        # print(f"enc3 shape: {enc3.shape}")
        # print(f"enc4 shape: {enc4.shape}")
        # print(f"bottleneck shape: {bottleneck.shape}")

        # 上采样到 enc4 的大小
        bottleneck_up = torch.nn.functional.interpolate(bottleneck, size=(16, 16), mode='bilinear', align_corners=False)
        dec4 = self.decoder4(torch.cat((bottleneck_up, enc4), dim=1))  # 拼接后传入 decoder4
        # print(f"dec4 shape: {dec4.shape}")

        dec3 = self.decoder3(torch.cat((dec4, enc3), dim=1))
        # print(f"dec3 shape: {dec3.shape}")

        dec2 = self.decoder2(torch.cat((dec3, enc2), dim=1))
        # print(f"dec2 shape: {dec2.shape}")

        dec1 = self.decoder1(torch.cat((dec2, enc1), dim=1))
        # print(f"dec1 shape: {dec1.shape}")

        output = self.final_conv(dec1)
        return output


# # 测试模型
# if __name__ == "__main__":
# # 创建模型实例
#     input_channels = 7 
#     output_channels = 1  
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = UNet(input_channels, output_channels).to(device)

#     input_data = torch.randn(1,input_channels ,256, 256).to(device)

#     output = model(input_data)

#     print(output.shape)





