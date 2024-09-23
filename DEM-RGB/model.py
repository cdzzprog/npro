import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Net model
class UNet(nn.Module):
    def __init__(self, img_channels, output_channels):
        super(UNet, self).__init__()
        
        self.encoder1 = self.contract_block(img_channels, 16)
        self.encoder2 = self.contract_block(16, 32)
        self.encoder3 = self.contract_block(32, 64)
        self.encoder4 = self.contract_block(64, 128)
        self.bottleneck = self.contract_block(128, 256)
        
        self.decoder4 = self.expand_block(256, 128)
        self.decoder3 = self.expand_block(128, 64)
        self.decoder2 = self.expand_block(64, 32)
        self.decoder1 = self.expand_block(32, 16)
        
        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=(1, 1), activation='sigmoid')

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
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
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
        
        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Concatenate along channel dimension
        
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        
        output = self.final_conv(dec1)
        return output

# Model instantiation
IMG_WIDTH = 256  # Example width
IMG_HEIGHT = 256  # Example height
IMG_CHANNELS = 3  # Example channels
output_channels = 1  # For binary segmentation

model = UNet(IMG_CHANNELS, output_channels)
