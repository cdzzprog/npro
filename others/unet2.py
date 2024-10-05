import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # 编码器
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        # 解码器
        self.upconv6 = self.upconv_block(1024, 512)
        self.upconv7 = self.upconv_block(512, 256)
        self.upconv8 = self.upconv_block(256, 128)
        self.upconv9 = self.upconv_block(128, 64)

        # 合并操作
        self.merge6 = self.conv_block(1024, 512, batch_norm=False)
        self.merge7 = self.conv_block(512, 256, batch_norm=False)
        self.merge8 = self.conv_block(256, 128, batch_norm=False)
        self.merge9 = self.conv_block(128, 64, batch_norm=False)

        # 输出层
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, batch_norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))

        # 解码
        up6 = self.upconv6(enc5)
        merge6 = torch.cat([up6, enc4], dim=1)
        conv6 = self.merge6(merge6)

        up7 = self.upconv7(conv6)
        merge7 = torch.cat([up7, enc3], dim=1)
        conv7 = self.merge7(merge7)

        up8 = self.upconv8(conv7)
        merge8 = torch.cat([up8, enc2], dim=1)
        conv8 = self.merge8(merge8)

        up9 = self.upconv9(conv8)
        merge9 = torch.cat([up9, enc1], dim=1)
        conv9 = self.merge9(merge9)

        outputs = self.final(conv9)
        return outputs

# 测试模型的定义
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)  # 这里设置输入通道和输出通道数
    x = torch.randn(4, 3, 256, 256)  # 创建一个形状为 (4, 3, 256, 256) 的输入张量
    output = model(x)
    print(output.shape)  # 应该打印 (4, 1, 256, 256)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         # 编码器
#         self.enc1 = self.conv_block(3, 64)   # 输入通道数调整为 3
#         self.enc2 = self.conv_block(64, 128)
#         self.enc3 = self.conv_block(128, 256)
#         self.enc4 = self.conv_block(256, 512)
#         self.enc5 = self.conv_block(512, 1024)

#         # 解码器
#         self.upconv6 = self.upconv_block(1024, 512)
#         self.upconv7 = self.upconv_block(512, 256)
#         self.upconv8 = self.upconv_block(256, 128)
#         self.upconv9 = self.upconv_block(128, 64)

#         # 合并操作
#         self.merge6 = self.conv_block(1024, 512, batch_norm=False)
#         self.merge7 = self.conv_block(512, 256, batch_norm=False)
#         self.merge8 = self.conv_block(256, 128, batch_norm=False)
#         self.merge9 = self.conv_block(128, 64, batch_norm=False)

#         # 输出层
#         self.final = nn.Conv2d(64, 1, kernel_size=1)

#     def conv_block(self, in_channels, out_channels, batch_norm=True):
#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         ]
#         if batch_norm:
#             layers.insert(2, nn.BatchNorm2d(out_channels))
#         return nn.Sequential(*layers)

#     def upconv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # 编码
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(F.max_pool2d(enc1, 2))
#         enc3 = self.enc3(F.max_pool2d(enc2, 2))
#         enc4 = self.enc4(F.max_pool2d(enc3, 2))
#         enc5 = self.enc5(F.max_pool2d(enc4, 2))

#         # 解码
#         up6 = self.upconv6(enc5)
#         merge6 = torch.cat([up6, enc4], dim=1)
#         conv6 = self.merge6(merge6)

#         up7 = self.upconv7(conv6)
#         merge7 = torch.cat([up7, enc3], dim=1)
#         conv7 = self.merge7(merge7)

#         up8 = self.upconv8(conv7)
#         merge8 = torch.cat([up8, enc2], dim=1)
#         conv8 = self.merge8(merge8)

#         up9 = self.upconv9(conv8)
#         merge9 = torch.cat([up9, enc1], dim=1)
#         conv9 = self.merge9(merge9)

#         outputs = self.final(conv9)
#         return outputs

# # 测试模型的定义
# if __name__ == "__main__":
#     model = UNet()
#     x = torch.randn(4, 3, 256, 256)  # 创建一个形状为 (4, 3, 256, 256) 的输入张量
#     output = model(x)
#     print(output.shape)  # 应该打印 (4, 1, 256, 256)

    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F 

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         # 编码器
#         self.enc1 = self.conv_block(1, 64)
#         self.enc2 = self.conv_block(64, 128)
#         self.enc3 = self.conv_block(128, 256)
#         self.enc4 = self.conv_block(256, 512)
#         self.enc5 = self.conv_block(512, 1024)

#         # 解码器
#         self.upconv6 = self.upconv_block(1024, 512)
#         self.upconv7 = self.upconv_block(512, 256)
#         self.upconv8 = self.upconv_block(256, 128)
#         self.upconv9 = self.upconv_block(128, 64)

#         # 合并操作
#         self.merge6 = self.conv_block(1024, 512, batch_norm=False)
#         self.merge7 = self.conv_block(512, 256, batch_norm=False)
#         self.merge8 = self.conv_block(256, 128, batch_norm=False)
#         self.merge9 = self.conv_block(128, 64, batch_norm=False)

#         # 输出层
#         self.final = nn.Conv2d(64, 1, kernel_size=1)

#     def conv_block(self, in_channels, out_channels, batch_norm=True):
#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         ]
#         if batch_norm:
#             layers.insert(2, nn.BatchNorm2d(out_channels))
#         return nn.Sequential(*layers)

#     def upconv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # 编码
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(F.max_pool2d(enc1, 2))
#         enc3 = self.enc3(F.max_pool2d(enc2, 2))
#         enc4 = self.enc4(F.max_pool2d(enc3, 2))
#         enc5 = self.enc5(F.max_pool2d(enc4, 2))

#         # 解码
#         up6 = self.upconv6(enc5)
#         merge6 = torch.cat([up6, enc4], dim=1)
#         conv6 = self.merge6(merge6)

#         up7 = self.upconv7(conv6)
#         merge7 = torch.cat([up7, enc3], dim=1)
#         conv7 = self.merge7(merge7)

#         up8 = self.upconv8(conv7)
#         merge8 = torch.cat([up8, enc2], dim=1)
#         conv8 = self.merge8(merge8)

#         up9 = self.upconv9(conv8)
#         merge9 = torch.cat([up9, enc1], dim=1)
#         conv9 = self.merge9(merge9)

#         outputs = self.final(conv9)
#         return outputs

# x = torch.randn(1, 1, 160, 160)  # 修改为适应模型输入要求的尺寸和通道数
# model = UNet()
# print(model)
# preds = model(x)
# print(preds.shape)  # 输出模型预测结果的形状






# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

#定义模型
# def unet_models():
#     #输入层
#     inputs=Input(shape(256,256,1))

#     #编码器
#     conv1=Conv2D(64,3,activation="relu",padding="same")(iputs)
#     conv1=Conv2D(64,3,activation="relu",padding="same")(conv1)
#     pool1=MaxPooling2D(pool_size=(2,2))(conv1)

#     conv2=Conv2D(128,3,activation="relu",padding="same")(pool1)
#     conv2=Conv2D(128,3,activation="relu",padding="same")(conv2)
#     pool2=MaxPooling2D(pool_size=(2,2))(conv2)

#     conv3=Conv2D(256,3,activation="relu",padding="same")(pool2)
#     conv3=Conv2D(256,3,activation="relu",padding="same")(conv3)
#     pool3=MaxPooling2D(pool_size=(2,2))(conv3)

#     conv4=Conv2D(512,3,activation="relu",padding="same")(pool3)
#     conv4=Conv2D(512,3,activation="relu",padding="same")(conv4)
#     pool4=MaxPooling2D(pool_size=(2,2))(conv4)

#     conv5=Conv2D(1024,3,activation="relu",padding="same")(pool4)
#     conv5=Conv2D(1024,3,activation="relu",padding="same")(conv5)
   
#     #解码器

#     up6 = UpSampling2D(size=(2, 2))(conv5)
#     up6=Conv2D(512,2,activation="relu",padding="same")(up6)
#     merge6=Concatenate()([conv4,up6])
#     conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

#     up7 = UpSampling2D(size=(2, 2))(conv6)
#     up6=Conv2D(256,2,activation="relu",padding="same")(up7)
#     merge7=Concatenate()([conv3,up7])
#     conv7= Conv2D(256, 3, activation='relu', padding='same')(merge7)
#     conv7= Conv2D(256, 3, activation='relu', padding='same')(conv7)


#     up8 = UpSampling2D(size=(2, 2))(conv7)
#     up8=Conv2D(128,2,activation="relu",padding="same")(up8)
#     merge8=Concatenate()([conv2,up8])
#     conv8= Conv2D(128, 3, activation='relu', padding='same')(merge8)
#     conv8= Conv2D(128, 3, activation='relu', padding='same')(conv8)


#     up9 = UpSampling2D(size=(2, 2))(conv8)
#     up9=Conv2D(128,2,activation="relu",padding="same")(up9)
#     merge9=Concatenate()([conv1,up9])
#     conv9= Conv2D(128, 3, activation='relu', padding='same')(merge9)
#     conv9= Conv2D(128, 3, activation='relu', padding='same')(conv9)

#     outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
  
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

    
