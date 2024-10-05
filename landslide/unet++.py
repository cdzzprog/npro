import torch.nn as nn
import torch
from osgeo import gdal
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define UNet++ model
class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(UNetPlusPlus, self).__init__()

        self.enc1 = self.conv_block(input_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.center = self.conv_block(256, 512)
        
        self.up3_2 = self.conv_block(256 + 128, 128)
        self.up2_2 = self.conv_block(128 + 64, 64)
        self.up1_2 = self.conv_block(64 + 32, 32)

        self.up3_1 = self.conv_block(128 + 128, 128)
        self.up2_1 = self.conv_block(64 + 64, 64)
        self.up1_1 = self.conv_block(32 + 32, 32)

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
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        center = self.center(self.pool(enc4))

        print(f"enc3 shape: {enc3.shape}")
        print(f"center shape after upsample: {self.up(center).shape}")

        up3_2 = self.up3_2(torch.cat([self.crop_and_concat(self.up(center), enc3)], dim=1))
        print(f"up3_2 shape: {up3_2.shape}")

        up2_2 = self.up2_2(torch.cat([self.crop_and_concat(self.up(up3_2), enc2)], dim=1))
        up1_2 = self.up1_2(torch.cat([self.crop_and_concat(self.up(up2_2), enc1)], dim=1))

        up3_1 = self.up3_1(torch.cat([self.crop_and_concat(up3_2, enc3)], dim=1))
        up2_1 = self.up2_1(torch.cat([self.crop_and_concat(up2_2, enc2)], dim=1))
        up1_1 = self.up1_1(torch.cat([self.crop_and_concat(up1_2, enc1)], dim=1))

        final = self.final(up1_1).squeeze()

        return torch.sigmoid(final)


        
    def crop_and_concat(self, upsampled, bypass):
        # 获取bypass的尺寸
        _, _, h_bypass, w_bypass = bypass.size()
        _, _, h_upsampled, w_upsampled = upsampled.size()
        
        # 计算裁剪尺寸
        delta_h = (h_upsampled - h_bypass) // 2
        delta_w = (w_upsampled - w_bypass) // 2

        upsampled = upsampled[:, :, delta_h:delta_h + h_bypass, delta_w:delta_w + w_bypass]

        return torch.cat((upsampled, bypass), 1)

        cropped = self.crop_and_concat(self.up(center), enc3)
        print(f"Cropped shape: {cropped.shape}")




        
class RSDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images = self.read_multiband_images(images_dir)
        self.labels = self.read_singleband_labels(labels_dir)

    def read_multiband_images(self, images_dir):
        images = []
        for image_file in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_file)
            rsdl_data = gdal.Open(image_path)
            if rsdl_data is None:
                raise FileNotFoundError(f"Unable to open image file: {image_path}")
            images.append(np.stack([rsdl_data.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0))
        return images

    def read_singleband_labels(self, labels_dir):
        labels = []
        for label_file in os.listdir(labels_dir):
            label_path = os.path.join(labels_dir, label_file)
            rsdl_data = gdal.Open(label_path)
            if rsdl_data is None:
                raise FileNotFoundError(f"Unable to open label file: {label_path}")
            labels.append(rsdl_data.GetRasterBand(1).ReadAsArray())
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)

def custom_collate_fn(batch):
        images, labels = zip(*batch)
        images = [image.clone().detach() for image in images]
        labels = [label.clone().detach() for label in labels]
        
        # 找到最大宽度和高度
        max_width = max([img.shape[2] for img in images])
        max_height = max([img.shape[1] for img in images])

        padded_images = []
        padded_labels = []
        
        for img, lbl in zip(images, labels):
            # 填充图像和标签
            pad_img = torch.zeros((img.shape[0], max_height, max_width))
            pad_img[:, :img.shape[1], :img.shape[2]] = img
            pad_lbl = torch.zeros((max_height, max_width))
            pad_lbl[:lbl.shape[0], :lbl.shape[1]] = lbl
            padded_images.append(pad_img)
            padded_labels.append(pad_lbl)
        
        return torch.stack(padded_images), torch.stack(padded_labels)

# Directories for training data
images_dir = 'E:\数据集\山体滑坡数据集\landslide\image3/'
labels_dir = 'E:\数据集\山体滑坡数据集\landslide\mask3/'

dataset = RSDataset(images_dir, labels_dir)
trainloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

model = UNetPlusPlus(3, 1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images = images.float().to(device)
        labels = labels.float().to(device) / 255.0
        outputs = model(images)
        labels = labels.squeeze(0)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

torch.save(model.state_dict(), 'unet_plus_plus_model.pth')

# Define a separate test DataLoader
test_images_dir = 'E:\数据集\山体滑坡数据集\landslide\image1/'
test_labels_dir = 'E:\数据集\山体滑坡数据集\landslide\mask1/'

test_dataset = RSDataset(test_images_dir, test_labels_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.float().to(device)
            labels = labels.float().to(device) / 255.0
            
            outputs = model(images).squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            all_predictions.extend(outputs.flatten())
            all_labels.extend(labels.flatten())

    return np.array(all_labels), np.array(all_predictions)

def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve (AP={average_precision:.2f})')
    plt.show()

# Evaluate and plot
y_true, y_scores = evaluate_model(model, test_loader)
plot_precision_recall_curve(y_true, y_scores)

# Print average precision score
average_precision = average_precision_score(y_true, y_scores)
print(f'Average Precision Score: {average_precision:.4f}')