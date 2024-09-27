import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import train_loader,valid_loader
from model import UNet


model = UNet(img_channels=7, output_channels=1)  
# model.train()
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()  # 根据任务调整损失函数
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


train_loss_history = []
val_loss_history = []
# 训练过程
def main(args):
    

    for epoch in range(0,args.epochs):
     
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)

            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)

    # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in valid_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        val_loss_history.append(val_loss)

        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


torch.save(model.state_dict(), "model_save.pth")




# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
# ax1.plot(train_loss_history, label='Train Loss')
# ax1.plot(val_loss_history, label='Validation Loss')
# ax1.set_title('Loss Over Epochs')
# ax1.set_ylabel('Loss')
# ax1.set_xlabel('Epoch')
# ax1.legend()


# plt.show()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument("--num-classes", default=1, type=int)                                                           # 类别数；不包含背景
    parser.add_argument("--device", default="cuda", help="training device")                                             # 默认使用GPU
    parser.add_argument("-b", "--batch-size", default=2, type=int)                                                      # batch_size
    parser.add_argument("--epochs", default=50, type=int, metavar="N",                                                  # epochs
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')                               # 超参数；学习率
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')                                    # 打印频率
    parser.add_argument("--amp", default=True, type=bool,                                                               # 使用混合精度训练，较老显卡（如10系列）不支持，需要改为False
                        help="Use torch.cud"
                             "a.amp for mixed precision training")
    args = parser.parse_args()

    return args
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
