import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset import TRAIN_XX, TRAIN_YY

# 将数据转换为 PyTorch 张量
X_tensor = torch.tensor(TRAIN_XX, dtype=torch.float32)
Y_tensor = torch.tensor(TRAIN_YY, dtype=torch.float32)

# 划分训练集和验证集
x_train, x_valid, y_train, y_valid = train_test_split(X_tensor, Y_tensor, test_size=0.2, shuffle=True)

# 创建 TensorDataset
train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 示例：遍历 DataLoader
for batch_X, batch_Y in train_loader:
    # 这里可以进行模型训练
    print(batch_X.shape, batch_Y.shape)
    break  # 只打印第一个批次