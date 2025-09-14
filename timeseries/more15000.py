# import time
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import os

# def calculate_mae(real, pred):
#     """计算平均绝对误差"""
#     return np.mean(np.abs(real - pred))

# def calculate_rmse(real, pred):
#     """计算均方根误差"""
#     return np.sqrt(np.mean((real - pred) ** 2))

# def calculate_mape(real, pred):
#     """计算平均绝对百分比误差"""
#     # 避免除零错误
#     mask = real != 0
#     if np.sum(mask) == 0:
#         return np.inf
#     return np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100

# def calculate_r2(real, pred):
#     """计算决定系数R²"""
#     return r2_score(real, pred)

# def calculate_metrics(real, pred):
#     """计算所有评价指标"""
#     mae = calculate_mae(real, pred)
#     rmse = calculate_rmse(real, pred)
#     mape = calculate_mape(real, pred)
#     r2 = calculate_r2(real, pred)
    
#     return {
#         'MAE': mae,
#         'RMSE': rmse,
#         'MAPE': mape,
#         'R2': r2
#     }

# np.random.seed(0)
# torch.manual_seed(0)

# # 读取数据
# data_path = r'E:\sar\csv\transposed_output2019all.csv'  # 填你自己的数据地址
# raw_data = pd.read_csv(data_path)

# # 自动选择15000个数值列进行训练
# # 排除可能的索引列或日期列
# numeric_columns = [col for col in raw_data.columns if raw_data[col].dtype in ['float64', 'int64']]

# # 如果数据列数超过15000，选择前15000列；否则使用所有数值列
# if len(numeric_columns) >= 15000:
#     target_columns = numeric_columns[:15000]
#     print(f"数据包含{len(numeric_columns)}个数值列，选择前15000列进行训练")
# else:
#     target_columns = numeric_columns
#     print(f"数据只包含{len(numeric_columns)}个数值列，将全部用于训练")

# print(f"选择的训练列数量: {len(target_columns)}")
# print(f"数据总长度: {len(raw_data)}")

# # 数据预处理
# multi_point_data = raw_data[target_columns].values  # shape: (time_steps, num_points)
# num_points = len(target_columns)
# print(f"数据形状: {multi_point_data.shape}")

# # 定义参数
# test_size = 0.4
# train_size = 0.6
# pre_len = 4
# train_window = 30

# # 计算训练集和测试集的分割点
# train_len = int(train_size * len(multi_point_data))
# test_len = int(test_size * len(multi_point_data))

# train_data = multi_point_data[:train_len]
# test_data = multi_point_data[-test_len:]

# print(f"训练集尺寸: {train_data.shape}")
# print(f"测试集尺寸: {test_data.shape}")

# # 对每个点分别进行标准化
# print("开始数据标准化...")
# scalers = {}
# train_data_normalized = np.zeros_like(train_data)
# test_data_normalized = np.zeros_like(test_data)

# # 批量处理标准化以提高效率
# for i, col in enumerate(target_columns):
#     if i % 1000 == 0:  # 每1000列打印一次进度
#         print(f"标准化进度: {i}/{len(target_columns)}")
    
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     train_data_normalized[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
#     test_data_normalized[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
#     scalers[col] = scaler

# print("数据标准化完成")

# # 转化为深度学习模型需要的类型Tensor
# train_data_normalized = torch.FloatTensor(train_data_normalized)
# test_data_normalized = torch.FloatTensor(test_data_normalized)

# def create_multipoint_sequences(input_data, tw, pre_len):
#     """创建多点时间序列数据
#     input_data: (time_steps, num_points)
#     返回: list of (input_seq, target_seq)
#     """
#     inout_seq = []
#     L = len(input_data)
#     for i in range(L - tw - pre_len + 1):
#         train_seq = input_data[i:i + tw]  # (tw, num_points)
#         train_label = input_data[i + tw:i + tw + pre_len]  # (pre_len, num_points)
#         inout_seq.append((train_seq, train_label))
#     return inout_seq

# # 定义训练器的输入
# print("创建训练序列...")
# train_inout_seq = create_multipoint_sequences(train_data_normalized, train_window, pre_len)
# print(f"训练序列数量: {len(train_inout_seq)}")
# print(f"每个输入序列形状: {train_inout_seq[0][0].shape}")
# print(f"每个标签序列形状: {train_inout_seq[0][1].shape}")

# class MultiPointLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim=512, output_dim=None, num_layers=2, dropout=0.2):
#         super(MultiPointLSTM, self).__init__()
        
#         self.input_dim = input_dim  # 输入特征数（点的数量）
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.output_dim = output_dim if output_dim else input_dim
        
#         # 由于输入维度很大，添加输入投影层
#         self.input_projection = nn.Linear(input_dim, hidden_dim)
        
#         # LSTM层
#         self.lstm = nn.LSTM(
#             input_size=hidden_dim,  # 使用投影后的维度
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, self.output_dim * pre_len)
        
#         # Dropout层
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, input_dim)
#         batch_size, seq_len, _ = x.size()
        
#         # 输入投影
#         x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
#         x = torch.relu(x)
        
#         # 初始化隐藏状态和细胞状态
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
#         # LSTM前向传播
#         lstm_out, _ = self.lstm(x, (h0, c0))
        
#         # 取最后一个时间步的输出
#         last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
#         # 应用dropout
#         last_output = self.dropout(last_output)
        
#         # 输出投影
#         output = self.output_projection(last_output)  # (batch_size, output_dim * pre_len)
        
#         # 重新整形为 (batch_size, pre_len, output_dim)
#         output = output.view(batch_size, pre_len, self.output_dim)
        
#         return output

# # 模型参数 - 针对大规模数据调整
# lstm_model = MultiPointLSTM(
#     input_dim=num_points,
#     hidden_dim=1024,  # 增大隐藏层以处理更多点
#     output_dim=num_points,
#     num_layers=3,
#     dropout=0.3
# )

# # 检查是否有GPU可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"使用设备: {device}")
# lstm_model = lstm_model.to(device)

# # 计算模型参数数量
# total_params = sum(p.numel() for p in lstm_model.parameters())
# print(f"模型总参数数量: {total_params:,}")

# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0005, weight_decay=1e-5)  # 降低学习率
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.7, patience=50, verbose=True
# )

# epochs = 1000  # 减少epoch数量，避免过长训练时间
# batch_size = 8  # 使用批处理
# Train = True  # 设置为训练模式

# if Train:
#     losses = []
#     lstm_model.train()
#     start_time = time.time()
    
#     print("开始训练...")
#     best_loss = float('inf')
#     patience = 150
#     patience_counter = 0
    
#     for epoch in range(epochs):
#         epoch_losses = []
        
#         # 批处理训练
#         for i in range(0, len(train_inout_seq), batch_size):
#             optimizer.zero_grad()
            
#             batch_seq = []
#             batch_labels = []
            
#             # 构建批次
#             for j in range(i, min(i + batch_size, len(train_inout_seq))):
#                 seq, labels = train_inout_seq[j]
#                 batch_seq.append(seq)
#                 batch_labels.append(labels)
            
#             # 转换为张量并移动到设备
#             batch_seq = torch.stack(batch_seq).to(device)  # (batch_size, seq_len, num_points)
#             batch_labels = torch.stack(batch_labels).to(device)  # (batch_size, pre_len, num_points)
            
#             y_pred = lstm_model(batch_seq)
#             single_loss = loss_function(y_pred, batch_labels)
            
#             single_loss.backward()
#             # 梯度裁剪，防止梯度爆炸
#             torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             epoch_losses.append(single_loss.item())
        
#         # 计算epoch平均损失
#         avg_loss = np.mean(epoch_losses)
#         losses.append(avg_loss)
#         scheduler.step(avg_loss)
        
#         # 早停机制
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             patience_counter = 0
#             # 保存最佳模型
#             torch.save({
#                 'model_state_dict': lstm_model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scalers': scalers,
#                 'target_columns': target_columns,
#                 'loss': best_loss,
#                 'epoch': epoch
#             }, 'best_15000point_model.pth')
#         else:
#             patience_counter += 1
            
#         # 每20个epoch打印一次
#         if (epoch + 1) % 20 == 0:
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.8f}, LR: {current_lr:.8f}')
            
#         # 早停
#         if patience_counter > patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break
    
#     # 保存最终模型
#     torch.save({
#         'model_state_dict': lstm_model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scalers': scalers,
#         'target_columns': target_columns,
#         'loss': avg_loss,
#         'epoch': epoch
#     }, 'final_15000point_model.pth')
    
#     print(f"训练完成，用时: {(time.time() - start_time) / 60:.4f} 分钟")
    
#     # 绘制训练损失曲线
#     plt.figure(figsize=(12, 6))
#     plt.plot(losses)
#     plt.title('15000-Point LSTM Training Loss Curve')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.yscale('log')
#     plt.grid(True)
#     plt.savefig('15000point_training_loss.png', dpi=300, bbox_inches='tight')
#     plt.show()

# # 测试阶段
# print("\n开始测试...")
# lstm_model.eval()

# # 创建测试序列
# test_inout_seq = create_multipoint_sequences(test_data_normalized, train_window, pre_len)
# print(f"测试序列数量: {len(test_inout_seq)}")

# # 存储所有预测结果
# all_predictions = []
# all_targets = []

# with torch.no_grad():
#     # 批处理预测以节省内存
#     for i in range(0, len(test_inout_seq), batch_size):
#         batch_seq = []
#         batch_targets = []
        
#         for j in range(i, min(i + batch_size, len(test_inout_seq))):
#             seq, target = test_inout_seq[j]
#             batch_seq.append(seq)
#             batch_targets.append(target)
        
#         batch_seq = torch.stack(batch_seq).to(device)
#         predictions = lstm_model(batch_seq)
        
#         all_predictions.extend(predictions.cpu().numpy())
#         all_targets.extend(batch_targets)

# # 转换为numpy数组
# all_predictions = np.array(all_predictions)  # (num_samples, pre_len, num_points)
# all_targets = np.array(all_targets)          # (num_samples, pre_len, num_points)

# print(f"预测结果形状: {all_predictions.shape}")
# print(f"真实值形状: {all_targets.shape}")

# # 计算整体指标（由于点数过多，只计算整体指标）
# print("计算整体预测指标...")

# # 将所有预测和真实值展平
# all_pred_flat = all_predictions.flatten()
# all_target_flat = all_targets.flatten()

# # 反归一化（采样部分数据进行计算，避免内存不足）
# sample_indices = np.random.choice(len(all_pred_flat), size=min(100000, len(all_pred_flat)), replace=False)
# sample_pred = all_pred_flat[sample_indices]
# sample_target = all_target_flat[sample_indices]

# # 对于整体指标，我们需要知道每个点属于哪一列，这里简化处理
# print("由于数据量过大，计算采样数据的整体指标:")
# overall_mae = calculate_mae(sample_target, sample_pred)
# overall_rmse = calculate_rmse(sample_target, sample_pred)
# overall_mape = calculate_mape(sample_target, sample_pred)
# overall_r2 = calculate_r2(sample_target, sample_pred)

# print(f"整体指标 (基于{len(sample_indices)}个采样点):")
# print(f"  MAE: {overall_mae:.6f}")
# print(f"  RMSE: {overall_rmse:.6f}")
# print(f"  MAPE: {overall_mape:.4f}%")
# print(f"  R²: {overall_r2:.6f}")

# # 可视化部分预测结果（只显示前几个点）
# fig, axes = plt.subplots(min(5, num_points), 1, figsize=(15, 3*min(5, num_points)))
# if min(5, num_points) == 1:
#     axes = [axes]

# for i in range(min(5, num_points)):
#     # 选择前20个样本进行可视化
#     n_vis = min(20, len(all_predictions))
    
#     pred_flat = all_predictions[:n_vis, :, i].flatten()
#     target_flat = all_targets[:n_vis, :, i].flatten()
    
#     axes[i].plot(target_flat, label='real', alpha=0.8)
#     axes[i].plot(pred_flat, label='预测值', alpha=0.8)
#     axes[i].set_title(f'第{i+1}个点预测结果对比')
#     axes[i].legend()
#     axes[i].grid(True)

# plt.tight_layout()
# plt.savefig('15000point_prediction_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# print(f"\n训练完成！共训练了{num_points}个点的时间序列预测模型")
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def calculate_mae(real, pred):
    """Calculate the mean absolute error"""
    return np.mean(np.abs(real - pred))

def calculate_rmse(real, pred):
    """Calculate the root mean square error"""
    return np.sqrt(np.mean((real - pred) ** 2))

def calculate_mape(real, pred):
    """Calculate the mean absolute percentage error"""
    # Avoid division by zero
    mask = real != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100

def calculate_r2(real, pred):
    """Calculate the coefficient of determination R²"""
    return r2_score(real, pred)

def calculate_metrics(real, pred):
    """Calculate all evaluation metrics"""
    mae = calculate_mae(real, pred)
    rmse = calculate_rmse(real, pred)
    mape = calculate_mape(real, pred)
    r2 = calculate_r2(real, pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def denormalize_predictions(predictions, targets, scalers, target_columns):
    """反归一化预测结果和真实值"""
    pred_denorm = np.zeros_like(predictions)
    target_denorm = np.zeros_like(targets)
    
    # predictions shape: (num_samples, pre_len, num_points)
    # targets shape: (num_samples, pre_len, num_points)
    
    for i, col in enumerate(target_columns):
        scaler = scalers[col]
        # 对每个时间点进行反归一化
        for t in range(predictions.shape[1]):  # pre_len
            pred_denorm[:, t, i] = scaler.inverse_transform(
                predictions[:, t, i].reshape(-1, 1)
            ).flatten()
            target_denorm[:, t, i] = scaler.inverse_transform(
                targets[:, t, i].reshape(-1, 1)
            ).flatten()
    
    return pred_denorm, target_denorm

np.random.seed(0)
torch.manual_seed(0)

# Read data
data_path = r'E:\sar\csv\transposed_output2019_2022.csv'  # Enter your own data address
raw_data = pd.read_csv(data_path)

# Automatically select 15,000 numeric columns for training
# Exclude possible index columns or date columns
numeric_columns = [col for col in raw_data.columns if raw_data[col].dtype in ['float64', 'int64']]

# If the number of data columns exceeds 15000, select the first 15000 columns; otherwise, use all numeric columns
if len(numeric_columns) >= 15000:
    target_columns = numeric_columns[:15000]
    print(f"The data contains {len(numeric_columns)} numeric columns, selecting the first 15000 columns for training")
else:
    target_columns = numeric_columns
    print(f"The data only contains {len(numeric_columns)} numeric columns, all will be used for training")

print(f"Number of selected training columns: {len(target_columns)}")
print(f"Total length of data: {len(raw_data)}")

# Data preprocessing
multi_point_data = raw_data[target_columns].values  # shape: (time_steps, num_points)
num_points = len(target_columns)
print(f"Data shape: {multi_point_data.shape}")

# Define parameters
test_size = 0.2
train_size = 0.8
pre_len = 6
train_window = 10

# Calculate the split point for the training set and test set
train_len = int(train_size * len(multi_point_data))
test_len = int(test_size * len(multi_point_data))

train_data = multi_point_data[:train_len]
test_data = multi_point_data[-test_len:]

print(f"Training set size: {train_data.shape}")
print(f"Test set size: {test_data.shape}")

# Standardize each point individually
print("Starting data standardization...")
scalers = {}
train_data_normalized = np.zeros_like(train_data)
test_data_normalized = np.zeros_like(test_data)

# Batch processing for normalization to improve efficiency
for i, col in enumerate(target_columns):
    if i % 1000 == 0:  # Print progress every 1000 columns
        print(f"Standardization progress: {i}/{len(target_columns)}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_normalized[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
    test_data_normalized[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
    scalers[col] = scaler

print("Data normalization completed")

# Convert to the type required by deep learning models Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized)
test_data_normalized = torch.FloatTensor(test_data_normalized)

def create_multipoint_sequences(input_data, tw, pre_len):
    """Create multi-point time series data
    input_data: (time_steps, num_points)
    Returns: list of (input_seq, target_seq)
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - pre_len + 1):
        train_seq = input_data[i:i + tw]  # (tw, num_points)
        train_label = input_data[i + tw:i + tw + pre_len]  # (pre_len, num_points)
        inout_seq.append((train_seq, train_label))
    return inout_seq

# Define the trainer's input
print("Creating training sequences...")
train_inout_seq = create_multipoint_sequences(train_data_normalized, train_window, pre_len)
print(f"Number of training sequences: {len(train_inout_seq)}")
print(f"Shape of each input sequence: {train_inout_seq[0][0].shape}")
print(f"Shape of each label sequence: {train_inout_seq[0][1].shape}")

class MultiPointLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=None, num_layers=2, dropout=0.2):
        super(MultiPointLSTM, self).__init__()

        self.input_dim = input_dim  # Number of input features (number of points)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim else input_dim

        # Since the input dimension is large, add an input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,  # Using the projected dimension
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, self.output_dim * pre_len)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        x = torch.relu(x)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # LSTM forward propagation
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Apply dropout
        last_output = self.dropout(last_output)

        # Output projection
        output = self.output_projection(last_output)  # (batch_size, output_dim * pre_len)

        # Reshape to (batch_size, pre_len, output_dim)
        output = output.view(batch_size, pre_len, self.output_dim)

        return output

# Model Parameters - Adjusted for Large-Scale Data
lstm_model = MultiPointLSTM(
    input_dim=num_points,
    hidden_dim=1024,  # Increase hidden layer to process more points
    output_dim=num_points,
    num_layers=3,
    dropout=0.3
)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
lstm_model = lstm_model.to(device)

# Calculate the number of model parameters
total_params = sum(p.numel() for p in lstm_model.parameters())
print(f"Total number of model parameters: {total_params:,}")

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0005, weight_decay=1e-5)  # Reduce learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=50, verbose=True
)

epochs = 1000  # Reduce the number of epochs to avoid excessively long training time
batch_size = 8  # Use batch processing
Train = True  # Set to training mode

if Train:
    losses = []
    lstm_model.train()
    start_time = time.time()

    print("Starting training...")
    best_loss = float('inf')
    patience = 150
    patience_counter = 0

    for epoch in range(epochs):
        epoch_losses = []

        # Batch training
        for i in range(0, len(train_inout_seq), batch_size):
            optimizer.zero_grad()

            batch_seq = []
            batch_labels = []

            # Construct batch
            for j in range(i, min(i + batch_size, len(train_inout_seq))):
                seq, labels = train_inout_seq[j]
                batch_seq.append(seq)
                batch_labels.append(labels)

            # Convert to tensor and move to device
            batch_seq = torch.stack(batch_seq).to(device)  # (batch_size, seq_len, num_points)
            batch_labels = torch.stack(batch_labels).to(device)  # (batch_size, pre_len, num_points)

            y_pred = lstm_model(batch_seq)
            single_loss = loss_function(y_pred, batch_labels)

            single_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(single_loss.item())

        # Calculate epoch average loss
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Early stopping mechanism
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save the best model
            torch.save({
                'model_state_dict': lstm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scalers': scalers,
                'target_columns': target_columns,
                'loss': best_loss,
                'epoch': epoch
            }, 'best_15000point_model.pth')
        else:
            patience_counter += 1

        # Print every 20 epochs
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.8f}, LR: {current_lr:.8f}')

        # Early stopping
        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save the final model
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scalers': scalers,
        'target_columns': target_columns,
        'loss': avg_loss,
        'epoch': epoch
    }, 'final_15000point_model.pth')

    print(f"Training completed, time taken: {(time.time() - start_time) / 60:.4f} minutes")

    # Plot training loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.title('15000-Point LSTM Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('15000point_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

# Test Phase
print("\nStarting testing...")
lstm_model.eval()

# Create test sequences
test_inout_seq = create_multipoint_sequences(test_data_normalized, train_window, pre_len)
print(f"Number of test sequences: {len(test_inout_seq)}")

# Store all prediction results
all_predictions = []
all_targets = []

with torch.no_grad():
    # Batch processing for memory saving
    for i in range(0, len(test_inout_seq), batch_size):
        batch_seq = []
        batch_targets = []

        for j in range(i, min(i + batch_size, len(test_inout_seq))):
            seq, target = test_inout_seq[j]
            batch_seq.append(seq)
            batch_targets.append(target)

        batch_seq = torch.stack(batch_seq).to(device)
        predictions = lstm_model(batch_seq)

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(batch_targets)

# Convert to numpy array
all_predictions = np.array(all_predictions)  # (num_samples, pre_len, num_points)
all_targets = np.array(all_targets)          # (num_samples, pre_len, num_points)

print(f"Prediction shape: {all_predictions.shape}")
print(f"True values shape: {all_targets.shape}")

# 反归一化预测结果和真实值
print("Denormalizing predictions and targets...")
all_pred_denorm, all_target_denorm = denormalize_predictions(
    all_predictions, all_targets, scalers, target_columns
)

print("Calculating overall prediction metrics on denormalized data...")

# Flatten all denormalized predictions and true values
all_pred_flat = all_pred_denorm.flatten()
all_target_flat = all_target_denorm.flatten()

# 计算整体指标
print("Calculating metrics for all data points:")
overall_mae = calculate_mae(all_target_flat, all_pred_flat)
overall_rmse = calculate_rmse(all_target_flat, all_pred_flat)
overall_mape = calculate_mape(all_target_flat, all_pred_flat)
overall_r2 = calculate_r2(all_target_flat, all_pred_flat)

print(f"Overall metrics (based on denormalized data):")
print(f"  MAE: {overall_mae:.6f}")
print(f"  RMSE: {overall_rmse:.6f}")
print(f"  MAPE: {overall_mape:.4f}%")
print(f"  R²: {overall_r2:.6f}")

# 也计算每个时间步的平均指标
print("\nMetrics by time step (average across all points):")
for t in range(pre_len):
    step_pred = all_pred_denorm[:, t, :].flatten()
    step_target = all_target_denorm[:, t, :].flatten()
    
    step_mae = calculate_mae(step_target, step_pred)
    step_rmse = calculate_rmse(step_target, step_pred)
    step_mape = calculate_mape(step_target, step_pred)
    step_r2 = calculate_r2(step_target, step_pred)
    
    print(f"  Time step {t+1}: MAE={step_mae:.6f}, RMSE={step_rmse:.6f}, MAPE={step_mape:.4f}%, R²={step_r2:.6f}")

# Visualize some predicted results (only show the first few points)
fig, axes = plt.subplots(min(5, num_points), 1, figsize=(15, 3*min(5, num_points)))
if min(5, num_points) == 1:
    axes = [axes]

for i in range(min(5, num_points)):
    # Select the first 20 samples for visualization, use denormalized data
    n_vis = min(20, len(all_pred_denorm))

    pred_flat = all_pred_denorm[:n_vis, :, i].flatten()
    target_flat = all_target_denorm[:n_vis, :, i].flatten()

    axes[i].plot(target_flat, label='True Values', alpha=0.8)
    axes[i].plot(pred_flat, label='Predicted Values', alpha=0.8)
    axes[i].set_title(f'Prediction Results Comparison for Point {i+1}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig('15000point_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nTraining completed! A total of {num_points} time series prediction models were trained")
print("All metrics are calculated based on denormalized (original scale) data")
