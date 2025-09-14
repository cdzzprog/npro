import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from scipy.signal import savgol_filter  # 导入SG滤波

def calculate_mae(real, pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(real - pred))

def calculate_rmse(real, pred):
    """计算均方根误差"""
    return np.sqrt(np.mean((real - pred) ** 2))

def calculate_mape(real, pred):
    """计算平均绝对百分比误差"""
    # 避免除零错误
    mask = real != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100

def calculate_r2(real, pred):
    """计算决定系数R²"""
    return r2_score(real, pred)

def calculate_metrics(real, pred):
    """计算所有评价指标"""
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

np.random.seed(0)
torch.manual_seed(0)

# 读取数据
data_path = r'E:\sar\csv\transposed_output2019_2022.csv'  # 填你自己的数据地址
raw_data = pd.read_csv(data_path)

# 选择要训练的列（可以选择多个点）
# 方式1: 选择特定的列
#target_columns = ['2', '63', '85', '197', '237', '267', '269', '271', '273', '275', '277', '279', '281', '283', '285', '287', '289', '291', '293', '295', '297', '299', '301', '303', '305', '307', '309', '311', '313', '315', '317', '319', '321', '323', '325', '327']  # 可以根据需要修改选择的列
# target_columns = ['2']
# 方式2: 选择所有数值列（除了可能的日期/索引列）
# target_columns = [col for col in raw_data.columns if raw_data[col].dtype in ['float64', 'int64']]
target_columns = [col for col in raw_data.columns[:2] if raw_data[col].dtype in ['float64', 'int64']]

# target_columns = raw_data.columns[:100].tolist()
print(f"选择的训练列: {target_columns}")
print(f"数据总长度: {len(raw_data)}")

# 数据预处理
multi_point_data = raw_data[target_columns].values  # shape: (time_steps, num_points)
num_points = len(target_columns)
print(f"数据形状: {multi_point_data.shape}")

# ============================= 添加SG滤波处理 =============================
def apply_sg_filter(data, window_length=15, polyorder=2):
    """应用Savitzky-Golay滤波器平滑数据"""
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        # 处理数据长度小于窗口长度的情况
        wl = min(window_length, len(data))
        # 确保窗口长度是奇数
        wl = wl if wl % 2 == 1 else wl - 1
        if wl > polyorder:
            filtered_data[:, i] = savgol_filter(data[:, i], window_length=wl, polyorder=polyorder)
        else:
            # 数据太短无法应用SG滤波，使用原始数据
            filtered_data[:, i] = data[:, i]
            print(f"警告: 列{i}数据长度({len(data)})过短，无法应用SG滤波(需要最小长度{window_length})")
    return filtered_data

# 应用SG滤波
print("\n应用Savitzky-Golay滤波...")
window_length = 5  # 滤波窗口长度(奇数)
polyorder = 3      # 多项式阶数
multi_point_data = apply_sg_filter(multi_point_data, window_length, polyorder)
print("SG滤波完成!")
# ============================= SG滤波结束 =============================

# 定义参数
test_size = 0.15
train_size = 0.85
pre_len = 4
train_window = 10

# 计算训练集和测试集的分割点
train_len = int(train_size * len(multi_point_data))
test_len = int(test_size * len(multi_point_data))

train_data = multi_point_data[:train_len]
test_data = multi_point_data[-test_len:]

print(f"训练集尺寸: {train_data.shape}")
print(f"测试集尺寸: {test_data.shape}")

# 对每个点分别进行标准化
scalers = {}
train_data_normalized = np.zeros_like(train_data)
test_data_normalized = np.zeros_like(test_data)

for i, col in enumerate(target_columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_normalized[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
    test_data_normalized[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
    scalers[col] = scaler

# 转化为深度学习模型需要的类型Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized)
test_data_normalized = torch.FloatTensor(test_data_normalized)

def create_multipoint_sequences(input_data, tw, pre_len):
    """创建多点时间序列数据
    input_data: (time_steps, num_points)
    返回: list of (input_seq, target_seq)
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - pre_len + 1):
        train_seq = input_data[i:i + tw]  # (tw, num_points)
        train_label = input_data[i + tw:i + tw + pre_len]  # (pre_len, num_points)
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 定义训练器的输入
train_inout_seq = create_multipoint_sequences(train_data_normalized, train_window, pre_len)
print(f"训练序列数量: {len(train_inout_seq)}")
print(f"每个输入序列形状: {train_inout_seq[0][0].shape}")
print(f"每个标签序列形状: {train_inout_seq[0][1].shape}")

class MultiPointLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None, num_layers=2, dropout=0.2):
        super(MultiPointLSTM, self).__init__()
        
        self.input_dim = input_dim  # 输入特征数（点的数量）
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim else input_dim
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层 - 输出每个时间步的所有点的预测值
        self.fc = nn.Linear(hidden_dim, self.output_dim * pre_len)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 应用dropout
        last_output = self.dropout(last_output)
        
        # 全连接层
        output = self.fc(last_output)  # (batch_size, output_dim * pre_len)
        
        # 重新整形为 (batch_size, pre_len, output_dim)
        output = output.view(batch_size, pre_len, self.output_dim)
        
        return output

# 模型参数
lstm_model = MultiPointLSTM(
    input_dim=num_points,
    hidden_dim=1024,  # 增加隐藏层维度以处理多点数据
    output_dim=num_points,
    num_layers=10,    # 增加层数
    dropout=0.3
)

# 计算模型参数数量
total_params = sum(p.numel() for p in lstm_model.parameters())
print(f"模型总参数数量: {total_params:,}")

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100, verbose=True
)

epochs = 3000
Train = True  # 设置为训练模式

if Train:
    losses = []
    lstm_model.train()
    start_time = time.time()
    
    print("开始训练...")
    best_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            
            # 增加batch维度
            seq = seq.unsqueeze(0)      # (1, seq_len, num_points)
            labels = labels.unsqueeze(0)  # (1, pre_len, num_points)
            
            y_pred = lstm_model(seq)
            single_loss = loss_function(y_pred, labels)
            
            single_loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(single_loss.item())
        
        # 计算epoch平均损失
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': lstm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scalers': scalers,
                'target_columns': target_columns,
                'loss': best_loss,
                'epoch': epoch
            }, 'best_multipoint_model.pth')
        else:
            patience_counter += 1
            
        # 每50个epoch打印一次
        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.8f}, LR: {current_lr:.8f}')
            
        # 早停
        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 保存最终模型
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scalers': scalers,
        'target_columns': target_columns,
        'loss': avg_loss,
        'epoch': epoch
    }, 'final_multipoint_model.pth')
    
    print(f"训练完成，用时: {(time.time() - start_time) / 60:.4f} 分钟")
    
    # 绘制训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.title('Multi-Point LSTM Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('multipoint_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

# 测试阶段
print("\n开始测试...")
lstm_model.eval()

# 创建测试序列
test_inout_seq = create_multipoint_sequences(test_data_normalized, train_window, pre_len)
print(f"测试序列数量: {len(test_inout_seq)}")

# 存储所有预测结果
all_predictions = []
all_targets = []

with torch.no_grad():
    for seq, target in test_inout_seq:
        seq = seq.unsqueeze(0)
        prediction = lstm_model(seq)
        
        all_predictions.append(prediction.squeeze(0).numpy())
        all_targets.append(target.numpy())

# 转换为numpy数组
all_predictions = np.array(all_predictions)  # (num_samples, pre_len, num_points)
all_targets = np.array(all_targets)          # (num_samples, pre_len, num_points)

print(f"预测结果形状: {all_predictions.shape}")
print(f"真实值形状: {all_targets.shape}")

# 反归一化并计算各点的指标
all_metrics = {}
denorm_predictions = np.zeros_like(all_predictions)
denorm_targets = np.zeros_like(all_targets)

for i, col in enumerate(target_columns):
    # 反归一化
    pred_col = all_predictions[:, :, i].flatten().reshape(-1, 1)
    target_col = all_targets[:, :, i].flatten().reshape(-1, 1)
    
    pred_denorm = scalers[col].inverse_transform(pred_col).flatten()
    target_denorm = scalers[col].inverse_transform(target_col).flatten()
    
    denorm_predictions[:, :, i] = pred_denorm.reshape(all_predictions.shape[0], pre_len)
    denorm_targets[:, :, i] = target_denorm.reshape(all_targets.shape[0], pre_len)
    
    # 计算指标
    metrics = calculate_metrics(target_denorm, pred_denorm)
    all_metrics[col] = metrics
    
    print(f"\n{col} 列的预测指标:")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²: {metrics['R2']:.4f}")

# 计算整体平均指标
overall_metrics = {}
for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
    overall_metrics[metric] = np.mean([all_metrics[col][metric] for col in target_columns])
    overall_metrics[f'{metric}_std'] = np.std([all_metrics[col][metric] for col in target_columns])

print(f"\n整体平均指标:")
print(f"  MAE: {overall_metrics['MAE']:.4f} ± {overall_metrics['MAE_std']:.4f}")
print(f"  RMSE: {overall_metrics['RMSE']:.4f} ± {overall_metrics['RMSE_std']:.4f}")
print(f"  MAPE: {overall_metrics['MAPE']:.2f}% ± {overall_metrics['MAPE_std']:.2f}%")
print(f"  R²: {overall_metrics['R2']:.4f} ± {overall_metrics['R2_std']:.4f}")

start_index = train_len + train_window
end_index = start_index + len(test_inout_seq) * pre_len
time_indices = np.arange(start_index, end_index)

# 创建结果DataFrame
results_dfs = []

for i, col in enumerate(target_columns):
    # 提取该列的反归一化预测值和真实值
    pred_flat = denorm_predictions[:, :, i].flatten()
    target_flat = denorm_targets[:, :, i].flatten()
    
    # 确保长度匹配
    min_length = min(len(time_indices), len(pred_flat), len(target_flat))
    time_indices_col = time_indices[:min_length]
    pred_flat = pred_flat[:min_length]
    target_flat = target_flat[:min_length]
    
    # 创建该列的DataFrame
    col_df = pd.DataFrame({
        'Time_Index': time_indices_col,
        f'{col}_True': target_flat,
        f'{col}_Pred': pred_flat
    })
    results_dfs.append(col_df)

# 合并所有列的结果
result_df = pd.concat(results_dfs, axis=1)

# 移除重复的时间索引列
result_df = result_df.loc[:, ~result_df.columns.duplicated()]

# 保存到CSV
result_csv_path = 'prediction_results.csv'
result_df.to_csv(result_csv_path, index=False)
print(f"\n预测结果已保存至: {result_csv_path}")







# 可视化部分预测结果
fig, axes = plt.subplots(len(target_columns), 1, figsize=(15, 3*len(target_columns)))
if len(target_columns) == 1:
    axes = [axes]

for i, col in enumerate(target_columns):
    # 选择前50个样本进行可视化
    n_vis = min(50, len(denorm_predictions))
    
    pred_flat = denorm_predictions[:n_vis, :, i].flatten()
    target_flat = denorm_targets[:n_vis, :, i].flatten()
    
    axes[i].plot(target_flat, label='真实值', alpha=0.8)
    axes[i].plot(pred_flat, label='预测值', alpha=0.8)
    axes[i].set_title(f'{col} 列预测结果对比 (R²={all_metrics[col]["R2"]:.3f})')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig('multipoint_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()