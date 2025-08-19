import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.signal import savgol_filter
import os
import gc

def calculate_mae(real, pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(real - pred))

def calculate_rmse(real, pred):
    """计算均方根误差"""
    return np.sqrt(np.mean((real - pred) ** 2))

def calculate_mape(real, pred):
    """计算平均绝对百分比误差"""
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

# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

# 读取数据
data_path = r'E:\sar\csv\transposed_output2019_2022.csv'  # 填你自己的数据地址
raw_data = pd.read_csv(data_path)

# 选择要训练的列（可以选择多个点）
target_columns = [col for col in raw_data.columns if raw_data[col].dtype in ['float64', 'int64']]

print(f"选择的训练列: {len(target_columns)}个点")
print(f"数据总长度: {len(raw_data)}个时间步")

# 数据预处理
multi_point_data = raw_data[target_columns].values.astype(np.float32)  # 使用float32节省内存
num_points = len(target_columns)
print(f"数据形状: {multi_point_data.shape} (时间步×点数)")

# 应用Savitzky-Golay滤波器平滑数据
print("应用Savitzky-Golay滤波器平滑数据...")
window_length = min(11, len(multi_point_data) // 10 * 2 + 1)  # 确保是奇数
if window_length < 5:
    window_length = 5
polyorder = 3

# 只平滑训练集部分
train_len = int(0.85 * len(multi_point_data))
for i in range(num_points):
    train_part = multi_point_data[:train_len, i]
    smoothed_train = savgol_filter(train_part, window_length=window_length, polyorder=polyorder)
    multi_point_data[:len(smoothed_train), i] = smoothed_train

# 定义参数
test_size = 0.15
train_size = 0.85
pre_len = 6
train_window = 10

# 计算训练集和测试集的分割点
train_data = multi_point_data[:train_len]
test_data = multi_point_data[train_len:]

print(f"训练集尺寸: {train_data.shape}")
print(f"测试集尺寸: {test_data.shape}")

# 对每个点分别进行标准化
scalers = {}
train_data_normalized = np.zeros_like(train_data, dtype=np.float32)
test_data_normalized = np.zeros_like(test_data, dtype=np.float32)

for i, col in enumerate(target_columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_normalized[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
    test_data_normalized[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
    scalers[col] = scaler

# 检查是否有NaN值
if np.isnan(train_data_normalized).any() or np.isnan(test_data_normalized).any():
    print("警告: 数据中存在NaN值! 进行填充处理...")
    train_data_normalized = np.nan_to_num(train_data_normalized, nan=0.0)
    test_data_normalized = np.nan_to_num(test_data_normalized, nan=0.0)

# 释放内存
del train_data, test_data, multi_point_data
gc.collect()

# 转化为深度学习模型需要的类型Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized)
test_data_normalized = torch.FloatTensor(test_data_normalized)

def create_multipoint_sequences(input_data, tw, pre_len):
    """创建多点时间序列数据"""
    inout_seq = []
    L = len(input_data)
    # 使用步长为1，最大化利用有限数据
    for i in range(0, L - tw - pre_len + 1, 1):
        train_seq = input_data[i:i + tw]  # (tw, num_points)
        train_label = input_data[i + tw:i + tw + pre_len]  # (pre_len, num_points)
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 定义训练器的输入
train_inout_seq = create_multipoint_sequences(train_data_normalized, train_window, pre_len)
print(f"训练序列数量: {len(train_inout_seq)} (从{len(train_data_normalized)}个时间步创建)")

class LightweightLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=None, num_layers=3, dropout=0.1):
        super(LightweightLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim else input_dim
        
        # 使用小型LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 简化全连接层
        self.fc = nn.Linear(hidden_dim, self.output_dim)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化权重，防止梯度爆炸"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)
        
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
        
        # 全连接层 - 为每个点单独预测
        predictions = []
        for _ in range(pre_len):
            predictions.append(self.fc(last_output))
        
        # 组合预测结果 (batch_size, pre_len, output_dim)
        return torch.stack(predictions, dim=1)

# 模型参数 - 使用轻量级模型
lstm_model = LightweightLSTM(
    input_dim=num_points,
    hidden_dim=128,  # 大幅减小隐藏层维度
    output_dim=num_points,
    num_layers=3,   # 单层LSTM
    dropout=0.1
)

# 计算模型参数数量
total_params = sum(p.numel() for p in lstm_model.parameters())
print(f"模型总参数数量: {total_params:,}")

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0001, weight_decay=1e-6)  # 更低的学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, verbose=True, min_lr=1e-7
)

epochs = 3000
Train = True  # 设置为训练模式

if Train:
    losses = []
    lstm_model.train()
    start_time = time.time()
    
    print("开始训练...")
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    # 准备数据
    train_seqs = torch.stack([seq for seq, _ in train_inout_seq])
    train_labels = torch.stack([label for _, label in train_inout_seq])
    
    # 使用小批量训练
    batch_size = min(16, len(train_seqs))  # 动态调整批量大小
    print(f"使用批量大小: {batch_size}")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # 手动小批量训练
        num_batches = (len(train_seqs) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_seqs))
            
            batch_seqs = train_seqs[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # 前向传播
            y_pred = lstm_model(batch_seqs)
            single_loss = loss_function(y_pred, batch_labels)
            
            # 检查损失是否为NaN
            if torch.isnan(single_loss):
                print(f"检测到NaN损失，跳过该批次")
                continue
                
            single_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=0.1)
            optimizer.step()
            
            epoch_losses.append(single_loss.item())
        
        # 计算epoch平均损失
        if not epoch_losses:
            print("所有批次损失均为NaN，停止训练")
            avg_loss = float('inf')  # 防止未定义
            break
            
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
            
        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.8f}, LR: {current_lr:.8f}')
            
        # 早停
        if patience_counter >= patience:
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

# 手动小批量测试
test_seqs = torch.stack([seq for seq, _ in test_inout_seq])
test_labels = torch.stack([label for _, label in test_inout_seq])
test_batch_size = min(16, len(test_seqs))

with torch.no_grad():
    num_batches = (len(test_seqs) + test_batch_size - 1) // test_batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * test_batch_size
        end_idx = min((batch_idx + 1) * test_batch_size, len(test_seqs))
        
        batch_seqs = test_seqs[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]
        
        prediction = lstm_model(batch_seqs)
        all_predictions.append(prediction.numpy())
        all_targets.append(batch_labels.numpy())

# 合并批次结果
if all_predictions:
    all_predictions = np.concatenate(all_predictions, axis=0)  # (num_samples, pre_len, num_points)
    all_targets = np.concatenate(all_targets, axis=0)          # (num_samples, pre_len, num_points)
else:
    print("警告: 没有预测结果!")
    all_predictions = np.zeros((0, pre_len, num_points))
    all_targets = np.zeros((0, pre_len, num_points))

print(f"预测结果形状: {all_predictions.shape}")
print(f"真实值形状: {all_targets.shape}")

# 反归一化并计算各点的指标
all_metrics = {}
denorm_predictions = np.zeros_like(all_predictions)
denorm_targets = np.zeros_like(all_targets)

# 只处理前100个点，避免过多计算
max_points_to_process = min(100, num_points)
print(f"只处理前{max_points_to_process}个点以节省时间...")

for i in range(max_points_to_process):
    col = target_columns[i]
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
    
    # 只打印部分点的指标
    if i % 10 == 0:
        print(f"\n{col} 列的预测指标:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")

# 计算整体平均指标
if all_metrics:
    overall_metrics = {}
    for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
        metric_values = [all_metrics[col][metric] for col in list(all_metrics.keys())[:max_points_to_process]]
        # 过滤掉无效值
        valid_metrics = [v for v in metric_values if not np.isinf(v) and not np.isnan(v)]
        if valid_metrics:
            overall_metrics[metric] = np.mean(valid_metrics)
            overall_metrics[f'{metric}_std'] = np.std(valid_metrics)
        else:
            overall_metrics[metric] = np.nan
            overall_metrics[f'{metric}_std'] = np.nan

    print(f"\n整体平均指标 (基于{max_points_to_process}个点):")
    print(f"  MAE: {overall_metrics['MAE']:.4f} ± {overall_metrics['MAE_std']:.4f}")
    print(f"  RMSE: {overall_metrics['RMSE']:.4f} ± {overall_metrics['RMSE_std']:.4f}")
    print(f"  MAPE: {overall_metrics['MAPE']:.2f}% ± {overall_metrics['MAPE_std']:.2f}%")
    print(f"  R²: {overall_metrics['R2']:.4f} ± {overall_metrics['R2_std']:.4f}")
else:
    print("\n没有有效指标可计算")

# 保存预测结果和原始真实值
start_index = train_len + train_window
end_index = start_index + len(test_inout_seq) * pre_len
time_indices = np.arange(start_index, end_index)

# 创建结果DataFrame
results_dfs = []

# 只保存前100个点
for i in range(min(100, num_points)):
    col = target_columns[i]
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
if results_dfs:
    result_df = pd.concat(results_dfs, axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    # 保存到CSV
    result_csv_path = 'prediction_results.csv'
    result_df.to_csv(result_csv_path, index=False)
    print(f"\n预测结果已保存至: {result_csv_path}")
else:
    print("\n没有预测结果可保存")

# 可视化部分预测结果
num_plots = min(3, max_points_to_process)  # 只绘制前3个点
if num_plots > 0:
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3*num_plots))
    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        col = target_columns[i]
        pred_flat = denorm_predictions[:, :, i].flatten()
        target_flat = denorm_targets[:, :, i].flatten()
        
        # 选择前50个样本进行可视化
        n_vis = min(50, len(pred_flat))
        
        axes[i].plot(target_flat[:n_vis], label='真实值', alpha=0.8)
        axes[i].plot(pred_flat[:n_vis], label='预测值', alpha=0.8)
        axes[i].set_title(f'{col} 列预测结果对比')
        axes[i].set_xlabel('时间步')
        axes[i].set_ylabel('值')
        axes[i].legend()
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.savefig('multipoint_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("没有足够数据生成可视化图表")

print("所有任务完成！")