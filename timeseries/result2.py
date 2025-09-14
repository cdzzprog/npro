import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import gc
from tqdm import tqdm
import math
from scipy.signal import savgol_filter
import seaborn as sns

# 设置matplotlib支持中文显示 (使用兼容性更好的字体)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def calculate_mae(real, pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(real - pred))

def calculate_rmse(real, pred):
    """计算均方根误差"""
    return np.sqrt(np.mean((real - pred) ** 2))

def calculate_mape(real, pred, epsilon=1e-6):
    """计算平均绝对百分比误差 (添加epsilon防止除零)"""
    # return np.mean(np.abs((real - pred) / (np.abs(real) + epsilon)) * 100
    return np.mean(np.abs((real - pred) / (np.abs(real) + epsilon)) * 100)
def calculate_r2(real, pred):
    """计算决定系数R²"""
    return r2_score(real, pred)

def calculate_metrics(real, pred):
    """计算所有评价指标"""
    return {
        'MAE': calculate_mae(real, pred),
        'RMSE': calculate_rmse(real, pred),
        'MAPE': calculate_mape(real, pred),
        'R2': calculate_r2(real, pred)
    }

def efficient_denormalize(data, scalers, target_columns):
    """高效反归一化数据"""
    denorm_data = np.zeros_like(data)
    for i, col in enumerate(tqdm(target_columns, desc="反归一化数据")):
        scaler = scalers[col]
        # 一次性处理所有时间步
        denorm_data[:, :, i] = scaler.inverse_transform(
            data[:, :, i].reshape(-1, 1)
        ).reshape(data.shape[0], data.shape[1])
    return denorm_data

def save_metrics_to_csv(metrics, filename):
    """保存指标到CSV文件"""
    # 处理不同类型的指标字典
    if isinstance(metrics, dict) and all(isinstance(v, dict) for v in metrics.values()):
        # 嵌套字典（如时间步或点指标）
        df = pd.DataFrame(metrics).T
    elif isinstance(metrics, dict) and all(isinstance(v, (int, float)) for v in metrics.values()):
        # 单层字典（如整体指标）
        df = pd.DataFrame([metrics])
    else:
        # 其他类型，创建空DataFrame
        df = pd.DataFrame()
    
    if not df.empty:
        df.to_csv(filename)
        print(f"指标已保存到 {filename}")
    else:
        print(f"无法保存指标到 {filename}，数据类型不支持")

def plot_metrics_by_timestep(metrics_by_step, filename):
    """绘制各时间步指标变化图"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Metrics by Timestep')
    
    # MAE
    axs[0, 0].plot([m['MAE'] for m in metrics_by_step.values()], 'o-')
    axs[0, 0].set_title('MAE Change')
    axs[0, 0].set_xlabel('Timestep')
    axs[0, 0].set_ylabel('MAE')
    axs[0, 0].grid(True)
    
    # RMSE
    axs[0, 1].plot([m['RMSE'] for m in metrics_by_step.values()], 'o-')
    axs[0, 1].set_title('RMSE Change')
    axs[0, 1].set_xlabel('Timestep')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].grid(True)
    
    # MAPE
    axs[1, 0].plot([m['MAPE'] for m in metrics_by_step.values()], 'o-')
    axs[1, 0].set_title('MAPE Change')
    axs[1, 0].set_xlabel('Timestep')
    axs[1, 0].set_ylabel('MAPE (%)')
    axs[1, 0].grid(True)
    
    # R2
    axs[1, 1].plot([m['R2'] for m in metrics_by_step.values()], 'o-')
    axs[1, 1].set_title('R² Change')
    axs[1, 1].set_xlabel('Timestep')
    axs[1, 1].set_ylabel('R²')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_point_metrics(point_metrics, filename):
    """绘制各点指标分布图"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Point Metrics Distribution')
    
    # 提取所有点的指标
    mae_values = [metrics['MAE'] for metrics in point_metrics.values()]
    rmse_values = [metrics['RMSE'] for metrics in point_metrics.values()]
    mape_values = [metrics['MAPE'] for metrics in point_metrics.values()]
    r2_values = [metrics['R2'] for metrics in point_metrics.values()]
    
    # MAE分布
    axs[0, 0].hist(mae_values, bins=50, alpha=0.7)
    axs[0, 0].set_title('MAE Distribution')
    axs[0, 0].set_xlabel('MAE')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].grid(True)
    
    # RMSE分布
    axs[0, 1].hist(rmse_values, bins=50, alpha=0.7)
    axs[0, 1].set_title('RMSE Distribution')
    axs[0, 1].set_xlabel('RMSE')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].grid(True)
    
    # MAPE分布
    axs[1, 0].hist(mape_values, bins=50, alpha=0.7)
    axs[1, 0].set_title('MAPE Distribution')
    axs[1, 0].set_xlabel('MAPE (%)')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].grid(True)
    
    # R2分布
    axs[1, 1].hist(r2_values, bins=50, alpha=0.7)
    axs[1, 1].set_title('R² Distribution')
    axs[1, 1].set_xlabel('R²')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def handle_outliers(data, threshold=3.5):
    """处理异常值"""
    # 使用中位数和IQR进行异常值处理
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # 将异常值替换为中位数
    data[(data < lower_bound) | (data > upper_bound)] = np.median(data)
    return data

def smooth_data(data, window_size=5, polyorder=3):
    """使用Savitzky-Golay滤波器平滑数据"""
    smoothed_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        # 对于短序列使用更小的窗口
        win_size = min(window_size, len(data))
        if win_size % 2 == 0:  # 窗口大小必须是奇数
            win_size -= 1
        if win_size > polyorder:  # 窗口大小必须大于多项式阶数
            smoothed_data[:, i] = savgol_filter(data[:, i], window_length=win_size, polyorder=polyorder)
        else:
            smoothed_data[:, i] = data[:, i]  # 如果窗口太小，保持原数据
    return smoothed_data

# def add_time_features(data):
#     """添加时间相关特征"""
#     time_features = np.zeros((len(data), 4))  # [sin_hour, cos_hour, sin_day, cos_day]
    
#     for i in range(len(data)):
#         # 假设每个时间点是小时级数据
#         hour = i % 24
#         day = (i // 24) % 7
        
#         time_features[i, 0] = np.sin(2 * np.pi * hour / 24)
#         time_features[i, 1] = np.cos(2 * np.pi * hour / 24)
#         time_features[i, 2] = np.sin(2 * np.pi * day / 7)
#         time_features[i, 3] = np.cos(2 * np.pi * day / 7)
    
#     return np.hstack([data, time_features])
def add_time_features(data):
    """添加时间相关特征（按天为单位）"""
    time_features = np.zeros((len(data), 4))  # [sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year]
    
    for i in range(len(data)):
        # 假设每个时间点是按天记录的
        day_of_week = i % 7  # 计算星期几
        day_of_year = i % 365  # 计算一年中的第几天
        
        # 生成时间特征
        time_features[i, 0] = np.sin(2 * np.pi * day_of_week / 7)  # 星期几的正弦
        time_features[i, 1] = np.cos(2 * np.pi * day_of_week / 7)  # 星期几的余弦
        time_features[i, 2] = np.sin(2 * np.pi * day_of_year / 365)  # 年中的第几天的正弦
        time_features[i, 3] = np.cos(2 * np.pi * day_of_year / 365)  # 年中的第几天的余弦
    
    return np.hstack([data, time_features])  # 将时间特征与原始数据拼接
def postprocess_predictions(predictions, targets, alpha=0.3):
    """应用后处理平滑"""
    # 使用指数平滑
    smoothed = np.zeros_like(predictions)
    smoothed[0] = predictions[0]
    
    for i in range(1, len(predictions)):
        smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i-1]
    
    # 确保预测值在合理范围内
    min_val = np.min(targets)
    max_val = np.max(targets)
    smoothed = np.clip(smoothed, min_val * 0.9, max_val * 1.1)
    
    return smoothed

def analyze_errors(pred, target):
    """分析预测误差分布"""
    errors = np.abs(pred - target)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=100)
    plt.title('Absolute Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    relative_errors = errors / (np.abs(target) + 1e-6)
    plt.hist(relative_errors, bins=100)
    plt.title('Relative Error Distribution')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300)
    plt.close()
    
    # 识别高误差点
    high_error_indices = np.where(errors > np.percentile(errors, 95))[0]
    print(f"High-error points: {len(high_error_indices)}")
    
    return high_error_indices

# 增强的LSTM模型架构
class EnhancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=None, num_layers=3, dropout=0.3):
        super(EnhancedLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim else input_dim
        
        # 输入卷积层
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, 
            num_heads=4,
            batch_first=True
        )
        
        # 输出层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, self.output_dim * pre_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # 输入卷积处理
        x = x.permute(0, 2, 1)  # (batch, channels, seq)
        conv_out = F.relu(self.conv1(x))
        conv_out = F.relu(self.conv2(conv_out))
        conv_out = self.pool(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, seq, channels)
        
        # LSTM处理
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(conv_out, (h0, c0))
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步
        last_output = attn_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 全连接层
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        
        # 重塑输出
        output = output.view(batch_size, pre_len, self.output_dim)
        
        return output

# 主程序
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 定义参数 (提前定义pre_len)
    global pre_len
    pre_len = 4
    
    # 读取数据
    data_path = r'E:\sar\csv\transposed_output2019_2022.csv'
    print(f"Reading data: {data_path}")
    raw_data = pd.read_csv(data_path)
    
    # 自动选择15000个数值列进行训练
    numeric_columns = [col for col in raw_data.columns if raw_data[col].dtype in ['float64', 'int64']]
    
    if len(numeric_columns) >= 15000:
        target_columns = numeric_columns[:15000]
        print(f"Selected {len(target_columns)} numeric columns for training")
    else:
        target_columns = numeric_columns
        print(f"Using all {len(target_columns)} numeric columns")
    
    print(f"Data length: {len(raw_data)}")
    
    # 数据预处理 - 只选择数值列
    multi_point_data = raw_data[target_columns].values.astype(np.float32)
    num_points = len(target_columns)
    print(f"Data shape: {multi_point_data.shape}")
    
    # 检查数据分布
    plt.figure(figsize=(12, 6))
    plt.hist(multi_point_data.flatten(), bins=100, log=True)
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency (log)')
    plt.savefig('data_distribution.png', dpi=300)
    plt.close()
    
    # 定义参数
    test_size = 0.15
    train_size = 0.85
    train_window = 10
    
    # 计算训练集和测试集的分割点
    train_len = int(train_size * len(multi_point_data))
    test_len = int(test_size * len(multi_point_data))
    
    train_data = multi_point_data[:train_len]
    test_data = multi_point_data[-test_len:]
    
    print(f"Train size: {train_data.shape}")
    print(f"Test size: {test_data.shape}")
    
    # 处理异常值
    print("Handling outliers...")
    for i in tqdm(range(train_data.shape[1]), desc="Processing outliers"):
        train_data[:, i] = handle_outliers(train_data[:, i])
        test_data[:, i] = handle_outliers(test_data[:, i])
    
    # 数据平滑处理
    print("Applying smoothing...")
    train_data = smooth_data(train_data, window_size=7, polyorder=2)
    test_data = smooth_data(test_data, window_size=7, polyorder=2)
    
    # 添加时间特征
    print("Adding time features...")
    train_data = add_time_features(train_data)
    test_data = add_time_features(test_data)
    num_points += 4  # 更新特征数量
    print(f"Features after adding time: {num_points}")
    
    # 对每个点分别进行标准化
    print("Standardizing data...")
    scalers = {}
    train_data_normalized = np.zeros_like(train_data)
    test_data_normalized = np.zeros_like(test_data)
    
    for i in tqdm(range(num_points), desc="Standardizing"):
        scaler = RobustScaler()
        train_data_normalized[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
        test_data_normalized[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
        scalers[i] = scaler
    
    print("Standardization complete")
    
    # 转化为Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized)
    test_data_normalized = torch.FloatTensor(test_data_normalized)
    
    def create_multipoint_sequences(input_data, tw, pre_len):
        """创建多点时间序列数据"""
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw - pre_len + 1):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + tw:i + tw + pre_len]
            inout_seq.append((train_seq, train_label))
        return inout_seq
    
    print("Creating training sequences...")
    train_inout_seq = create_multipoint_sequences(train_data_normalized, train_window, pre_len)
    print(f"Training sequences: {len(train_inout_seq)}")
    
    # 创建模型
    lstm_model = EnhancedLSTM(
        input_dim=num_points,
        hidden_dim=1024,
        output_dim=num_points,
        num_layers=3,
        dropout=0.3
    )
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    lstm_model = lstm_model.to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 使用Huber损失函数
    loss_function = nn.HuberLoss(delta=1.0)
    
    # 优化器和学习率调度
    learning_rate = 0.001
    optimizer = torch.optim.AdamW(
        lstm_model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4
    )
    
    # 使用余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=100,
        eta_min=1e-6
    )
    
    epochs = 500
    batch_size = 32
    Train = True
    
    if Train:
        losses = []
        lstm_model.train()
        start_time = time.time()
        
        print("Starting training...")
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # 打乱训练序列
            np.random.shuffle(train_inout_seq)
            
            # 批处理训练
            for i in range(0, len(train_inout_seq), batch_size):
                optimizer.zero_grad()
                
                batch_seq = []
                batch_labels = []
                
                # 构建批次
                for j in range(i, min(i + batch_size, len(train_inout_seq))):
                    seq, labels = train_inout_seq[j]
                    batch_seq.append(seq)
                    batch_labels.append(labels)
                
                # 转换为张量并移动到设备
                batch_seq = torch.stack(batch_seq).to(device)
                batch_labels = torch.stack(batch_labels).to(device)
                
                y_pred = lstm_model(batch_seq)
                single_loss = loss_function(y_pred, batch_labels)
                
                single_loss.backward()
                clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(single_loss.item())
            
            # 计算epoch平均损失
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            # 更新学习率
            scheduler.step()
            
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
                    'num_points': num_points,
                    'loss': best_loss,
                    'epoch': epoch
                }, 'best_enhanced_model.pth')
                print(f"Saved best model at epoch {epoch+1}, Loss: {best_loss:.8f}")
            else:
                patience_counter += 1
                
            # 每10个epoch打印一次
            if (epoch + 1) % 10 == 0:
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
            'num_points': num_points,
            'loss': avg_loss,
            'epoch': epoch
        }, 'final_enhanced_model.pth')
        
        print(f"Training completed in {(time.time() - start_time) / 60:.4f} minutes")
        
        # 绘制训练损失曲线
        plt.figure(figsize=(12, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 测试阶段
    print("\nStarting testing...")
    
    # 加载最佳模型
    model_path = 'best_enhanced_model.pth' if os.path.exists('best_enhanced_model.pth') else 'final_enhanced_model.pth'
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    scalers = checkpoint['scalers']
    target_columns = checkpoint['target_columns']
    num_points = checkpoint['num_points']
    print(f"Loaded model (Epoch {checkpoint['epoch']+1}, Loss: {checkpoint['loss']:.8f})")
    
    lstm_model.eval()
    
    # 创建测试序列
    test_inout_seq = create_multipoint_sequences(test_data_normalized, train_window, pre_len)
    print(f"Test sequences: {len(test_inout_seq)}")
    
    test_seq_count = len(test_inout_seq)  # 保存测试序列的数量
    print(f"Test sequences: {test_seq_count}")
    # 存储所有预测结果
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        # 批处理预测
        for i in tqdm(range(0, len(test_inout_seq), batch_size), desc="Testing"):
            batch_seq = []
            batch_targets = []
            
            for j in range(i, min(i + batch_size, len(test_inout_seq))):
                seq, target = test_inout_seq[j]
                batch_seq.append(seq)
                batch_targets.append(target)
            
            batch_seq = torch.stack(batch_seq).to(device)
            predictions = lstm_model(batch_seq)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(torch.stack(batch_targets).numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Targets shape: {all_targets.shape}")
    
    # 释放内存
    del test_inout_seq, test_data_normalized
    gc.collect()
    
    # 反归一化预测结果和真实值
    print("Denormalizing...")
    all_pred_denorm = efficient_denormalize(all_predictions, scalers, list(range(num_points)))
    all_target_denorm = efficient_denormalize(all_targets, scalers, list(range(num_points)))
    print("Denormalization complete")
    
    # 释放更多内存
    del all_predictions, all_targets
    gc.collect()
    
    # 应用后处理平滑
    print("Applying post-processing...")
    for i in range(all_pred_denorm.shape[2]):
        all_pred_denorm[:, :, i] = postprocess_predictions(
            all_pred_denorm[:, :, i], 
            all_target_denorm[:, :, i],
            alpha=0.2
        )
    
    # 1. 整体指标计算
    print("Calculating overall metrics...")
    all_pred_flat = all_pred_denorm.reshape(-1)
    all_target_flat = all_target_denorm.reshape(-1)
    
    overall_metrics = calculate_metrics(all_target_flat, all_pred_flat)
    print("\nOverall Metrics:")
    print(f"  MAE: {overall_metrics['MAE']:.6f}")
    print(f"  RMSE: {overall_metrics['RMSE']:.6f}")
    print(f"  MAPE: {overall_metrics['MAPE']:.4f}%")
    print(f"  R²: {overall_metrics['R2']:.6f}")
    
    # 保存整体指标
    save_metrics_to_csv(overall_metrics, 'overall_metrics.csv')
    
    # 2. 各时间步指标计算
    print("\nCalculating metrics by timestep...")
    metrics_by_step = {}
    for t in range(pre_len):
        step_pred = all_pred_denorm[:, t, :].reshape(-1)
        step_target = all_target_denorm[:, t, :].reshape(-1)
        
        step_metrics = calculate_metrics(step_target, step_pred)
        metrics_by_step[f"Step_{t+1}"] = step_metrics
        print(f"Timestep {t+1}: MAE={step_metrics['MAE']:.6f}, RMSE={step_metrics['RMSE']:.6f}, MAPE={step_metrics['MAPE']:.4f}%, R²={step_metrics['R2']:.6f}")
    
    # 保存时间步指标
    # save_metrics_to_csv(metrics_by_step, 'metrics_by_step.csv')
    
    # 绘制时间步指标变化图
    # plot_metrics_by_timestep(metrics_by_step, 'metrics_by_timestep.png')
    
    # # 3. 各点指标计算（只计算前1000个点，避免内存问题）
    # print("\nCalculating point metrics...")
    # point_metrics = {}
    # num_points_to_calculate = min(1000, num_points)  # 只计算前1000个点
    
    # for i in tqdm(range(num_points_to_calculate), desc="Point metrics"):
    #     point_pred = all_pred_denorm[:, :, i].reshape(-1)
    #     point_target = all_target_denorm[:, :, i].reshape(-1)
        
    #     point_metrics[f"Point_{i}"] = calculate_metrics(point_target, point_pred)
    
    # # 保存点指标
    # save_metrics_to_csv(point_metrics, 'point_metrics.csv')
    
    # # 绘制点指标分布图
    # plot_point_metrics(point_metrics, 'point_metrics_distribution.png')
    print("\nCalculating point metrics for all points...")
    point_metrics = {}
    num_points = len(target_columns)  # 所有点的数量

    # 预先分配数组以提高效率
    mae_values = np.zeros(num_points)
    rmse_values = np.zeros(num_points)
    mape_values = np.zeros(num_points)
    r2_values = np.zeros(num_points)

    for i in tqdm(range(num_points), desc="Point metrics"):
        point_pred = all_pred_denorm[:, :, i].reshape(-1)
        point_target = all_target_denorm[:, :, i].reshape(-1)
        
        metrics = calculate_metrics(point_target, point_pred)
        point_metrics[f"Point_{i}"] = metrics
        
        # 存储指标值用于平均计算
        mae_values[i] = metrics['MAE']
        rmse_values[i] = metrics['RMSE']
        mape_values[i] = metrics['MAPE']
        r2_values[i] = metrics['R2']

    # 计算所有点的平均指标
    avg_metrics = {
        'MAE': np.mean(mae_values),
        'RMSE': np.mean(rmse_values),
        'MAPE': np.mean(mape_values),
        'R2': np.mean(r2_values)
    }

    print("\nAverage Metrics for All Points:")
    print(f"  MAE: {avg_metrics['MAE']:.6f}")
    print(f"  RMSE: {avg_metrics['RMSE']:.6f}")
    print(f"  MAPE: {avg_metrics['MAPE']:.4f}%")
    print(f"  R²: {avg_metrics['R2']:.6f}")

    # 保存所有点的平均指标
    save_metrics_to_csv(avg_metrics, 'average_metrics_all_points.csv')

    # 保存点指标（所有点）
    save_metrics_to_csv(point_metrics, 'point_metrics_all.csv')



    # 4. 找出四项指标综合最好的10个点
    print("\nFinding top 10 points...")
    
    # 计算每个点的综合得分
    point_scores = []
    for i, metrics in point_metrics.items():
        # 综合得分 = R² - (归一化的MAE + RMSE + MAPE/100)
        score = metrics['R2'] - (metrics['MAE'] / 10 + metrics['RMSE'] / 10 + metrics['MAPE'] / 1000)
        point_scores.append({
            'point_index': int(i.split('_')[1]),
            'score': score,
            'R2': metrics['R2'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE']
        })
    
    # 按综合得分排序
    point_scores_sorted = sorted(point_scores, key=lambda x: x['score'], reverse=True)
    top_10_points = point_scores_sorted[:10]
    
    # 输出最好的10个点
    print("\nTop 10 Points:")
    print("Rank | Point |   MAE   |  RMSE   |  MAPE   |   R²    | Score")
    for i, point in enumerate(top_10_points):
        print(f"{i+1:2d}   | {point['point_index']:6d} | {point['MAE']:.6f} | {point['RMSE']:.6f} | {point['MAPE']:.4f}% | {point['R2']:.6f} | {point['score']:.6f}")
    
    # 保存最好的10个点结果
    top_points_df = pd.DataFrame(top_10_points)
    top_points_df.to_csv('top_10_points.csv', index=False)
    print("Top points saved to top_10_points.csv")
    
    # 5. 可视化最好的10个点的预测结果
    print("\nVisualizing top points...")
    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    axes = axes.flatten()
    
    for i, point in enumerate(top_10_points):
        point_idx = point['point_index']
        
        # 选择前50个样本进行可视化
        n_vis = min(50, len(all_pred_denorm))
        
        pred_flat = all_pred_denorm[:n_vis, :, point_idx].flatten()
        target_flat = all_target_denorm[:n_vis, :, point_idx].flatten()
        
        # 绘制预测对比
        ax = axes[i]
        ax.plot(target_flat, label='True', alpha=0.8, linewidth=2)
        ax.plot(pred_flat, label='Pred', alpha=0.8, linestyle='--')
        
        # 添加指标信息
        metrics_text = (f"MAE: {point['MAE']:.4f}\n"
                       f"RMSE: {point['RMSE']:.4f}\n"
                       f"MAPE: {point['MAPE']:.2f}%\n"
                       f"R²: {point['R2']:.4f}")
        ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
        
        ax.set_title(f'Point {point_idx} (Rank: {i+1})')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('top_10_points_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化一个点的多时间步预测
    print("Visualizing best point...")
    best_point = top_10_points[0]
    point_idx = best_point['point_index']
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 选择前20个样本
    n_vis = min(20, len(all_pred_denorm))
    
    for t in range(pre_len):
        # 真实值和预测值
        target_vals = all_target_denorm[:n_vis, t, point_idx]
        pred_vals = all_pred_denorm[:n_vis, t, point_idx]
        
        # 绘制每个时间步的预测
        ax.plot(np.arange(t, n_vis * pre_len, pre_len), target_vals, 'o-', 
                label=f'Timestep {t+1} True', alpha=0.8)
        ax.plot(np.arange(t, n_vis * pre_len, pre_len), pred_vals, 'x--', 
                label=f'Timestep {t+1} Pred', alpha=0.8)
    
    # 添加指标信息
    metrics_text = (f"MAE: {best_point['MAE']:.4f}\n"
                   f"RMSE: {best_point['RMSE']:.4f}\n"
                   f"MAPE: {best_point['MAPE']:.2f}%\n"
                   f"R²: {best_point['R2']:.4f}")
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'Best Point {point_idx} Predictions')
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    plt.savefig('best_point_multistep_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 误差分析
    print("Analyzing errors...")
    high_error_points = analyze_errors(all_pred_flat, all_target_flat)
    
    # print("\nTraining and testing completed!")
    # print(f"Trained model for {num_points} points")
    # print("All metrics calculated on original scale")
        # ===================== 新增部分：保存预测结果和原始真实值 =====================
# ===================== 修改后的保存预测结果部分 =====================
    print("\nSaving prediction results with original true values...")

    # 重新读取原始数据以获取原始值
    print("Reloading original data...")
    raw_data = pd.read_csv(data_path)
    if 'Time_Index' in raw_data.columns:
        raw_data = raw_data.drop(columns=['Time_Index'])
        
    # 计算测试集中预测部分对应的原始索引
    start_index = train_len + train_window
    max_end_index = len(raw_data)  # 原始数据的总长度
    end_index = min(start_index + test_seq_count * pre_len, max_end_index)  # 确保不超过原始数据长度
    time_indices = np.arange(start_index, end_index)

    # 只处理原始特征，不包括添加的时间特征
    num_original_points = len(target_columns)

    # 创建结果DataFrame
    results_dfs = []
    column_names = []

    # 只保存前1000个点的结果以控制文件大小
    num_points_to_save = min(1000, num_original_points)

    # 从原始数据中提取真实值
    original_values = raw_data[target_columns].values.astype(np.float32)

    print(f"Original data length: {len(original_values)}")
    print(f"Time indices range: {start_index} to {end_index}")

    # 调整预测结果数组以匹配时间索引长度
    if end_index - start_index < all_pred_denorm.shape[0] * all_pred_denorm.shape[1]:
        # 计算需要保留的序列数
        num_sequences = (end_index - start_index) // pre_len
        # 调整预测结果数组
        all_pred_denorm = all_pred_denorm[:num_sequences]
        all_target_denorm = all_target_denorm[:num_sequences]
        print(f"Adjusted prediction array to {num_sequences} sequences")

    for i in tqdm(range(num_points_to_save), desc="Saving points"):
        col_name = target_columns[i]
        
        # 从原始数据中提取该列的真实值
        col_original_values = original_values[time_indices, i]
        
        # 提取该列的反归一化预测值
        # 首先将预测结果展平
        pred_flat = all_pred_denorm[:, :, i].flatten()
        
        # 确保长度匹配
        min_length = min(len(time_indices), len(pred_flat))
        time_indices_col = time_indices[:min_length]
        pred_flat = pred_flat[:min_length]
        true_flat = col_original_values[:min_length]
        
        # 创建该列的DataFrame
        col_df = pd.DataFrame({
            'Time_Index': time_indices_col,
            f'{col_name}_True': true_flat,
            f'{col_name}_Pred': pred_flat
        })
        results_dfs.append(col_df)
        column_names.append(col_name)

    # 合并所有列的结果
    result_df = pd.concat(results_dfs, axis=1)

    # 移除重复的时间索引列
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    # 保存到CSV
    result_csv_path = 'prediction_results_with_original_values.csv'
    result_df.to_csv(result_csv_path, index=False)
    print(f"\n包含原始真实值的预测结果已保存至: {result_csv_path}")

    # 另外保存每个点的指标
    point_metrics_df = pd.DataFrame({
        'Point_Index': list(range(num_points_to_save)),
        'Column_Name': column_names,
        'MAE': [point_metrics[f"Point_{i}"]['MAE'] for i in range(num_points_to_save)],
        'RMSE': [point_metrics[f"Point_{i}"]['RMSE'] for i in range(num_points_to_save)],
        'MAPE': [point_metrics[f"Point_{i}"]['MAPE'] for i in range(num_points_to_save)],
        'R2': [point_metrics[f"Point_{i}"]['R2'] for i in range(num_points_to_save)]
    })
    point_metrics_df.to_csv('point_metrics_with_names.csv', index=False)
    print("包含列名的点指标已保存至: point_metrics_with_names.csv")

    # 释放内存
    del all_pred_denorm, all_target_denorm, result_df, raw_data, original_values
    gc.collect()