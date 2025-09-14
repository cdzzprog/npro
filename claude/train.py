import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import gc
from tqdm import tqdm
import math
from scipy.signal import savgol_filter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def calculate_mae(real, pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(real - pred))

def calculate_rmse(real, pred):
    """计算均方根误差"""
    return np.sqrt(np.mean((real - pred) ** 2))

def calculate_mape(real, pred, epsilon=1e-8):
    """计算平均绝对百分比误差 (改进版本)"""
    # 避免分母为0的情况，使用更大的epsilon
    mask = np.abs(real) > epsilon
    if np.sum(mask) == 0:
        return 100.0  # 如果所有真实值都接近0，返回固定值
    
    mape_values = np.abs((real[mask] - pred[mask]) / real[mask]) * 100
    # 限制MAPE的最大值，避免异常值
    mape_values = np.clip(mape_values, 0, 1000)
    return np.mean(mape_values)

def calculate_r2(real, pred):
    """计算决定系数R²"""
    try:
        return r2_score(real, pred)
    except:
        return -999.0  # 如果计算失败返回标志值

def calculate_metrics(real, pred):
    """计算所有评价指标"""
    return {
        'MAE': calculate_mae(real, pred),
        'RMSE': calculate_rmse(real, pred),
        'MAPE': calculate_mape(real, pred),
        'R2': calculate_r2(real, pred)
    }

def robust_normalize_data(data):
    """稳健的数据标准化"""
    normalized_data = np.zeros_like(data)
    scalers = {}
    
    for i in tqdm(range(data.shape[1]), desc="Normalizing data"):
        column_data = data[:, i]
        
        # 检查数据的方差，如果方差过小使用不同的处理方法
        if np.var(column_data) < 1e-8:
            # 方差过小的列直接标准化为0
            normalized_data[:, i] = np.zeros_like(column_data)
            # 创建一个假的scaler用于反归一化
            scaler = StandardScaler()
            scaler.mean_ = np.mean(column_data)
            scaler.scale_ = 1.0
        else:
            # 使用StandardScaler进行标准化
            scaler = StandardScaler()
            normalized_data[:, i] = scaler.fit_transform(column_data.reshape(-1, 1)).flatten()
            # 限制标准化后的值在合理范围内
            normalized_data[:, i] = np.clip(normalized_data[:, i], -5, 5)
        
        scalers[i] = scaler
    
    return normalized_data, scalers

def handle_outliers_iqr(data, factor=1.5):
    """使用IQR方法处理异常值"""
    processed_data = data.copy()
    
    for i in range(data.shape[1]):
        column = data[:, i]
        Q1 = np.percentile(column, 25)
        Q3 = np.percentile(column, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # 将异常值替换为边界值而不是中位数
        processed_data[:, i] = np.clip(column, lower_bound, upper_bound)
    
    return processed_data

def create_time_features(length, features=['day_of_week', 'day_of_year']):
    """创建时间特征"""
    time_features = []
    
    for i in range(length):
        feature_vector = []
        
        if 'day_of_week' in features:
            day_of_week = i % 7
            feature_vector.extend([
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7)
            ])
        
        if 'day_of_year' in features:
            day_of_year = i % 365
            feature_vector.extend([
                np.sin(2 * np.pi * day_of_year / 365),
                np.cos(2 * np.pi * day_of_year / 365)
            ])
        
        time_features.append(feature_vector)
    
    return np.array(time_features)

# 简化但更有效的LSTM模型
class OptimizedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=None, num_layers=2, dropout=0.2):
        super(OptimizedLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim else input_dim
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 简化为单向LSTM
        )
        
        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, self.output_dim)
        
        # 残差连接的线性层
        if input_dim == self.output_dim:
            self.residual = True
        else:
            self.residual = False
            self.residual_proj = nn.Linear(input_dim, self.output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # 输入归一化
        x = self.input_norm(x)
        
        # LSTM处理
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 全连接层
        output = F.relu(self.fc1(last_output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        # 残差连接
        if self.residual:
            output = output + x[:, -1, :]  # 添加残差连接
        elif hasattr(self, 'residual_proj'):
            output = output + self.residual_proj(x[:, -1, :])
        
        return output.unsqueeze(1)  # 返回形状为 (batch_size, 1, output_dim)

def create_sequences(data, window_size, prediction_steps=1):
    """创建序列数据，支持多步预测"""
    sequences = []
    for i in range(len(data) - window_size - prediction_steps + 1):
        seq_x = data[i:i + window_size]
        seq_y = data[i + window_size:i + window_size + prediction_steps]
        sequences.append((seq_x, seq_y))
    return sequences

def train_model_with_validation(model, train_data, val_data, epochs, batch_size, device):
    """带验证集的训练函数"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 1500
    patience_counter = 0
    
    model.train()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_losses = []
        
        # 打乱训练数据
        np.random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch_x = []
            batch_y = []
            
            for j in range(i, min(i + batch_size, len(train_data))):
                x, y = train_data[j]
                batch_x.append(x)
                batch_y.append(y)
            
            batch_x = torch.stack(batch_x).to(device)
            batch_y = torch.stack(batch_y).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        # 验证阶段
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_x = []
                batch_y = []
                
                for j in range(i, min(i + batch_size, len(val_data))):
                    x, y = val_data[j]
                    batch_x.append(x)
                    batch_y.append(y)
                
                batch_x = torch.stack(batch_x).to(device)
                batch_y = torch.stack(batch_y).to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_losses.append(loss.item())
        
        # 计算平均损失
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch
            }, 'best_optimized_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses

# 主程序
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 读取数据
    data_path = r'E:\sar\csv\transposed_output2019_2022.csv'
    print(f"Reading data from: {data_path}")
    raw_data = pd.read_csv(data_path)
    
    # 选择数值列
    numeric_columns = [col for col in raw_data.columns if raw_data[col].dtype in ['float64', 'int64']]
    
    # 限制处理的列数以避免内存问题
    max_columns = min(1000, len(numeric_columns))  # 先用1000个点测试
    target_columns = numeric_columns[:max_columns]
    print(f"Selected {len(target_columns)} columns for training")
    
    # 提取数据
    data = raw_data[target_columns].values.astype(np.float32)
    print(f"Data shape: {data.shape}")
    
    # 检查并处理NaN值
    if np.isnan(data).any():
        print("Found NaN values, filling with interpolation...")
        data = pd.DataFrame(data).interpolate().fillna(method='bfill').fillna(method='ffill').values
    
    # 数据预处理
    print("Processing data...")
    
    # 1. 处理异常值
    data = handle_outliers_iqr(data, factor=2.0)
    
    # 2. 数据分割
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    n_train = int(len(data) * train_ratio)
    n_val = int(len(data) * val_ratio)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # 3. 数据标准化
    print("Normalizing data...")
    train_data_norm, scalers = robust_normalize_data(train_data)
    
    # 对验证集和测试集使用相同的scaler
    val_data_norm = np.zeros_like(val_data)
    test_data_norm = np.zeros_like(test_data)
    
    for i in range(data.shape[1]):
        scaler = scalers[i]
        if hasattr(scaler, 'scale_'):
            val_data_norm[:, i] = np.clip((val_data[:, i] - scaler.mean_) / scaler.scale_, -5, 5)
            test_data_norm[:, i] = np.clip((test_data[:, i] - scaler.mean_) / scaler.scale_, -5, 5)
        else:
            val_data_norm[:, i] = val_data[:, i]
            test_data_norm[:, i] = test_data[:, i]
    
    # 4. 创建序列
    window_size = 10
    prediction_steps = 1  # 预测下一个时间步
    
    print("Creating sequences...")
    train_sequences = create_sequences(train_data_norm, window_size, prediction_steps)
    val_sequences = create_sequences(val_data_norm, window_size, prediction_steps)
    test_sequences = create_sequences(test_data_norm, window_size, prediction_steps)
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    
    # 转换为张量
    def sequences_to_tensors(sequences):
        tensor_sequences = []
        for x, y in sequences:
            tensor_sequences.append((torch.FloatTensor(x), torch.FloatTensor(y)))
        return tensor_sequences
    
    train_sequences = sequences_to_tensors(train_sequences)
    val_sequences = sequences_to_tensors(val_sequences)
    test_sequences = sequences_to_tensors(test_sequences)
    
    # 5. 创建模型
    input_dim = data.shape[1]
    model = OptimizedLSTM(
        input_dim=input_dim,
        hidden_dim=256,  # 减小隐藏层维度
        output_dim=input_dim,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 6. 训练模型
    print("Starting training...")
    train_losses, val_losses = train_model_with_validation(
        model, train_sequences, val_sequences, 
        epochs=3000, batch_size=32, device=device
    )
    
    # 7. 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    # 8. 加载最佳模型并测试
    checkpoint = torch.load('best_optimized_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 9. 在测试集上评估
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for x, y in tqdm(test_sequences):
            x = x.unsqueeze(0).to(device)
            y = y.to(device)
            
            pred = model(x)
            
            all_predictions.append(pred.squeeze().cpu().numpy())
            all_targets.append(y.squeeze().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 10. 反归一化
    print("Denormalizing predictions...")
    pred_denorm = np.zeros_like(all_predictions)
    target_denorm = np.zeros_like(all_targets)
    
    for i in range(data.shape[1]):
        scaler = scalers[i]
        if hasattr(scaler, 'scale_'):
            pred_denorm[:, i] = all_predictions[:, i] * scaler.scale_ + scaler.mean_
            target_denorm[:, i] = all_targets[:, i] * scaler.scale_ + scaler.mean_
        else:
            pred_denorm[:, i] = all_predictions[:, i]
            target_denorm[:, i] = all_targets[:, i]
    
    # 11. 计算评估指标
    print("Calculating metrics...")
    overall_metrics = calculate_metrics(target_denorm.flatten(), pred_denorm.flatten())
    
    print("\n=== Overall Metrics ===")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 计算每个点的指标
    point_metrics = {}
    for i in range(min(10, data.shape[1])):  # 只显示前10个点的指标
        point_metrics[f'Point_{i}'] = calculate_metrics(target_denorm[:, i], pred_denorm[:, i])
    
    print("\n=== Sample Point Metrics (First 10 points) ===")
    for point, metrics in point_metrics.items():
        print(f"{point}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    # 12. 可视化结果
    plt.subplot(1, 2, 2)
    # 选择一个点进行可视化
    point_idx = 0
    plt.plot(target_denorm[:50, point_idx], label='True', alpha=0.8)
    plt.plot(pred_denorm[:50, point_idx], label='Predicted', alpha=0.8)
    plt.title(f'Prediction vs True (Point {point_idx})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTraining and evaluation completed!")
    print(f"Best model saved as 'best_optimized_model.pth'")
    print(f"Results plot saved as 'training_results.png'")
