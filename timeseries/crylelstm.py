import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import seaborn as snshuasa
# ... 保留原有的计算指标函数 (calculate_mae, calculate_rmse, etc.) ...
# 设置matplotlib支持中文显示 (使用兼容性更好的字体)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")

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
# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

# 读取数据
data_path = r'E:\sar\csv\transposed_output2019_2022.csv'
raw_data = pd.read_csv(data_path)

# 选择所有数值列
target_columns = [col for col in raw_data.columns if raw_data[col].dtype in ['float64', 'int64']]
print(f"总点数: {len(target_columns)}")
print(f"数据总长度: {len(raw_data)}")

# 参数设置
test_size = 0.15
train_size = 0.85
pre_len = 6
train_window = 10
batch_size = 10  # 每次处理的点数

# 计算训练集和测试集的分割点
train_len = int(train_size * len(raw_data))
test_len = len(raw_data) - train_len

# 初始化结果存储
all_results = pd.DataFrame()
all_metrics = {}

# 创建结果目录
os.makedirs('batch_results', exist_ok=True)
os.makedirs('batch_models', exist_ok=True)

# 循环处理每个批次
for batch_start in range(0, len(target_columns), batch_size):
    batch_end = min(batch_start + batch_size, len(target_columns))
    batch_columns = target_columns[batch_start:batch_end]
    num_points = len(batch_columns)
    
    print(f"\n处理批次: {batch_start+1}-{batch_end} ({num_points}个点)")
    
    # 提取批次数据
    batch_data = raw_data[batch_columns].values
    
    # 分割数据集
    train_data = batch_data[:train_len]
    test_data = batch_data[-test_len:]
    
    # 标准化
    scalers = {}
    train_data_normalized = np.zeros_like(train_data)
    test_data_normalized = np.zeros_like(test_data)
    
    for i, col in enumerate(batch_columns):
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_normalized[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
        test_data_normalized[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
        scalers[col] = scaler
    
    # 转为Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized)
    test_data_normalized = torch.FloatTensor(test_data_normalized)
    
    # 创建序列函数
    def create_sequences(input_data, tw, pre_len):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw - pre_len + 1):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + tw:i + tw + pre_len]
            inout_seq.append((train_seq, train_label))
        return inout_seq
    
    # 创建序列
    train_inout_seq = create_sequences(train_data_normalized, train_window, pre_len)
    print(f"训练序列数量: {len(train_inout_seq)}")
    
    # 定义LSTM模型
    class BatchLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, output_dim=None, num_layers=2, dropout=0.2):
            super(BatchLSTM, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.output_dim = output_dim if output_dim else input_dim
            
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.fc = nn.Linear(hidden_dim, self.output_dim * pre_len)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            batch_size, seq_len, _ = x.size()
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            
            lstm_out, _ = self.lstm(x, (h0, c0))
            last_output = lstm_out[:, -1, :]
            last_output = self.dropout(last_output)
            
            output = self.fc(last_output)
            output = output.view(batch_size, pre_len, self.output_dim)
            return output
    
    # 初始化模型
    lstm_model = BatchLSTM(
        input_dim=num_points,
        hidden_dim=128,
        output_dim=num_points,
        num_layers=2,
        dropout=0.3
    )
    
    # 训练参数
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )
    
    epochs = 1000
    best_loss = float('inf')
    patience = 100
    patience_counter = 0
    losses = []
    
    # 训练模型
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_losses = []
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            seq = seq.unsqueeze(0)
            labels = labels.unsqueeze(0)
            
            y_pred = lstm_model(seq)
            single_loss = loss_function(y_pred, labels)
            
            single_loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(single_loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(lstm_model.state_dict(), f'batch_models/best_model_batch_{batch_start}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停于轮次 {epoch+1}, 损失: {avg_loss:.6f}")
            break
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    print(f"批次训练完成, 用时: {(time.time() - start_time)/60:.2f} 分钟")
    
    # 测试模型
    lstm_model.load_state_dict(torch.load(f'batch_models/best_model_batch_{batch_start}.pth'))
    lstm_model.eval()
    
    test_inout_seq = create_sequences(test_data_normalized, train_window, pre_len)
    
    batch_predictions = []
    batch_targets = []
    
    with torch.no_grad():
        for seq, target in test_inout_seq:
            seq = seq.unsqueeze(0)
            prediction = lstm_model(seq)
            batch_predictions.append(prediction.squeeze(0).numpy())
            batch_targets.append(target.numpy())
    
    batch_predictions = np.array(batch_predictions)
    batch_targets = np.array(batch_targets)
    
    # 反标准化并保存结果
    start_index = train_len + train_window
    end_index = start_index + len(test_inout_seq) * pre_len
    time_indices = np.arange(start_index, end_index)
    
    batch_results = pd.DataFrame({'Time_Index': time_indices})
    
    for i, col in enumerate(batch_columns):
        pred_col = batch_predictions[:, :, i].flatten().reshape(-1, 1)
        target_col = batch_targets[:, :, i].flatten().reshape(-1, 1)
        
        pred_denorm = scalers[col].inverse_transform(pred_col).flatten()
        target_denorm = scalers[col].inverse_transform(target_col).flatten()
        
        # 保存到批次结果
        batch_results[f'{col}_True'] = target_denorm
        batch_results[f'{col}_Pred'] = pred_denorm
        
        # 计算指标
        metrics = calculate_metrics(target_denorm, pred_denorm)
        all_metrics[col] = metrics
    
    # 保存批次结果
    batch_results.to_csv(f'batch_results/batch_{batch_start}_results.csv', index=False)
    print(f"批次结果保存至: batch_results/batch_{batch_start}_results.csv")
    
    # 合并到总结果
    if all_results.empty:
        all_results = batch_results
    else:
        # 合并时只保留一个Time_Index列
        all_results = pd.merge(
            all_results, 
            batch_results.drop(columns=['Time_Index']), 
            left_index=True, 
            right_index=True,
            how='outer'
        )

# 保存最终结果
all_results.to_csv('final_predictions.csv', index=False)
print("所有批次处理完成! 最终结果保存至 final_predictions.csv")

# 计算并保存整体指标
metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
metrics_df.to_csv('point_metrics.csv')

# 计算平均指标
avg_metrics = {
    'MAE': metrics_df['MAE'].mean(),
    'RMSE': metrics_df['RMSE'].mean(),
    'MAPE': metrics_df['MAPE'].mean(),
    'R2': metrics_df['R2'].mean()
}

print("\n整体平均指标:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")

print("所有任务完成!")