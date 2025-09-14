
# import time
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
 
# np.random.seed(0)
 
 
# def calculate_mae(y_true, y_pred):
#     # 平均绝对误差
#     mae = np.mean(np.abs(y_true - y_pred))
#     return mae
 
 
# true_data = pd.read_csv(r'E:\sar\csv\transposed_output2019all.csv') # 填你自己的数据地址
 
# target = '6'
 
 
# # 这里加一些数据的预处理, 最后需要的格式是pd.series
 
# true_data = np.array(true_data['6'])
 
# # 定义窗口大小
# test_data_size = 32
# # 训练集和测试集的尺寸划分
# test_size = 0.4
# train_size = 0.6
# # 标准化处理
# scaler_train = MinMaxScaler(feature_range=(0, 1))
# scaler_test = MinMaxScaler(feature_range=(0, 1))
# train_data = true_data[:int(train_size * len(true_data))]
# test_data = true_data[-int(test_size * len(true_data)):]
# print("训练集尺寸:", len(train_data))
# print("测试集尺寸:", len(test_data))

# train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
# test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))
# # 转化为深度学习模型需要的类型Tensor
# train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
# test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)
# print("训练集归一化后的尺寸:", test_data_normalized.shape)
 
# def create_inout_sequences(input_data, tw, pre_len):
#     inout_seq = []
#     L = len(input_data)
#     for i in range(L - tw):
#         train_seq = input_data[i:i + tw]
#         if (i + tw + 4) > len(input_data):
#             break
#         train_label = input_data[i + tw:i + tw + pre_len]
#         inout_seq.append((train_seq, train_label))
#     return inout_seq
 
# pre_len = 4
# train_window = 16
# # 定义训练器的的输入
# train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len)
 
 
# class LSTM(nn.Module):
#     def __init__(self, input_dim=1, hidden_dim=350, output_dim=1):
#         super(LSTM, self).__init__()
 
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
 
#     def forward(self, x):
#         x = x.unsqueeze(1)
 
#         h0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)
#         c0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)
 
#         out, _ = self.lstm(x, (h0_lstm, c0_lstm))
#         out = out[:, -1]
#         out = self.fc(out)
 
#         return out
 
 
# lstm_model = LSTM(input_dim=1, output_dim=pre_len, hidden_dim=train_window)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
# epochs = 100
# Train = False # 训练还是预测
 
# if Train:
#     losss = []
#     lstm_model.train()  # 训练模式
#     start_time = time.time()  # 计算起始时间
#     for i in range(epochs):
#         for seq, labels in train_inout_seq:
#             lstm_model.train()
#             optimizer.zero_grad()
 
#             y_pred = lstm_model(seq)
 
#             single_loss = loss_function(y_pred, labels)
 
#             single_loss.backward()
#             optimizer.step()
#             print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
#             losss.append(single_loss.detach().numpy())
#     torch.save(lstm_model.state_dict(), 'save_model.pth')
#     print(f"模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")
#     plt.plot(losss)
#     # 设置图表标题和坐标轴标签
#     plt.title('Training Error')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     # 保存图表到本地
#     plt.savefig('training_error.png')
# else:
#     # 加载模型进行预测
#     lstm_model.load_state_dict(torch.load('save_model.pth'))
#     lstm_model.eval()  # 评估模式
#     # results = []
#     # reals = []
#     # losss = []
#     test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)
#     # print("测试集尺寸:", len(test_inout_seq))
#     # print("开始预测...")    
#     # for seq, labels in test_inout_seq:
#     #     pred = lstm_model(seq)[0].item()
#     #     results.append(pred)
#     #     mae = calculate_mae(pred, labels.detach().numpy())  # MAE误差计算绝对值(预测值  - 真实值)
#     #     reals.append(labels.detach().numpy())
#     #     losss.append(mae)
 
#     # print("模型预测结果：", results)
#     # print("预测误差MAE:", losss)
 
#     # plt.style.use('ggplot')
 
#     # # 创建折线图
#     # plt.plot(results, label='real', color='blue')  # 实际值
#     # plt.plot(reals, label='LSTM', color='red', linestyle='--')  # 预测值
 
#     # # 增强视觉效果
#     # # plt.grid(True)
#     # plt.title('real vs LSTM')
#     # plt.xlabel('time')
#     # plt.ylabel('value')
#     # plt.legend()
#     # plt.savefig('test——results.png')
#     results = []
#     reals = []

#     for seq, labels in test_inout_seq:
#         pred = lstm_model(seq)[0].item()  # 获取当前的预测值
#         results.append(pred)  # 只记录当前的预测值
#         reals.append(labels.detach().numpy())  # 只记录当前的真实值

#     # 绘制预测曲线与真实曲线
#     plt.plot(results, label='LSTM Predictions', color='red', linestyle='--')  # 预测值
#     plt.plot([item[0] for item in reals], label='Real Values', color='blue')  # 实际值
#     plt.grid(True)
#     # 增强视觉效果
#     plt.title('Real vs LSTM Predictions')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.savefig('test_results.png')


import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
 
 
def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
 
 
# 读取数据
true_data = pd.read_csv(r'E:\sar\csv\transposed_output2019all.csv') # 填你自己的数据地址
 
target = '6'
 
# 数据预处理
true_data = np.array(true_data['6'])
 
# 定义参数
test_data_size = 32
test_size = 0.4
train_size = 0.6
 
# 标准化处理
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))
train_data = true_data[:int(train_size * len(true_data))]
test_data = true_data[-int(test_size * len(true_data)):]
print("训练集尺寸:", len(train_data))
print("测试集尺寸:", len(test_data))

train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))

# 转化为深度学习模型需要的类型Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)
print("测试集归一化后的尺寸:", test_data_normalized.shape)
 
def create_inout_sequences(input_data, tw, pre_len):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - pre_len + 1):  # 修正边界条件
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq
 
pre_len = 4
train_window = 30

# 定义训练器的输入
train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len)
print("训练序列数量:", len(train_inout_seq))
 
 
class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=350, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
 
    def forward(self, x):
        # x shape: (batch_size, seq_len) -> (batch_size, seq_len, input_dim)
        x = x.unsqueeze(-1)
 
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
 
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
 
        return out
 
 
# 模型参数
lstm_model = LSTM(input_dim=1, output_dim=pre_len, hidden_dim=64, num_layers=2)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)

epochs = 4000  # 修改为10000轮训练
Train = False # 设置为训练模式
 
if Train:
    losses = []
    lstm_model.train()
    start_time = time.time()
    
    print("开始训练...")
    best_loss = float('inf')
    patience = 500
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            
            # 增加batch维度
            seq = seq.unsqueeze(0)
            labels = labels.unsqueeze(0)
            
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
        scheduler.step()
        
        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(lstm_model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        # 每100个epoch打印一次
        if (epoch + 1) % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.8f}, LR: {current_lr:.6f}')
            
        # 早停
        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 保存最终模型
    torch.save(lstm_model.state_dict(), 'final_model.pth')
    print(f"训练完成，用时: {(time.time() - start_time) / 60:.4f} 分钟")
    
    # 绘制训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# else:
#     # 预测模式
#     try:
#         lstm_model.load_state_dict(torch.load('best_model.pth'))
#         print("加载最佳模型")
#     except:
#         lstm_model.load_state_dict(torch.load('final_model.pth'))
#         print("加载最终模型")
        
#     lstm_model.eval()
    
#     test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)
#     print("测试序列数量:", len(test_inout_seq))
    
#     results = []
#     reals = []
#     maes = []
    
#     print("开始预测...")
#     with torch.no_grad():
#         for seq, labels in test_inout_seq:
#             seq = seq.unsqueeze(0)  # 添加batch维度
            
#             pred = lstm_model(seq)
#             pred_values = pred.squeeze(0).numpy()
#             real_values = labels.numpy()
            
#             results.append(pred_values)
#             reals.append(real_values)
            
#             # 计算MAE
#             mae = calculate_mae(real_values, pred_values)
#             maes.append(mae)
    
#     # 将预测结果展平用于可视化
#     results_flat = np.concatenate(results)
#     reals_flat = np.concatenate(reals)
    
#     print(f"平均MAE: {np.mean(maes):.6f}")
#     print(f"MAE标准差: {np.std(maes):.6f}")
    
#     # 反归一化
#     results_denorm = scaler_test.inverse_transform(results_flat.reshape(-1, 1)).flatten()
#     reals_denorm = scaler_test.inverse_transform(reals_flat.reshape(-1, 1)).flatten()
    
#     # 绘制预测结果
#     plt.figure(figsize=(15, 8))
#     plt.plot(reals_denorm[:200], label='Real Values', color='blue', linewidth=2)
#     plt.plot(results_denorm[:200], label='LSTM Predictions', color='red', linestyle='--', linewidth=2)
#     plt.grid(True, alpha=0.3)
#     plt.title('Real vs LSTM Predictions (First 200 points)', fontsize=14)
#     plt.xlabel('Time Steps', fontsize=12)
#     plt.ylabel('Value', fontsize=12)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # 绘制误差分布
#     plt.figure(figsize=(10, 6))
#     plt.hist(maes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
#     plt.title('MAE Distribution')
#     plt.xlabel('MAE')
#     plt.ylabel('Frequency')
#     plt.grid(True, alpha=0.3)
#     plt.savefig('mae_distribution.png', dpi=300, bbox_inches='tight')
#     plt.show()

else:
    # 预测模式
    try:
        lstm_model.load_state_dict(torch.load('best_model.pth'))
        print("加载最佳模型")
    except:
        lstm_model.load_state_dict(torch.load('final_model.pth'))
        print("加载最终模型")
        
    lstm_model.eval()
    
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)
    print("测试序列数量:", len(test_inout_seq))
    
    results = []
    reals = []
    all_metrics = []
    
    print("开始预测...")
    with torch.no_grad():
        for seq, labels in test_inout_seq:
            seq = seq.unsqueeze(0)  # 添加batch维度
            
            pred = lstm_model(seq)
            pred_values = pred.squeeze(0).numpy()
            real_values = labels.numpy()
            
            results.append(pred_values)
            reals.append(real_values)
            
            # 计算各种指标
            metrics = calculate_metrics(real_values, pred_values)
            all_metrics.append(metrics)
    
    # 将预测结果展平用于可视化和整体评估
    results_flat = np.concatenate(results)
    reals_flat = np.concatenate(reals)
    
    # 计算整体指标
    overall_metrics = calculate_metrics(reals_flat, results_flat)
    
    # 计算各指标的平均值和标准差
    mae_values = [m['MAE'] for m in all_metrics]
    rmse_values = [m['RMSE'] for m in all_metrics]
    mape_values = [m['MAPE'] for m in all_metrics if not np.isinf(m['MAPE'])]
    r2_values = [m['R2'] for m in all_metrics]
    
    print("\n=== 模型评价指标 ===")
    print(f"整体MAE:  {overall_metrics['MAE']:.6f}")
    print(f"整体RMSE: {overall_metrics['RMSE']:.6f}")
    print(f"整体MAPE: {overall_metrics['MAPE']:.4f}%")
    print(f"整体R²:   {overall_metrics['R2']:.6f}")
    
    print("\n=== 各序列指标统计 ===")
    print(f"MAE  - 均值: {np.mean(mae_values):.6f}, 标准差: {np.std(mae_values):.6f}")
    print(f"RMSE - 均值: {np.mean(rmse_values):.6f}, 标准差: {np.std(rmse_values):.6f}")
    if mape_values:
        print(f"MAPE - 均值: {np.mean(mape_values):.4f}%, 标准差: {np.std(mape_values):.4f}%")
    print(f"R²   - 均值: {np.mean(r2_values):.6f}, 标准差: {np.std(r2_values):.6f}")
    
    # 反归一化
    results_denorm = scaler_test.inverse_transform(results_flat.reshape(-1, 1)).flatten()
    reals_denorm = scaler_test.inverse_transform(reals_flat.reshape(-1, 1)).flatten()
    
    # 计算反归一化后的指标（对实际应用更有意义）
    denorm_metrics = calculate_metrics(reals_denorm, results_denorm)
    print("\n=== 反归一化后的指标（实际单位）===")
    print(f"MAE:  {denorm_metrics['MAE']:.6f}")
    print(f"RMSE: {denorm_metrics['RMSE']:.6f}")
    print(f"MAPE: {denorm_metrics['MAPE']:.4f}%")
    print(f"R²:   {denorm_metrics['R2']:.6f}")
    
    # 绘制预测结果对比
    plt.figure(figsize=(15, 8))
    plt.plot(reals_denorm[:200], label='Real Values', color='blue', linewidth=2)
    plt.plot(results_denorm[:200], label='LSTM Predictions', color='red', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title(f'Real vs LSTM Predictions (MAE: {denorm_metrics["MAE"]:.6f}, RMSE: {denorm_metrics["RMSE"]:.6f})', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制散点图（真实值 vs 预测值）
    plt.figure(figsize=(10, 8))
    plt.scatter(reals_denorm, results_denorm, alpha=0.6, color='blue', s=20)
    # 绘制y=x理想线
    min_val = min(reals_denorm.min(), results_denorm.min())
    max_val = max(reals_denorm.max(), results_denorm.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Real Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Real vs Predicted Values (R² = {denorm_metrics["R2"]:.6f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制误差分布
    errors = results_denorm - reals_denorm
    plt.figure(figsize=(15, 5))
    
    # MAE分布
    plt.subplot(1, 3, 1)
    plt.hist(mae_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('MAE Distribution')
    plt.xlabel('MAE')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # RMSE分布
    plt.subplot(1, 3, 2)
    plt.hist(rmse_values, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('RMSE Distribution')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 预测误差分布
    plt.subplot(1, 3, 3)
    plt.hist(errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (Predicted - Real)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制指标趋势图（如果序列较多）
    if len(all_metrics) > 10:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(mae_values, color='blue', linewidth=1)
        plt.title('MAE Trend')
        plt.xlabel('Sequence Index')
        plt.ylabel('MAE')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(rmse_values, color='red', linewidth=1)
        plt.title('RMSE Trend')
        plt.xlabel('Sequence Index')
        plt.ylabel('RMSE')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(r2_values, color='green', linewidth=1)
        plt.title('R² Trend')
        plt.xlabel('Sequence Index')
        plt.ylabel('R²')
        plt.grid(True, alpha=0.3)
        
        if mape_values:
            plt.subplot(2, 2, 4)
            plt.plot(mape_values[:len(r2_values)], color='orange', linewidth=1)
            plt.title('MAPE Trend')
            plt.xlabel('Sequence Index')
            plt.ylabel('MAPE (%)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('metrics_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 保存结果到文件
    results_summary = {
        'overall_metrics': overall_metrics,
        'denormalized_metrics': denorm_metrics,
        'mae_stats': {'mean': np.mean(mae_values), 'std': np.std(mae_values)},
        'rmse_stats': {'mean': np.mean(rmse_values), 'std': np.std(rmse_values)},
        'r2_stats': {'mean': np.mean(r2_values), 'std': np.std(r2_values)}
    }
    
    if mape_values:
        results_summary['mape_stats'] = {'mean': np.mean(mape_values), 'std': np.std(mape_values)}
    
    # 将结果保存为txt文件
    with open('model_evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== LSTM模型评价结果 ===\n\n")
        f.write("整体指标（归一化）:\n")
        f.write(f"MAE:  {overall_metrics['MAE']:.6f}\n")
        f.write(f"RMSE: {overall_metrics['RMSE']:.6f}\n")
        f.write(f"MAPE: {overall_metrics['MAPE']:.4f}%\n")
        f.write(f"R²:   {overall_metrics['R2']:.6f}\n\n")
        
        f.write("整体指标（实际单位）:\n")
        f.write(f"MAE:  {denorm_metrics['MAE']:.6f}\n")
        f.write(f"RMSE: {denorm_metrics['RMSE']:.6f}\n")
        f.write(f"MAPE: {denorm_metrics['MAPE']:.4f}%\n")
        f.write(f"R²:   {denorm_metrics['R2']:.6f}\n\n")
        
        f.write("各序列指标统计:\n")
        f.write(f"MAE  - 均值: {np.mean(mae_values):.6f}, 标准差: {np.std(mae_values):.6f}\n")
        f.write(f"RMSE - 均值: {np.mean(rmse_values):.6f}, 标准差: {np.std(rmse_values):.6f}\n")
        f.write(f"R²   - 均值: {np.mean(r2_values):.6f}, 标准差: {np.std(r2_values):.6f}\n")
        if mape_values:
            f.write(f"MAPE - 均值: {np.mean(mape_values):.4f}%, 标准差: {np.std(mape_values):.4f}%\n")
    
    print("\n评价结果已保存到 'model_evaluation_results.txt'")
    print("图表已保存: test_results.png, scatter_plot.png, error_distributions.png")
    if len(all_metrics) > 10:
        print("            metrics_trends.png")
