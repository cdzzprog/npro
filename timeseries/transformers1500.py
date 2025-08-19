import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import gc
from tqdm import tqdm
import math
import seaborn as sns
import warnings
from torch.utils.data import Dataset, DataLoader

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def calculate_metrics(real, pred):
    """计算评估指标"""
    epsilon = 1e-6
    mae = np.mean(np.abs(real - pred))
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    mape = np.mean(np.abs((real - pred) / (np.abs(real) + epsilon)) * 100)
    r2 = r2_score(real, pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def save_metrics(metrics, filename):
    """保存指标到文件"""
    if isinstance(metrics, dict):
        if all(isinstance(v, dict) for v in metrics.values()):
            df = pd.DataFrame(metrics).T
        else:
            df = pd.DataFrame([metrics])
        df.to_csv(filename)
        print(f"指标保存至 {filename}")
    else:
        print(f"无法保存指标: 不支持的格式")

def plot_learning_curve(train_loss, val_loss, filename):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='训练损失')
    if val_loss and len(val_loss) > 0:
        plt.plot(val_loss, label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataPreprocessor:
    """数据预处理管道"""
    def __init__(self, n_points=500, min_seq_length=10):
        self.n_points = n_points
        self.min_seq_length = min_seq_length
        self.scaler = None
        self.target_columns = None
    
    def load_and_prepare(self, filepath):
        """加载并准备数据"""
        print(f"加载数据: {filepath}")
        df = pd.read_csv(filepath)
        
        # 选择数值列
        numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        if len(numeric_cols) > self.n_points:
            self.target_columns = numeric_cols[:self.n_points]
            print(f"选择前{self.n_points}个点")
        else:
            self.target_columns = numeric_cols
            print(f"使用所有{len(numeric_cols)}个点")
        
        data = df[self.target_columns].values.astype(np.float32)
        print(f"原始数据形状: {data.shape}")
        
        # 基本数据检查
        print(f"NaN数量: {np.isnan(data).sum()}")
        print(f"Inf数量: {np.isinf(data).sum()}")
        
        # 处理NaN和Inf
        data = np.nan_to_num(data)
        
        return data
    
    def scale_data(self, data):
        """标准化数据"""
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(data)
    
    def prepare_datasets(self, scaled_data, seq_length, pred_length, train_ratio=0.7, val_ratio=0.15):
        """准备数据集"""
        total_sequences = len(scaled_data) - seq_length - pred_length + 1
        
        if total_sequences < 10:
            print(f"警告: 总序列数({total_sequences})较少，将使用所有数据训练")
            train_ratio = 1.0
            val_ratio = 0.0
        
        train_end = int(total_sequences * train_ratio)
        val_end = train_end + int(total_sequences * val_ratio)
        
        # 创建所有序列
        X, y = self.create_sequences(scaled_data, seq_length, pred_length)
        
        # 分割数据集
        train_X, train_y = X[:train_end], y[:train_end]
        val_X, val_y = X[train_end:val_end], y[train_end:val_end]
        test_X, test_y = X[val_end:], y[val_end:]
        
        print(f"\n数据集形状:")
        print(f"训练集: {len(train_X)} 样本")
        print(f"验证集: {len(val_X)} 样本")
        print(f"测试集: {len(test_X)} 样本")
        
        return (train_X, train_y), (val_X, val_y), (test_X, test_y)
    
    def create_sequences(self, data, seq_length, pred_length):
        """创建时间序列样本"""
        X, y = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length:i+seq_length+pred_length])
        return np.array(X), np.array(y)
    
    def inverse_scale(self, data):
        """反标准化数据"""
        if data.ndim == 3:
            orig_shape = data.shape
            flat_data = data.reshape(-1, orig_shape[-1])
            unscaled = self.scaler.inverse_transform(flat_data)
            return unscaled.reshape(orig_shape)
        else:
            return self.scaler.inverse_transform(data)

class EnhancedLSTM(nn.Module):
    """增强的LSTM模型，解决恒定预测问题"""
    def __init__(self, input_size, output_size, pred_length, hidden_size=128, num_layers=1, dropout=0.1):
        super(EnhancedLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pred_length = pred_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 输出层 - 预测多个时间步
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, output_size * pred_length)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # 嵌入层
        embedded = self.embedding(x)
        
        # LSTM层
        lstm_out, _ = self.lstm(embedded)
        
        # 只取最后一个时间步
        last_out = lstm_out[:, -1, :]
        
        # 输出层
        output = self.output_layer(last_out)
        
        # 重塑为 (batch_size, pred_len, output_size)
        output = output.view(batch_size, self.pred_length, self.output_size)
        
        return output

class Trainer:
    """模型训练器"""
    def __init__(self, model, device, lr=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.HuberLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        if len(train_loader) == 0:
            return 0.0
            
        self.model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(X_batch)
            
            # 确保预测和目标形状匹配
            if outputs.shape != y_batch.shape:
                print(f"形状不匹配: 输出 {outputs.shape} vs 目标 {y_batch.shape}")
                continue
            
            # 计算损失
            loss = self.criterion(outputs, y_batch)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
        
        return epoch_loss / len(train_loader.dataset)
    
    def evaluate(self, data_loader):
        """评估模型"""
        if len(data_loader) == 0 or len(data_loader.dataset) == 0:
            return float('inf')
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                
                # 确保预测和目标形状匹配
                if outputs.shape != y_batch.shape:
                    print(f"形状不匹配: 输出 {outputs.shape} vs 目标 {y_batch.shape}")
                    continue
                
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        
        return total_loss / len(data_loader.dataset)
    
    def predict(self, data_loader):
        """生成预测"""
        if len(data_loader) == 0:
            return np.array([]), np.array([])
            
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                
                # 确保预测和目标形状匹配
                if outputs.shape != y_batch.shape:
                    print(f"形状不匹配: 输出 {outputs.shape} vs 目标 {y_batch.shape}")
                    continue
                
                all_outputs.append(outputs.detach().cpu().numpy())
                all_targets.append(y_batch.detach().cpu().numpy())
        
        if len(all_outputs) > 0:
            return np.concatenate(all_outputs, axis=0), np.concatenate(all_targets, axis=0)
        return np.array([]), np.array([])
    
    def train(self, train_loader, val_loader, epochs=100, patience=15):
        """训练模型"""
        best_val_loss = float('inf')
        no_improve = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            if val_loss != float('inf'):
                self.scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss if val_loss != float('inf') else None)
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            # 打印训练信息
            print(f'Epoch {epoch}/{epochs}: '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss if val_loss != float("inf") else "N/A"}, '
                  f'Time: {epoch_time:.2f}s')
            
            # 检查早停
            if val_loss < best_val_loss and val_loss != float('inf'):
                best_val_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"验证损失提升至 {best_val_loss:.6f}，保存模型")
            else:
                no_improve += 1
                print(f"验证损失未提升 {no_improve}/{patience}")
                if no_improve >= patience:
                    print(f"早停在 {epoch} epoch")
                    break
        
        # 保存损失曲线
        valid_val_losses = [v for v in val_losses if v is not None]
        plot_learning_curve(train_losses, valid_val_losses, 'learning_curve.png')
        
        return train_losses, val_losses

def analyze_results(predictions, targets, preprocessor, sample_points=5):
    """分析预测结果"""
    if len(predictions) == 0 or len(targets) == 0:
        print("没有预测结果可供分析")
        return
    
    print(f"预测形状: {predictions.shape}")
    print(f"目标形状: {targets.shape}")
    
    # 反标准化
    preds_unscaled = preprocessor.inverse_scale(predictions)
    targets_unscaled = preprocessor.inverse_scale(targets)
    
    # 整体评估 - 确保形状匹配
    if preds_unscaled.shape != targets_unscaled.shape:
        print(f"警告: 预测和目标形状不匹配 {preds_unscaled.shape} vs {targets_unscaled.shape}")
        # 尝试截断较长的数组
        min_len = min(len(preds_unscaled), len(targets_unscaled))
        preds_unscaled = preds_unscaled[:min_len]
        targets_unscaled = targets_unscaled[:min_len]
    
    # 展平所有维度
    preds_flat = preds_unscaled.reshape(-1)
    targets_flat = targets_unscaled.reshape(-1)
    
    # 检查形状是否匹配
    if len(preds_flat) != len(targets_flat):
        print(f"严重错误: 展平后形状不匹配 {len(preds_flat)} vs {len(targets_flat)}")
        return
    
    # 整体评估
    print("\n整体评估:")
    overall_metrics = calculate_metrics(targets_flat, preds_flat)
    print(f"MAE: {overall_metrics['MAE']:.4f}")
    print(f"RMSE: {overall_metrics['RMSE']:.4f}")
    print(f"MAPE: {overall_metrics['MAPE']:.2f}%")
    print(f"R²: {overall_metrics['R2']:.4f}")
    
    # 保存整体指标
    save_metrics(overall_metrics, 'overall_metrics.csv')
    
    # 随机选择几个点进行详细分析
    n_points = preds_unscaled.shape[2] if preds_unscaled.ndim == 3 else 1
    if n_points > sample_points:
        sample_indices = np.random.choice(n_points, sample_points, replace=False)
    else:
        sample_indices = range(n_points)
    
    for i, point_idx in enumerate(sample_indices):
        print(f"\n点 {point_idx} 的分析:")
        
        # 提取该点的预测和目标
        if preds_unscaled.ndim == 3:  # (samples, timesteps, features)
            point_preds = preds_unscaled[:, :, point_idx].reshape(-1)
            point_targets = targets_unscaled[:, :, point_idx].reshape(-1)
        else:  # (samples, features)
            point_preds = preds_unscaled[:, point_idx].reshape(-1)
            point_targets = targets_unscaled[:, point_idx].reshape(-1)
        
        # 计算指标
        point_metrics = calculate_metrics(point_targets, point_preds)
        print(f"MAE: {point_metrics['MAE']:.4f}")
        print(f"RMSE: {point_metrics['RMSE']:.4f}")
        print(f"MAPE: {point_metrics['MAPE']:.2f}%")
        print(f"R²: {point_metrics['R2']:.4f}")
        
        # 绘制预测对比图
        plt.figure(figsize=(12, 6))
        plt.plot(point_targets[:200], 'b-', label='真实值')
        plt.plot(point_preds[:200], 'r--', label='预测值')
        plt.title(f'点 {point_idx} 预测对比')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'point_{point_idx}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 检查预测多样性
    print("\n预测多样性检查:")
    if len(preds_unscaled) > 0:
        sample_size = min(5, len(preds_unscaled))
        if preds_unscaled.ndim == 3:
            # (samples, timesteps, features)
            sample_preds = preds_unscaled[:sample_size, :, :min(5, preds_unscaled.shape[2])]
        else:
            # (samples, features)
            sample_preds = preds_unscaled[:sample_size, :min(5, preds_unscaled.shape[1])]
        
        for i in range(sample_size):
            print(f"样本 {i+1}:")
            if sample_preds.ndim == 3:
                for t in range(min(5, sample_preds.shape[1])):
                    print(f"  时间步 {t+1}: {sample_preds[i, t].round(4)}")
            else:
                print(f"  预测值: {sample_preds[i].round(4)}")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 配置参数
    config = {
        'data_path': r'E:\sar\csv\transposed_output2019all.csv',
        'n_points': 500,  # 减少点数以适应小数据量
        'seq_length': 10,  # 输入序列长度
        'pred_length': 4,  # 预测长度
        'batch_size': 8,   # 减小批大小
        'epochs': 50,      # 减少训练轮数
        'patience': 10,    # 早停耐心值
        'hidden_size': 128,# 减小隐藏层大小
        'num_layers': 1,   # 减少LSTM层数
        'dropout': 0.1,    # 减小dropout
        'lr': 0.001        # 学习率
    }
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理
    preprocessor = DataPreprocessor(n_points=config['n_points'])
    data = preprocessor.load_and_prepare(config['data_path'])
    
    # 检查数据长度并调整序列长度
    if len(data) < config['seq_length'] + config['pred_length']:
        min_required = config['seq_length'] + config['pred_length']
        print(f"数据长度({len(data)})不足所需的最小长度({min_required})")
        
        # 自动调整序列长度
        max_seq_length = max(5, len(data) - config['pred_length'] - 1)
        if max_seq_length < 5:
            print("数据量过少，无法创建有效序列")
            return
        
        config['seq_length'] = max_seq_length
        print(f"自动调整序列长度为: {config['seq_length']}")
    
    # 标准化数据
    scaled_data = preprocessor.scale_data(data)
    
    # 创建数据集
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = preprocessor.prepare_datasets(
        scaled_data,
        config['seq_length'],
        config['pred_length']
    )
    
    # 创建数据加载器
    train_dataset = TimeSeriesDataset(train_X, train_y)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(config['batch_size'], len(train_dataset)), 
        shuffle=True if len(train_dataset) > 1 else False
    )
    
    val_dataset = TimeSeriesDataset(val_X, val_y)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=min(config['batch_size'], len(val_dataset)) if len(val_dataset) > 0 else 1, 
        shuffle=False
    )
    
    test_dataset = TimeSeriesDataset(test_X, test_y)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=min(config['batch_size'], len(test_dataset)) if len(test_dataset) > 0 else 1, 
        shuffle=False
    )
    
    # 创建模型 - 添加pred_length参数
    model = EnhancedLSTM(
        input_size=len(preprocessor.target_columns),
        output_size=len(preprocessor.target_columns),
        pred_length=config['pred_length'],  # 添加预测长度参数
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # 打印模型信息
    print(f"\n模型架构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")
    
    # 训练模型
    trainer = Trainer(
        model, 
        device, 
        lr=config['lr']
    )
    
    print("\n开始训练...")
    train_losses, val_losses = trainer.train(
        train_loader, 
        val_loader, 
        epochs=config['epochs'], 
        patience=config['patience']
    )
    
    # 加载最佳模型
    if os.path.exists('best_model.pth'):
        print("\n加载最佳模型进行测试...")
        model.load_state_dict(torch.load('best_model.pth'))
    
        # 测试集评估
        test_loss = trainer.evaluate(test_loader)
        print(f"测试损失: {test_loss:.6f}")
        
        # 生成预测
        print("\n生成预测...")
        predictions, targets = trainer.predict(test_loader)
        
        # 分析结果
        analyze_results(
            predictions, 
            targets, 
            preprocessor
        )
    
    print("\n训练和评估完成!")

if __name__ == "__main__":
    main()