import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Calculate MAE
def calculate_mae(real, pred):
    return np.mean(np.abs(real - pred))

# Calculate RMSE
def calculate_rmse(real, pred):
    return np.sqrt(np.mean((real - pred) ** 2))

# Calculate MAPE
def calculate_mape(real, pred):
    mask = real != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100

# Calculate R2 score
def calculate_r2(real, pred):
    return r2_score(real, pred)

# Calculate all metrics
def calculate_metrics(real, pred):
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

# Set seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Read the dataset
data_path = r'E:\sar\csv\transposed_output2019all.csv'  # Replace with your dataset path
raw_data = pd.read_csv(data_path)

# Select columns for training (you can adjust as needed)
target_columns = ['2', '63', '85', '197', '237', '267', '269', '271', '273', '275', '277', '279', '281', '283', '285', '287', '289', '291', '293', '295', '297', '299', '301', '303', '305', '307', '309', '311', '313', '315', '317', '319', '321', '323', '325', '327']
print(f"Selected columns for training: {target_columns}")
print(f"Total data length: {len(raw_data)}")

# Data preprocessing
multi_point_data = raw_data[target_columns].values  # (time_steps, num_points)
num_points = len(target_columns)
print(f"Data shape: {multi_point_data.shape}")

# Define training and testing sizes
test_size = 0.15
train_size = 0.85
pre_len = 4
train_window = 10

train_len = int(train_size * len(multi_point_data))
test_len = int(test_size * len(multi_point_data))

train_data = multi_point_data[:train_len]
test_data = multi_point_data[-test_len:]

print(f"Training data size: {train_data.shape}")
print(f"Test data size: {test_data.shape}")

# Normalize each point separately
scalers = {}
train_data_normalized = np.zeros_like(train_data)
test_data_normalized = np.zeros_like(test_data)

for i, col in enumerate(target_columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_normalized[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
    test_data_normalized[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
    scalers[col] = scaler

# Convert to PyTorch tensors
train_data_normalized = torch.FloatTensor(train_data_normalized)
test_data_normalized = torch.FloatTensor(test_data_normalized)

# Create sequences for LSTM
def create_multipoint_sequences(input_data, tw, pre_len):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - pre_len + 1):
        train_seq = input_data[i:i + tw]  # (tw, num_points)
        train_label = input_data[i + tw:i + tw + pre_len]  # (pre_len, num_points)
        inout_seq.append((train_seq, train_label))
    return inout_seq

# Prepare training sequences
train_inout_seq = create_multipoint_sequences(train_data_normalized, train_window, pre_len)
print(f"Training sequences: {len(train_inout_seq)}")
print(f"Shape of one input sequence: {train_inout_seq[0][0].shape}")
print(f"Shape of one label sequence: {train_inout_seq[0][1].shape}")

# Define MultiPointLSTM model
class MultiPointLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None, num_layers=2, dropout=0.2, bidirectional=False):
        super(MultiPointLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim else input_dim
        
        # LSTM Layer (bi-directional)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), self.output_dim * pre_len)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        h0 = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        last_output = lstm_out[:, -1, :]  # Get the output of the last time step
        
        last_output = self.dropout(last_output)
        
        output = self.fc(last_output)  # Get predictions
        output = output.view(batch_size, pre_len, self.output_dim)  # Reshape
        
        return output

# Model initialization
lstm_model = MultiPointLSTM(
    input_dim=num_points,
    hidden_dim=128,  # Increased hidden dimensions
    output_dim=num_points,
    num_layers=7,    # Multiple layers
    dropout=0.2,
    bidirectional=True  # Use bidirectional LSTM
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

# Training function
def train_model(model, train_seq, num_epochs=100, batch_size=1, patience=10):
    best_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for seq, label in train_seq:
            seq = seq.unsqueeze(0)  # Add batch dimension
            label = label.unsqueeze(0)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(seq)
            
            # Calculate loss
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_seq)
        scheduler.step(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

# Train the model
train_model(lstm_model, train_inout_seq, num_epochs=100)

# Load the best model for evaluation
lstm_model.load_state_dict(torch.load('best_model.pth'))
lstm_model.eval()

# Testing function
def test_model(model, test_data, scalers, pre_len=4):
    model.eval()
    test_inout_seq = create_multipoint_sequences(test_data, train_window, pre_len)
    
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for seq, label in test_inout_seq:
            seq = seq.unsqueeze(0)
            output = model(seq)
            
            predictions.append(output.squeeze().cpu().numpy())
            real_values.append(label.numpy())
    
    predictions = np.array(predictions)
    real_values = np.array(real_values)
    
    # Denormalize predictions and real values
    denormalized_preds = np.zeros_like(predictions)
    denormalized_real = np.zeros_like(real_values)
    
    for i, col in enumerate(target_columns):
        denormalized_preds[:, :, i] = scalers[col].inverse_transform(predictions[:, :, i])
        denormalized_real[:, :, i] = scalers[col].inverse_transform(real_values[:, :, i])
    
    return denormalized_preds, denormalized_real

# Get predictions and real values
denormalized_preds, denormalized_real = test_model(lstm_model, test_data_normalized, scalers)

# Evaluate the model
metrics = {}
for i, col in enumerate(target_columns):
    print(f"Evaluating point {col}")
    real = denormalized_real[:, :, i].flatten()
    pred = denormalized_preds[:, :, i].flatten()
    
    metrics[col] = calculate_metrics(real, pred)

# Print average metrics
average_metrics = {metric: np.mean([metrics[col][metric] for col in target_columns]) for metric in metrics['2'].keys()}
print(f"Average Metrics: {average_metrics}")

