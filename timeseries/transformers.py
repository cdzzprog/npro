import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
#import tushare as ts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import sys
import os
import gc
import argparse
import warnings
class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self,feature_size=1,hidden_size=128,num_layers=3,nhead=4,dropout=0.2):
        super(Transformer, self).__init__()
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        
        """
        `d_model`：模型的维度，也就是输入和输出的特征维度。
        `nhead`：注意力头数，控制多头注意力的并行度。
        """
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers,mask_check=False) 
        
        self.decoder = nn.Linear(hidden_size, 1) #feature_size是input的个数，1为output个数
        self.init_weights()
    
    #init_weight主要是用于设置decoder的参数
    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
 
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
 
    def forward(self, src, device='cpu'):
        output, (h0,c0) = self.lstm(src)
        # output (batch_size, time_stamp, hidden_size)
        batch_size, time_stamp, hidden_size = output.shape
        
        #print(output.reshape (time_stamp,batch_size,hidden_size).shape)
        #print(output.shape, h0.shape)
        #mask = self._generate_square_subsequent_mask(len(x)).to(device)
        mask = None
        #output = output.reshape(time_stamp,batch_size,hidden_size)
        output = self.transformer_encoder(output)
        #print(output.shape)
        output = self.decoder(output[:,-1,:])
        return output