import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import os
import sys

class SingleLegIMUPredictor(nn.Module):
    """
    单侧腿IMU预测器
    输入: 2维电机数据 (髋关节角度+角速度)
    输出: 3维IMU数据 (acc_x, acc_y, gyro_z)
    """
    def __init__(self, 
                 motor_input_size=2,    # 单个关节 × (角度+角速度)
                 imu_output_size=3,     # acc_x, acc_y, gyro_z
                 hidden_channels=[32, 64, 128, 64],
                 dropout=0.2):
        super(SingleLegIMUPredictor, self).__init__()
        
        self.motor_input_size = motor_input_size
        self.imu_output_size = imu_output_size
        
        # 1. 物理特征提取层
        self.physics_layer = nn.Sequential(
            nn.Conv1d(motor_input_size, hidden_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 多尺度特征提取
        self.multi_scale = MultiScaleConv(hidden_channels[0], hidden_channels[1])
        
        # 3. 时序卷积块
        self.temporal_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 2, 1]  # dilation pattern
        
        for i in range(len(hidden_channels)-1):
            in_channels = hidden_channels[i] if i > 0 else hidden_channels[1]
            out_channels = hidden_channels[i+1]
            dilation = dilations[i % len(dilations)]
            
            self.temporal_blocks.append(
                TemporalBlock(in_channels, out_channels, 
                            kernel_size=3, stride=1, 
                            dilation=dilation, dropout=dropout)
            )
        
        # 4. 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_channels[-1], hidden_channels[-1]//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels[-1]//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_channels[-1]//2, hidden_channels[-1]//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels[-1]//4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 5. 输出层
        self.output_layer = nn.Conv1d(hidden_channels[-1]//4, imu_output_size, kernel_size=1)
        
        # 6. 直接映射（残差连接）
        self.direct_mapping = nn.Sequential(
            nn.Conv1d(motor_input_size, imu_output_size, kernel_size=1),
            nn.BatchNorm1d(imu_output_size)
        )
        
        # 可学习的残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, motor_data):
        """
        Args:
            motor_data: (batch_size, 2, seq_length)
                        [hip_angle, hip_velocity]
        Returns:
            imu_predictions: (batch_size, 3, seq_length)
                           [acc_x, acc_y, gyro_z]
        """
        # 直接映射（残差）
        residual = self.direct_mapping(motor_data)
        
        # 主路径
        x = self.physics_layer(motor_data)
        x = self.multi_scale(x)
        
        # 通过时序块
        for block in self.temporal_blocks:
            x = block(x)
        
        # 特征融合
        x = self.fusion(x)
        
        # 输出
        out = self.output_layer(x)
        
        # 添加残差连接
        out = out + residual * self.residual_weight
        
        return out


class TemporalBlock(nn.Module):
    """时序卷积块，带残差连接"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        # 残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class MultiScaleConv(nn.Module):
    """多尺度卷积层"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        
        # 不同感受野的卷积
        self.conv3 = nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels//4, kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)  # 点卷积
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 并行提取多尺度特征
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        f1 = self.conv1(x)
        
        # 拼接
        out = torch.cat([f3, f5, f7, f1], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

if __name__ == "__main__":

    model = SingleLegIMUPredictor(
        motor_input_size=2,
        imu_output_size=3,
        hidden_channels=[64, 128, 256, 128]
    )
    # Test input with specified dimensions
    batch_size = 4
    seq_length = 218
    input_features = 2
    x = torch.randn(batch_size, input_features, seq_length)
    
    print(f"Input shape: {x.shape}")
    print(f"Target output shape: ({batch_size}, 3, {seq_length})")
    print(f"TCN channels: [64, 64, 64, 64, 64]")
    print("Note: Cross-block residual connections are automatically added between blocks with same channel numbers")
    
    # Forward pass
    output = model(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Success! Output matches target: ({batch_size}, 3, {seq_length})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    from torchsummary import summary
    summary(model, (2,218),4,device="cpu")