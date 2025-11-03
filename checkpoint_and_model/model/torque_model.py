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


class SingleLegTorquePredictor(nn.Module):
    """
    单侧腿扭矩预测器
    输入: 2维电机数据 (髋关节角度+角速度)
    输出: 1维扭矩数据
    """
    def __init__(self, 
                 motor_input_size=2,     # 单个关节 × (角度+角速度)
                 torque_output_size=1,   # 扭矩输出
                 hidden_channels=[32, 64, 128, 64],
                 dropout=0.2):
        super(SingleLegTorquePredictor, self).__init__()
        
        self.motor_input_size = motor_input_size
        self.torque_output_size = torque_output_size
        
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
        
        # 5. 输出层 - 修改为输出1维扭矩
        self.output_layer = nn.Conv1d(hidden_channels[-1]//4, torque_output_size, kernel_size=1)
        
        # 6. 直接映射（残差连接）- 修改为1维输出
        self.direct_mapping = nn.Sequential(
            nn.Conv1d(motor_input_size, torque_output_size, kernel_size=1),
            nn.BatchNorm1d(torque_output_size)
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
            torque_predictions: (batch_size, 1, seq_length)
                               扭矩预测值
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


class EnhancedTorquePredictor(nn.Module):
    """
    增强版扭矩预测器 - 带注意力机制
    """
    def __init__(self, 
                 motor_input_size=2,
                 torque_output_size=1,
                 hidden_channels=[32, 64, 128, 64],
                 dropout=0.2,
                 use_attention=True):
        super(EnhancedTorquePredictor, self).__init__()
        
        self.motor_input_size = motor_input_size
        self.torque_output_size = torque_output_size
        self.use_attention = use_attention
        
        # 1. 输入编码
        self.encoder = nn.Sequential(
            nn.Conv1d(motor_input_size, hidden_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 多尺度特征提取
        self.multi_scale = MultiScaleConv(hidden_channels[0], hidden_channels[1])
        
        # 3. 时序卷积块（带膨胀卷积）
        self.temporal_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8, 4, 2, 1]
        
        for i in range(len(hidden_channels)-1):
            in_channels = hidden_channels[i] if i > 0 else hidden_channels[1]
            out_channels = hidden_channels[i+1]
            dilation = dilations[i % len(dilations)]
            
            self.temporal_blocks.append(
                TemporalBlock(in_channels, out_channels, 
                            kernel_size=3, stride=1, 
                            dilation=dilation, dropout=dropout)
            )
        
        # 4. 注意力机制（可选）
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_channels[-1],
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_channels[-1])
        
        # 5. 特征融合和输出
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_channels[-1], hidden_channels[-1]//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels[-1]//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_channels[-1]//2, hidden_channels[-1]//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels[-1]//4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(hidden_channels[-1]//4, torque_output_size, kernel_size=1)
        )
        
        # 6. 残差连接
        self.residual = nn.Sequential(
            nn.Conv1d(motor_input_size, torque_output_size, kernel_size=1),
            nn.BatchNorm1d(torque_output_size)
        )
        
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
        Returns:
            torque_predictions: (batch_size, 1, seq_length)
        """
        batch_size, _, seq_len = motor_data.shape
        
        # 残差连接
        residual = self.residual(motor_data)
        
        # 主路径
        x = self.encoder(motor_data)
        x = self.multi_scale(x)
        
        # 时序卷积
        for block in self.temporal_blocks:
            x = block(x)
        
        # 注意力机制（可选）
        if self.use_attention:
            # 重塑为 (batch, seq_len, channels) 格式
            x_att = x.permute(0, 2, 1)
            x_att, _ = self.attention(x_att, x_att, x_att)
            x_att = self.attention_norm(x_att)
            # 重塑回 (batch, channels, seq_len)
            x = x + x_att.permute(0, 2, 1)
        
        # 解码输出
        out = self.decoder(x)
        
        # 添加残差
        out = out + residual * self.residual_weight
        
        return out


# 测试代码
if __name__ == "__main__":
    # 测试基础模型
    print("Testing SingleLegTorquePredictor...")
    model = SingleLegTorquePredictor(
        motor_input_size=2,
        torque_output_size=1,
        hidden_channels=[32, 64, 128, 64],
        dropout=0.2
    )
    
    # 创建测试输入
    batch_size = 8
    seq_length = 218
    test_input = torch.randn(batch_size, 2, seq_length)
    
    # 前向传播
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 1, {seq_length})")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 测试增强版模型
    print("\n" + "="*50)
    print("Testing EnhancedTorquePredictor with attention...")
    enhanced_model = EnhancedTorquePredictor(
        motor_input_size=2,
        torque_output_size=1,
        hidden_channels=[32, 64, 128, 64],
        dropout=0.2,
        use_attention=True
    )
    
    output_enhanced = enhanced_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output_enhanced.shape}")
    
    total_params_enhanced = sum(p.numel() for p in enhanced_model.parameters())
    print(f"Total parameters (enhanced): {total_params_enhanced:,}")
    
    print("\nModels created successfully!")