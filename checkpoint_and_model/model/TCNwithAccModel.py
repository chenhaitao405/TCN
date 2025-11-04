'''
集成SingleLegIMUPredictor和TCN的扭矩预测模型
输入流程：
1. 电机数据(2维) -> SingleLegIMUPredictor -> IMU预测(3维)
2. 拼接：电机数据(2维) + IMU预测(3维) = 5维特征
3. 5维特征 -> TCN -> 扭矩预测
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Optional
from model.acc_model import SingleLegIMUPredictor

class Chomp1d(nn.Module):
    """Removes the padding added by causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block containing two temporal convolutional layers for TCN"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights with small random values"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of temporal blocks with exponentially increasing dilation"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential increase in dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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


class IMUTemporalBlock(nn.Module):
    """时序卷积块，带残差连接 (for IMU predictor)"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(IMUTemporalBlock, self).__init__()
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


class HipMomentTCNWithIMU(nn.Module):
    """
    集成IMU预测的TCN扭矩估计模型
    
    数据流:
    1. 输入: 电机数据 (batch_size, 2, seq_length) [hip_angle, hip_velocity]
    2. IMU预测: SingleLegIMUPredictor -> (batch_size, 3, seq_length) [acc_x, acc_y, gyro_z]
    3. 特征拼接: 原始输入 + IMU预测 = (batch_size, 5, seq_length)
    4. TCN处理: 5维特征 -> 扭矩预测 (batch_size, 2, seq_length)
    """
    def __init__(self, 
                 motor_input_size=2,           # 电机输入维度
                 imu_predictor_config=None,    # IMU预测器配置
                 tcn_config=None,               # TCN配置
                 dropout=0.2,
                 center=0., 
                 scale=1.):
        super(HipMomentTCNWithIMU, self).__init__()
        
        # IMU预测器配置
        if imu_predictor_config is None:
            imu_predictor_config = {
                'motor_input_size': motor_input_size,
                'imu_output_size': 3,  # acc_x, acc_y, gyro_z
                'hidden_channels': [32, 64, 128, 64],
                'dropout': dropout
            }
        
        # TCN配置
        if tcn_config is None:
            tcn_config = {
                'num_channels': [50, 50, 50, 50, 50],  # 5 residual blocks
                'kernel_size': 4,
                'dropout': dropout
            }
        
        # 1. IMU预测器
        self.imu_predictor = SingleLegIMUPredictor(**imu_predictor_config)
        
        # 2. 特征融合后的输入维度：原始电机数据(2) + IMU预测(3) = 5
        tcn_input_size = motor_input_size + imu_predictor_config['imu_output_size']
        
        # 3. TCN网络
        self.tcn = TemporalConvNet(
            tcn_input_size, 
            tcn_config['num_channels'],
            kernel_size=tcn_config['kernel_size'],
            dropout=tcn_config['dropout']
        )
        
        # 4. 输出层
        self.linear = nn.Linear(tcn_config['num_channels'][-1], 1)  # 输出2维扭矩
        
        # 5. 归一化参数
        self.center = center
        self.scale = scale
        
        # 6. 计算有效历史长度
        kernel_size = tcn_config['kernel_size']
        num_layers = len(tcn_config['num_channels'])
        self.eff_hist = 1 + sum(2 * (kernel_size - 1) * (2 ** i) for i in range(num_layers))
        
        self.init_weights()
        
    def init_weights(self):
        """初始化输出层权重"""
        self.linear.weight.data.normal_(0, 0.01)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, 2, seq_length) - 电机数据 [hip_angle, hip_velocity]
        Returns:
            output: (batch_size, 2, seq_length) - 扭矩预测
        """
        # 1. 归一化输入
        x_normalized = (x - self.center) / self.scale
        # x_normalized = x
        # 2. 使用IMU预测器预测IMU数据
        imu_predictions = self.imu_predictor(x_normalized)
        
        # 3. 拼接原始输入和IMU预测
        # x_normalized: (batch_size, 2, seq_length)
        # imu_predictions: (batch_size, 3, seq_length)
        # combined: (batch_size, 5, seq_length)
        combined_features = torch.cat([x_normalized, imu_predictions], dim=1)
        
        # 4. 通过TCN网络
        tcn_out = self.tcn(combined_features)
        
        # 5. 准备输出层
        batch_size, channels, seq_len = tcn_out.shape
        tcn_out = tcn_out.permute(0, 2, 1).contiguous()  # (batch, time, channels)
        tcn_out = tcn_out.view(-1, channels)  # (batch*time, channels)
        
        # 6. 应用线性层
        output = self.linear(tcn_out)  # (batch*time, 2)
        
        # 7. 重塑回原始格式
        output = output.view(batch_size, seq_len, 1)  # (batch, time, 2)
        output = output.permute(0, 2, 1)  # (batch, 2, time)
        
        return output
    
    def get_effective_history(self):
        """获取模型的有效历史长度"""
        return self.eff_hist
    
    def get_imu_predictions(self, x):
        """
        单独获取IMU预测结果（用于调试或分析）
        Args:
            x: (batch_size, 2, seq_length) - 电机数据
        Returns:
            imu_predictions: (batch_size, 3, seq_length) - IMU预测
        """
        x_normalized = (x - self.center) / self.scale
        return self.imu_predictor(x_normalized)


# 示例使用
if __name__ == "__main__":
    # 创建模型
    model = HipMomentTCNWithIMU(
        motor_input_size=2,  # 髋关节角度和角速度
        imu_predictor_config={
            'motor_input_size': 2,
            'imu_output_size': 3,
            'hidden_channels': [64, 128, 256, 64],
            'dropout': 0.2
        },
        tcn_config={
            'num_channels': [50, 50, 50, 50, 50],
            'kernel_size': 4,
            'dropout': 0.2
        },
        dropout=0.2
    )
    
    # 打印模型信息
    print(f"有效历史长度: {model.get_effective_history()} timesteps")
    print(f"在200Hz采样率下: {model.get_effective_history() * 5}ms")
    
    # 测试输入
    batch_size = 4
    seq_length = 218
    x = torch.randn(batch_size, 2, seq_length)  # 2通道输入：髋关节角度和角速度
    
    # 前向传播
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 获取IMU预测（用于调试）
    imu_predictions = model.get_imu_predictions(x)
    print(f"IMU预测形状: {imu_predictions.shape}")
    
    # 验证输出形状
    assert output.shape == (batch_size, 1, seq_length)
    assert imu_predictions.shape == (batch_size, 3, seq_length)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 分别统计两个子模块的参数
    imu_params = sum(p.numel() for p in model.imu_predictor.parameters())
    tcn_params = sum(p.numel() for p in model.tcn.parameters())
    linear_params = sum(p.numel() for p in model.linear.parameters())
    
    print(f"\n参数统计:")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"  - IMU预测器参数: {imu_params:,}")
    print(f"  - TCN参数: {tcn_params:,}")
    print(f"  - 输出层参数: {linear_params:,}")