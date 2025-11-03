'''
Auxiliary Task Learning TCN for hip moment estimation
Uses auxiliary IMU prediction task to improve motor-only inference
Updated to use ThighIMUPredictor instead of AuxiliaryIMUPredictor
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Removes the padding added by causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block containing two temporal convolutional layers"""
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
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalBlockForIMU(nn.Module):
    """时序卷积块，带残差连接 - 用于ThighIMUPredictor"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlockForIMU, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class PhysicsInformedLayer(nn.Module):
    """物理信息层 - 提取运动学特征"""
    def __init__(self, input_size=4, output_size=32):
        super(PhysicsInformedLayer, self).__init__()
        
        # 差分特征（近似速度变化和加速度）
        self.diff1_conv = nn.Conv1d(input_size, output_size//4, kernel_size=2, padding=0)
        self.diff2_conv = nn.Conv1d(input_size, output_size//4, kernel_size=3, padding=0)
        
        # 原始特征
        self.feat_conv = nn.Conv1d(input_size, output_size//2, kernel_size=3, padding=1)
        
        self.bn = nn.BatchNorm1d(output_size)
        
    def forward(self, x):
        diff1 = F.pad(self.diff1_conv(x), (0, 1), mode='replicate')
        diff2 = F.pad(self.diff2_conv(x), (0, 2), mode='replicate')
        feat = self.feat_conv(x)
        out = torch.cat([diff1, diff2, feat], dim=1)
        return self.bn(out)


class ThighIMUPredictor(nn.Module):
    """
    大腿IMU加速度预测器
    输入: 4维电机数据 (左右髋关节角度+角速度)
    输出: 4维或自定义维度IMU加速度
    """
    def __init__(self, 
                 motor_input_size=4,
                 imu_output_size=4,
                 hidden_channels=[32, 64, 128, 64],
                 dropout=0.2):
        super(ThighIMUPredictor, self).__init__()
        
        self.motor_input_size = motor_input_size
        self.imu_output_size = imu_output_size
        
        # 1. 物理特征提取
        self.physics_layer = PhysicsInformedLayer(motor_input_size, hidden_channels[0])
        
        # 2. 多尺度特征提取
        self.multi_scale = nn.ModuleList([
            nn.Conv1d(hidden_channels[0], hidden_channels[1]//4, kernel_size=3, padding=1),
            nn.Conv1d(hidden_channels[0], hidden_channels[1]//4, kernel_size=5, padding=2),
            nn.Conv1d(hidden_channels[0], hidden_channels[1]//4, kernel_size=7, padding=3),
            nn.Conv1d(hidden_channels[0], hidden_channels[1]//4, kernel_size=1),
        ])
        self.multi_scale_bn = nn.BatchNorm1d(hidden_channels[1])
        
        # 3. 时序卷积块
        self.temporal_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 2, 1]
        
        for i in range(len(hidden_channels)-1):
            in_channels = hidden_channels[i] if i > 0 else hidden_channels[1]
            out_channels = hidden_channels[i+1]
            dilation = dilations[i % len(dilations)]
            
            self.temporal_blocks.append(
                TemporalBlockForIMU(in_channels, out_channels, 
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
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        """
        Args:
            x: (batch_size, 4, seq_length) motor data
        Returns:
            predictions: (batch_size, imu_output_size, seq_length) predicted IMU accelerations
        """
        # 直接映射
        residual = self.direct_mapping(x)
        
        # 主路径
        x = self.physics_layer(x)
        
        # 多尺度特征
        multi_scale_features = []
        for conv in self.multi_scale:
            multi_scale_features.append(conv(x))
        x = torch.cat(multi_scale_features, dim=1)
        x = self.multi_scale_bn(x)
        
        # 时序块
        for block in self.temporal_blocks:
            x = block(x)
        
        # 特征融合
        x = self.fusion(x)
        
        # 输出
        out = self.output_layer(x)
        
        # 添加残差
        out = out + residual * self.residual_weight
        
        return out


class HipMomentTCNWithAuxiliary(nn.Module):
    """
    TCN for hip moment estimation with auxiliary thigh IMU prediction task
    Now uses ThighIMUPredictor for better feature extraction
    
    Architecture:
    1. ThighIMUPredictor learns to map motor data to thigh IMU features with physics-informed layers
    2. Concatenate original motor data with predicted thigh IMU features
    3. Main TCN processes the concatenated features
    4. Final output predicts hip moments
    """
    def __init__(self, motor_input_size=4, predicted_thigh_imu_size=12, 
                 main_tcn_channels=[50, 50, 50, 50, 50],
                 aux_hidden_channels=[32, 64, 128, 64],  # Updated for ThighIMUPredictor
                 kernel_size=4, dropout=0.2, center=0., scale=1.):
        super(HipMomentTCNWithAuxiliary, self).__init__()
        
        # Use ThighIMUPredictor instead of AuxiliaryIMUPredictor
        self.thigh_imu_predictor = ThighIMUPredictor(
            motor_input_size=motor_input_size,
            imu_output_size=predicted_thigh_imu_size,
            hidden_channels=aux_hidden_channels,
            dropout=dropout
        )
        
        # Main TCN input size: original motor data + predicted thigh IMU features
        main_input_size = motor_input_size + predicted_thigh_imu_size
        
        # Main TCN for hip moment prediction
        self.main_tcn = TemporalConvNet(main_input_size, main_tcn_channels, 
                                      kernel_size=kernel_size, dropout=dropout)
        
        # Final linear layer to output hip moments (both hips)
        self.moment_predictor = nn.Linear(main_tcn_channels[-1], 2)
        
        # Normalization parameters
        self.center = center
        self.scale = scale
        
        # Calculate effective history
        self.eff_hist = 187  # Approximately 930ms at 200Hz
        
        self.init_weights()
        
    def init_weights(self):
        self.moment_predictor.weight.data.normal_(0, 0.01)

    def forward(self, motor_data, return_aux_output=False):
        """
        Forward pass with auxiliary thigh IMU prediction task
        
        Args:
            motor_data: (batch_size, 4, sequence_length) - motor encoder data
            return_aux_output: bool - whether to return auxiliary thigh IMU predictions
            
        Returns:
            hip_moments: (batch_size, 2, sequence_length) - predicted hip moments
            aux_thigh_imu_pred: (optional) (batch_size, predicted_thigh_imu_size, sequence_length) - predicted thigh IMU data
        """
        # Normalize input motor data
        motor_data_norm = (motor_data - self.center) / self.scale
        
        # Get auxiliary thigh IMU predictions using ThighIMUPredictor
        predicted_thigh_imu = self.thigh_imu_predictor(motor_data_norm)
        
        # Concatenate original motor data with predicted thigh IMU features
        combined_input = torch.cat([motor_data_norm, predicted_thigh_imu], dim=1)
        
        # Pass through main TCN
        main_features = self.main_tcn(combined_input)
        
        # Reshape for moment prediction
        batch_size, channels, seq_len = main_features.shape
        main_features = main_features.permute(0, 2, 1).contiguous()  # (batch, time, channels)
        main_features = main_features.view(-1, channels)  # (batch*time, channels)
        
        # Predict hip moments
        hip_moments = self.moment_predictor(main_features)  # (batch*time, 2)
        
        # Reshape back to original format
        hip_moments = hip_moments.view(batch_size, seq_len, 2)  # (batch, time, 2)
        hip_moments = hip_moments.permute(0, 2, 1)  # (batch, 2, time)
        
        if return_aux_output:
            return hip_moments, predicted_thigh_imu
        else:
            return hip_moments

    def get_effective_history(self):
        return self.eff_hist

    def predict_thigh_imu_from_motor(self, motor_data):
        """
        Standalone function to predict thigh IMU data from motor data
        Useful for visualization and analysis
        """
        motor_data_norm = (motor_data - self.center) / self.scale
        predicted_thigh_imu = self.thigh_imu_predictor(motor_data_norm)
        return predicted_thigh_imu


# Example usage and testing
if __name__ == "__main__":
    # Model parameters
    motor_input_size = 4  # Motor position and velocity for 2 joints
    predicted_thigh_imu_size = 4  
    
    # Create model with ThighIMUPredictor
    model = HipMomentTCNWithAuxiliary(
        motor_input_size=motor_input_size,
        predicted_thigh_imu_size=predicted_thigh_imu_size,
        aux_hidden_channels=[64, 128, 256, 128],  # ThighIMUPredictor's architecture
        dropout=0.2
    )
    
    # Test with random input
    batch_size = 4
    seq_length = 186
    motor_data = torch.randn(batch_size, motor_input_size, seq_length)
    
    # Forward pass with auxiliary output
    hip_moments, predicted_thigh_imu = model(motor_data, return_aux_output=True)
    
    print("=== Model with ThighIMUPredictor ===")
    print(f"Motor input shape: {motor_data.shape}")
    print(f"Predicted thigh IMU shape: {predicted_thigh_imu.shape}")
    print(f"Hip moments shape: {hip_moments.shape}")
    print(f"Combined input size to main TCN: {motor_input_size + predicted_thigh_imu_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Breakdown of parameters
    thigh_imu_params = sum(p.numel() for p in model.thigh_imu_predictor.parameters())
    main_tcn_params = sum(p.numel() for p in model.main_tcn.parameters())
    moment_predictor_params = sum(p.numel() for p in model.moment_predictor.parameters())
    
    print(f"\nParameter breakdown:")
    print(f"ThighIMUPredictor parameters: {thigh_imu_params:,}")
    print(f"Main TCN parameters: {main_tcn_params:,}")
    print(f"Moment predictor parameters: {moment_predictor_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test auxiliary thigh IMU prediction separately
    predicted_thigh_imu_only = model.predict_thigh_imu_from_motor(motor_data)
    print(f"\nStandalone thigh IMU prediction shape: {predicted_thigh_imu_only.shape}")
    
    # Verify data flow
    print(f"\nData flow verification:")
    print(f"1. Motor data: {motor_data.shape}")
    print(f"2. Predicted thigh IMU (via ThighIMUPredictor): {predicted_thigh_imu.shape}")
    print(f"3. Concatenated input: [{motor_input_size} + {predicted_thigh_imu_size}] = {motor_input_size + predicted_thigh_imu_size} channels")
    print(f"4. Final hip moments: {hip_moments.shape}")
    
    # Test with different IMU output sizes
    print(f"\n=== Testing with different IMU output sizes ===")
    for imu_size in [4, 8, 12, 16]:
        test_model = HipMomentTCNWithAuxiliary(
            motor_input_size=motor_input_size,
            predicted_thigh_imu_size=imu_size,
            aux_hidden_channels=[32, 64, 128, 64],
            dropout=0.2
        )
        test_output = test_model(motor_data)
        print(f"IMU size {imu_size}: Output shape {test_output.shape}")