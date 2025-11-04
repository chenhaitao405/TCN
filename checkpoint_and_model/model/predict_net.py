import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ================== 核心模型定义 ==================

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


class PhysicsInformedLayer(nn.Module):
    """物理信息层 - 提取运动学特征"""
    def __init__(self, input_size=4, output_size=32):
        super(PhysicsInformedLayer, self).__init__()
        
        # 提取不同阶的差分特征
        self.diff1_conv = nn.Conv1d(input_size, output_size//4, kernel_size=2, padding=0)
        self.diff2_conv = nn.Conv1d(input_size, output_size//4, kernel_size=3, padding=0)
        
        # 原始特征变换
        self.feat_conv = nn.Conv1d(input_size, output_size//2, kernel_size=3, padding=1)
        
        self.bn = nn.BatchNorm1d(output_size)
        
    def forward(self, x):
        # 一阶差分（速度变化）
        diff1 = F.pad(self.diff1_conv(x), (0, 1), mode='replicate')
        
        # 二阶差分（加速度特征）
        diff2 = F.pad(self.diff2_conv(x), (0, 2), mode='replicate')
        
        # 原始特征
        feat = self.feat_conv(x)
        
        # 组合所有特征
        out = torch.cat([diff1, diff2, feat], dim=1)
        
        return self.bn(out)


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


class SimplifiedIMUPredictor(nn.Module):
    """
    简化版IMU预测器
    输入: 4维电机数据 (左右髋关节角度+角速度)
    输出: 4维IMU数据 (左右大腿xy轴加速度)
    """
    def __init__(self, 
                 motor_input_size=4,    # 2个关节 × (角度+角速度)
                 imu_output_size=4,     # 2个大腿 × 2个加速度轴(x,y)
                 hidden_channels=[32, 64, 128, 64],
                 dropout=0.2):
        super(SimplifiedIMUPredictor, self).__init__()
        
        self.motor_input_size = motor_input_size
        self.imu_output_size = imu_output_size
        
        # 1. 物理特征提取
        self.physics_layer = PhysicsInformedLayer(motor_input_size, hidden_channels[0])
        
        # 2. 多尺度特征提取
        self.multi_scale = MultiScaleConv(hidden_channels[0], hidden_channels[1])
        
        # 3. 时序卷积块（使用不同的dilation捕捉不同时间尺度）
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
            motor_data: (batch_size, 4, seq_length)
                        [left_hip_angle, right_hip_angle, left_hip_velocity, right_hip_velocity]
        Returns:
            imu_predictions: (batch_size, 4, seq_length)
                           [left_thigh_acc_x, left_thigh_acc_y, right_thigh_acc_x, right_thigh_acc_y]
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


# ================== 训练和评估工具 ==================

class ThighAccelerationLoss(nn.Module):
    """专门针对大腿加速度预测的损失函数"""
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        super(ThighAccelerationLoss, self).__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # 平滑度权重
        self.gamma = gamma  # 峰值权重
        
    def forward(self, pred, target):
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target)
        
        # 平滑度损失（预测的加速度应该相对平滑）
        pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        smoothness_loss = F.mse_loss(pred_diff, target_diff)
        
        # 峰值损失（确保捕捉加速度峰值）
        pred_peaks = torch.abs(pred)
        target_peaks = torch.abs(target)
        peak_loss = F.mse_loss(pred_peaks, target_peaks)
        
        total_loss = self.alpha * mse_loss + self.beta * smoothness_loss + self.gamma * peak_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'smoothness': smoothness_loss.item(),
            'peak': peak_loss.item()
        }


# ================== 完整使用示例 ==================

class ThighAccelerationPredictor:
    """完整的大腿加速度预测系统"""
    
    def __init__(self, model=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        if model is None:
            self.model = SimplifiedIMUPredictor(
                motor_input_size=4,
                imu_output_size=4,
                hidden_channels=[32, 64, 128, 64],
                dropout=0.0  # 推理时不需要dropout
            )
        else:
            self.model = model
            
        self.model.to(self.device)
        self.model.eval()
        
        # 数据归一化参数
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        
    def set_normalization_params(self, input_stats, output_stats):
        """设置归一化参数"""
        self.input_mean = torch.tensor(input_stats['mean']).to(self.device)
        self.input_std = torch.tensor(input_stats['std']).to(self.device)
        self.output_mean = torch.tensor(output_stats['mean']).to(self.device)
        self.output_std = torch.tensor(output_stats['std']).to(self.device)
        
    def preprocess(self, motor_data):
        """预处理输入数据"""
        # 转换为tensor
        if not isinstance(motor_data, torch.Tensor):
            motor_data = torch.FloatTensor(motor_data)
            
        # 确保维度正确
        if motor_data.dim() == 2:
            motor_data = motor_data.unsqueeze(0)  # 添加batch维度
            
        # 移动到设备
        motor_data = motor_data.to(self.device)
        
        # 归一化
        if self.input_mean is not None:
            motor_data = (motor_data - self.input_mean.view(1, -1, 1)) / self.input_std.view(1, -1, 1)
            
        return motor_data
    
    def postprocess(self, predictions):
        """后处理预测结果"""
        # 反归一化
        if self.output_mean is not None:
            predictions = predictions * self.output_std.view(1, -1, 1) + self.output_mean.view(1, -1, 1)
            
        return predictions
    
    def predict(self, motor_data):
        """
        预测大腿加速度
        
        Args:
            motor_data: numpy array (4, seq_length) or (batch, 4, seq_length)
                       [left_hip_angle, right_hip_angle, left_hip_velocity, right_hip_velocity]
        
        Returns:
            predictions: numpy array (4, seq_length) or (batch, 4, seq_length)
                        [left_thigh_acc_x, left_thigh_acc_y, right_thigh_acc_x, right_thigh_acc_y]
        """
        # 预处理
        motor_tensor = self.preprocess(motor_data)
        
        # 推理
        with torch.no_grad():
            predictions = self.model(motor_tensor)
            
        # 后处理
        predictions = self.postprocess(predictions)
        
        # 转换回numpy
        return predictions.cpu().numpy()
    
    def visualize_prediction(self, motor_data, ground_truth=None):
        """可视化预测结果"""
        predictions = self.predict(motor_data)
        
        # 如果输入是批次，取第一个样本
        if predictions.ndim == 3:
            predictions = predictions[0]
        if motor_data.ndim == 3:
            motor_data = motor_data[0]
        if ground_truth is not None and ground_truth.ndim == 3:
            ground_truth = ground_truth[0]
            
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        
        # 输入数据可视化
        axes[0, 0].plot(motor_data[0], label='Left Hip Angle')
        axes[0, 0].plot(motor_data[1], label='Right Hip Angle')
        axes[0, 0].set_title('Hip Joint Angles')
        axes[0, 0].set_xlabel('Time steps')
        axes[0, 0].set_ylabel('Angle (rad)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(motor_data[2], label='Left Hip Velocity')
        axes[0, 1].plot(motor_data[3], label='Right Hip Velocity')
        axes[0, 1].set_title('Hip Joint Velocities')
        axes[0, 1].set_xlabel('Time steps')
        axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 左腿加速度预测
        axes[1, 0].plot(predictions[0], label='Predicted X', color='blue')
        axes[1, 0].plot(predictions[1], label='Predicted Y', color='red')
        if ground_truth is not None:
            axes[1, 0].plot(ground_truth[0], '--', label='True X', color='blue', alpha=0.5)
            axes[1, 0].plot(ground_truth[1], '--', label='True Y', color='red', alpha=0.5)
        axes[1, 0].set_title('Left Thigh Acceleration')
        axes[1, 0].set_xlabel('Time steps')
        axes[1, 0].set_ylabel('Acceleration (m/s²)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 右腿加速度预测
        axes[1, 1].plot(predictions[2], label='Predicted X', color='green')
        axes[1, 1].plot(predictions[3], label='Predicted Y', color='orange')
        if ground_truth is not None:
            axes[1, 1].plot(ground_truth[2], '--', label='True X', color='green', alpha=0.5)
            axes[1, 1].plot(ground_truth[3], '--', label='True Y', color='orange', alpha=0.5)
        axes[1, 1].set_title('Right Thigh Acceleration')
        axes[1, 1].set_xlabel('Time steps')
        axes[1, 1].set_ylabel('Acceleration (m/s²)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 误差分析
        if ground_truth is not None:
            error = np.abs(predictions - ground_truth)
            axes[2, 0].plot(error[0], label='Left X Error')
            axes[2, 0].plot(error[1], label='Left Y Error')
            axes[2, 0].plot(error[2], label='Right X Error')
            axes[2, 0].plot(error[3], label='Right Y Error')
            axes[2, 0].set_title('Absolute Errors')
            axes[2, 0].set_xlabel('Time steps')
            axes[2, 0].set_ylabel('Error (m/s²)')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            
            # 误差统计
            mae = np.mean(error, axis=1)
            rmse = np.sqrt(np.mean(error**2, axis=1))
            axes[2, 1].bar(['Left X', 'Left Y', 'Right X', 'Right Y'], mae, alpha=0.7, label='MAE')
            axes[2, 1].bar(['Left X', 'Left Y', 'Right X', 'Right Y'], rmse, alpha=0.7, label='RMSE')
            axes[2, 1].set_title('Error Statistics')
            axes[2, 1].set_ylabel('Error (m/s²)')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


# ================== 训练脚本示例 ==================

def train_model(train_loader, val_loader, epochs=100, lr=1e-3, device='cuda'):
    """训练模型的完整示例"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = SimplifiedIMUPredictor(
        motor_input_size=4,
        imu_output_size=4,
        hidden_channels=[32, 64, 128, 64],
        dropout=0.2
    ).to(device)
    
    # 损失函数和优化器
    criterion = ThighAccelerationLoss(alpha=1.0, beta=0.1, gamma=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {device}")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (motor_data, imu_target) in enumerate(train_loader):
            motor_data = motor_data.to(device)
            imu_target = imu_target.to(device)
            
            optimizer.zero_grad()
            predictions = model(motor_data)
            loss, components = criterion(predictions, imu_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for motor_data, imu_target in val_loader:
                motor_data = motor_data.to(device)
                imu_target = imu_target.to(device)
                
                predictions = model(motor_data)
                loss, _ = criterion(predictions, imu_target)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return model, train_losses, val_losses


# ================== 使用示例 ==================

if __name__ == "__main__":
    print("="*50)
    print("Simplified IMU Predictor for Thigh Acceleration")
    print("="*50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模拟数据
    batch_size = 32
    seq_length = 100
    
    # 生成模拟的电机数据 (髋关节角度和角速度)
    t = np.linspace(0, 2*np.pi, seq_length)
    
    # 模拟行走时的髋关节运动
    left_hip_angle = np.sin(t) * 0.5  # 左髋关节角度
    right_hip_angle = np.sin(t + np.pi) * 0.5  # 右髋关节角度（相位差180度）
    left_hip_velocity = np.cos(t) * 0.5  # 左髋关节角速度
    right_hip_velocity = np.cos(t + np.pi) * 0.5  # 右髋关节角速度
    
    # 组合成输入数据
    motor_data = np.stack([left_hip_angle, right_hip_angle, left_hip_velocity, right_hip_velocity])
    motor_data = motor_data[np.newaxis, :, :]  # 添加batch维度
    
    # 模拟的目标加速度数据（实际应该从IMU获取）
    # 这里用简单的导数关系模拟
    left_acc_x = -np.sin(t) * 0.5 + np.random.normal(0, 0.01, seq_length)
    left_acc_y = -np.cos(t) * 0.3 + np.random.normal(0, 0.01, seq_length)
    right_acc_x = -np.sin(t + np.pi) * 0.5 + np.random.normal(0, 0.01, seq_length)
    right_acc_y = -np.cos(t + np.pi) * 0.3 + np.random.normal(0, 0.01, seq_length)
    
    ground_truth = np.stack([left_acc_x, left_acc_y, right_acc_x, right_acc_y])
    ground_truth = ground_truth[np.newaxis, :, :]
    
    print(f"Input shape: {motor_data.shape}")  # (1, 4, 100)
    print(f"Output shape: {ground_truth.shape}")  # (1, 4, 100)
    
    # 初始化预测器
    predictor = ThighAccelerationPredictor()
    
    # 进行预测
    predictions = predictor.predict(motor_data)
    print(f"Prediction shape: {predictions.shape}")
    
    # 可视化结果
    print("\nVisualizing predictions...")
    predictor.visualize_prediction(motor_data, ground_truth)
    
    # 计算误差
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(np.mean((predictions - ground_truth)**2))
    print(f"\nError Metrics:")
    print(f"MAE: {mae:.4f} m/s²")
    print(f"RMSE: {rmse:.4f} m/s²")
    
    # 保存和加载模型示例
    print("\n" + "="*50)
    print("Model Save/Load Example")
    print("="*50)
    
    # 保存模型
    checkpoint = {
        'model_state_dict': predictor.model.state_dict(),
        'config': {
            'motor_input_size': 4,
            'imu_output_size': 4,
            'hidden_channels': [32, 64, 128, 64],
            'dropout': 0.0
        },
        'normalization': {
            'input_mean': [0, 0, 0, 0],
            'input_std': [1, 1, 1, 1],
            'output_mean': [0, 0, 0, 0],
            'output_std': [1, 1, 1, 1]
        }
    }
    
    torch.save(checkpoint, 'thigh_acceleration_model.pth')
    print("Model saved to 'thigh_acceleration_model.pth'")
    
    # 加载模型
    loaded_checkpoint = torch.load('thigh_acceleration_model.pth', map_location='cpu')
    new_model = SimplifiedIMUPredictor(**loaded_checkpoint['config'])
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    new_model.eval()
    print("Model loaded successfully!")
    
    # 使用加载的模型进行预测
    new_predictor = ThighAccelerationPredictor(model=new_model)
    new_predictions = new_predictor.predict(motor_data)
    
    # 验证预测结果一致
    prediction_diff = np.mean(np.abs(predictions - new_predictions))
    print(f"Difference between original and loaded model predictions: {prediction_diff:.6f}")
    print("(Should be close to 0 if loading was successful)")