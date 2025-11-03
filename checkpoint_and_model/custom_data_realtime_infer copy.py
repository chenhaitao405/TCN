import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import pickle 
from scipy import signal
from scipy.interpolate import interp1d  # 添加插值库

# Add model path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.ConvTimeNet_backbone import ConvTimeNet_backbone

class DataProcessor:
    """Data processing class for reading CSV and generating sliding window data"""
    
    def __init__(self, csv_path, window_size=109, stride=5, target_size=218):  # 修改默认窗口大小
        self.csv_path = csv_path
        self.window_size = window_size
        self.target_size = target_size  # 目标大小（模型期望的输入大小）
        self.stride = stride
        self.data = None
        self.left_windows = []
        self.right_windows = []
        
    def interpolate_window(self, window_data):
        """将窗口数据从window_size插值到target_size"""
        original_length = window_data.shape[1]  # 应该是window_size
        target_length = self.target_size
        
        if original_length == target_length:
            return window_data  # 不需要插值
        
        # 创建原始时间轴和目标时间轴
        original_time = np.linspace(0, 1, original_length)
        target_time = np.linspace(0, 1, target_length)
        
        # 对每个通道进行插值
        interpolated_data = np.zeros((window_data.shape[0], target_length))
        for i in range(window_data.shape[0]):
            interpolator = interp1d(original_time, window_data[i, :], kind='linear', 
                                   fill_value="extrapolate")
            interpolated_data[i, :] = interpolator(target_time)
        
        return interpolated_data
    
    def load_data(self):
        """Load CSV data including both left and right leg data"""
        print(f"Loading data: {self.csv_path}")
        # 读取数据
        self.data = pd.read_csv(self.csv_path)

        # 检查CSV文件中存在的列
        available_columns = self.data.columns.tolist()

        # 定义左右腿可能的键值
        left_key_pairs = [
            ['hip_angle_l', 'hip_angle_l_velocity'],
            ['motor_angle_L_float', 'motor_vel_L_float']
        ]
        
        right_key_pairs = [
            ['hip_angle_r', 'hip_angle_r_velocity'],
            ['motor_angle_R_float', 'motor_vel_R_float']
        ]

        # 查找存在的左腿键值对
        left_selected_pair = None
        for pair in left_key_pairs:
            if all(key in available_columns for key in pair):
                left_selected_pair = pair
                break

        # 查找存在的右腿键值对
        right_selected_pair = None
        for pair in right_key_pairs:
            if all(key in available_columns for key in pair):
                right_selected_pair = pair
                break

        if left_selected_pair is None and right_selected_pair is None:
            raise ValueError("未找到可用的数据列，请检查CSV文件格式")

        print(f"左腿使用数据列: {left_selected_pair}")
        print(f"右腿使用数据列: {right_selected_pair}")

        # 加载归一化参数
        global norm_stats
        path = 'checkpoint_and_model/norm_stats_5sensors.pkl'
        with open(path, 'rb') as f:
            norm_stats = pickle.load(f)

        # 处理左腿数据
        left_angle_data = None
        left_velocity_data = None
        if left_selected_pair is not None:
            left_angle_data = self.data[left_selected_pair[0]].values
            left_velocity_data = self.data[left_selected_pair[1]].values
            
            # 单位转换和归一化
            left_angle_data = (left_angle_data * 180 / np.pi) * -1
            left_velocity_data = (left_velocity_data * 180 / np.pi) * -1
            
            left_angle_data = left_angle_data / norm_stats['input_std'][0]
            left_velocity_data = left_velocity_data / norm_stats['input_std'][1]

        # 处理右腿数据
        right_angle_data = None
        right_velocity_data = None
        if right_selected_pair is not None:
            right_angle_data = self.data[right_selected_pair[0]].values
            right_velocity_data = self.data[right_selected_pair[1]].values
            
            # 单位转换和归一化
            right_angle_data = (right_angle_data * 180 / np.pi) * -1
            right_velocity_data = (right_velocity_data * 180 / np.pi) * -1
            
            right_angle_data = right_angle_data / norm_stats['input_std'][0]
            right_velocity_data = right_velocity_data / norm_stats['input_std'][1]

        # 打印数据信息
        if left_angle_data is not None:
            print(f"左腿数据长度: {len(left_angle_data)}")
            print(f"左腿角度数据范围: [{left_angle_data.min():.2f}, {left_angle_data.max():.2f}]")
            print(f"左腿角速度数据范围: [{left_velocity_data.min():.2f}, {left_velocity_data.max():.2f}]")
        
        if right_angle_data is not None:
            print(f"右腿数据长度: {len(right_angle_data)}")
            print(f"右腿角度数据范围: [{right_angle_data.min():.2f}, {right_angle_data.max():.2f}]")
            print(f"右腿角速度数据范围: [{right_velocity_data.min():.2f}, {right_velocity_data.max():.2f}]")
        
        return left_angle_data, left_velocity_data, right_angle_data, right_velocity_data
    
    def create_sliding_windows(self, left_angle_data, left_velocity_data, right_angle_data, right_velocity_data):
        """Create sliding windows for left and right legs with interpolation"""
        print(f"Creating sliding windows: window size={self.window_size}, target size={self.target_size}, stride={self.stride}")
        
        # 创建左腿滑动窗口
        self.left_windows = []
        if left_angle_data is not None:
            data_length = len(left_angle_data)
            for start_idx in range(0, data_length - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                window_angle = left_angle_data[start_idx:end_idx]
                window_velocity = left_velocity_data[start_idx:end_idx]
                window_data = np.stack([window_angle, window_velocity], axis=0)
                
                # 插值到目标大小
                interpolated_data = self.interpolate_window(window_data)
                
                self.left_windows.append({
                    'data': interpolated_data,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            print(f"Generated {len(self.left_windows)} left leg windows")

        # 创建右腿滑动窗口
        self.right_windows = []
        if right_angle_data is not None:
            data_length = len(right_angle_data)
            for start_idx in range(0, data_length - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                window_angle = right_angle_data[start_idx:end_idx]
                window_velocity = right_velocity_data[start_idx:end_idx]
                window_data = np.stack([window_angle, window_velocity], axis=0)
                
                # 插值到目标大小
                interpolated_data = self.interpolate_window(window_data)
                
                self.right_windows.append({
                    'data': interpolated_data,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            print(f"Generated {len(self.right_windows)} right leg windows")
        
        return self.left_windows, self.right_windows

class TorquePredictor:
    """Torque prediction class"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path):
        """Load trained model"""
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_state_dict = checkpoint['model_state_dict']
        else:
            print("No model path provided or file does not exist, using randomly initialized model")
            model_state_dict = None
        
        # Create model structure (same as during training)
        model = ConvTimeNet_backbone(
            c_in=2,
            n_layers=4,
            seq_len=218,
            context_window = 218,
            target_window = 218,
            patch_len= 32,
            stride=16,
            d_model=64,
            d_ff=128,
            dropout=0.2,
            act="gelu",
            enable_res_param=False,
            dw_ks=[5,5,7,7,13,13,19,19,],  # Depth-wise kernel sizes for each layer
            norm='batch',
            re_param=False,
            deformable=True,
            reduced_channels=16,
            revin = False,
            final_out=1,
        ).to(self.device)
        
        if model_state_dict:
            model.load_state_dict(model_state_dict)
            print("Model loaded successfully!")
        else:
            print("Using randomly initialized weights")


        return model
    
    def predict_batch(self, windows_data):
        """Batch predict torque"""
        if len(windows_data) == 0:
            return np.array([])
            
        print("Starting batch prediction...")
        
        # Convert data to tensor
        batch_data = torch.FloatTensor(windows_data).to(self.device)
        print(f"Input data shape before filtering: {batch_data.shape}")
        
        # Apply Butterworth filter to angular velocity (dimension 1)
        batch_data_np = batch_data.cpu().numpy()
        
        # Filter parameters
        cutoff_freq = 10.0  # 截止频率 10 Hz
        sampling_rate = 200.0  # 采样率 200 Hz
        filter_order = 2  # 二阶滤波器
        
        # Design Butterworth filter
        nyquist_freq = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        b, a = signal.butter(filter_order, normalized_cutoff, btype='low', analog=False)
        
        # Apply filter to each batch sample's angular velocity channel
        for i in range(batch_data_np.shape[0]):
            # Filter angular velocity channel (index 1)
            batch_data_np[i, 1, :] = signal.filtfilt(b, a, batch_data_np[i, 1, :])
        
        # Convert back to tensor
        batch_data = torch.FloatTensor(batch_data_np).to(self.device)
        print(f"Input data shape after filtering: {batch_data.shape}")


        import time
        t1 = time.time()
        with torch.no_grad():
            predictions = self.model(batch_data)
            print(f"Prediction output shape: {predictions.shape}")
        print(f'{time.time() - t1} s')


        predictions = predictions.cpu().numpy()
        predictions[:,-30:] = signal.filtfilt(b, a, predictions[:,-30:])
        return predictions

class Visualizer:
    """Visualization class"""
    
    def __init__(self, output_dir="output_visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def find_intersections_mean(self,curve1, curve2):

        # 确保输入是一维数组
        y1 = curve1.flatten()
        y2 = curve2.flatten()
        x = np.arange(len(y1))
        
        # 计算差值
        diff = y1 - y2
        
        # 找到符号变化的点（可能包含交点）
        sign_changes = np.where(np.diff(np.signbit(diff)))[0]
        
        intersections = []
        
        for idx in sign_changes:
            # 线性插值找到精确交点
            x1, x2 = x[idx], x[idx+1]
            y11, y12 = y1[idx], y1[idx+1]
            y21, y22 = y2[idx], y2[idx+1]
            
            # 解线性方程组求交点
            # y1 = a1*x + b1, y2 = a2*x + b2
            a1 = (y12 - y11) / (x2 - x1)
            b1 = y11 - a1 * x1
            a2 = (y22 - y21) / (x2 - x1)
            b2 = y21 - a2 * x1
            
            # 解交点: a1*x + b1 = a2*x + b2
            if a1 != a2:  # 避免平行线
                x_intersect = (b2 - b1) / (a1 - a2)
                y_intersect = a1 * x_intersect + b1
                
                # 确保交点在当前线段内
                if x1 <= x_intersect <= x2:
                    intersections.append((x_intersect, y_intersect))
        
        if intersections:
            intersections = np.array(intersections)
            x_mean = np.mean(intersections[:, 0])
            y_mean = np.mean(intersections[:, 1])
            return intersections, (x_mean, y_mean)
        else:
            return None, None
        
    def visualize_predictions(self, left_predictions, right_predictions, original_data, stride, window_size):
        """Visualize prediction results for both legs"""
        print("Generating visualization...")
        
        # Create time axis
        time_axis_original = np.arange(len(original_data['left_angle'] if original_data['left_angle'] is not None else original_data['right_angle']))
        
        # Create time axis for prediction data
        pred_time_indices = []
        num_predictions = max(len(left_predictions) if left_predictions is not None else 0, 
                             len(right_predictions) if right_predictions is not None else 0)
        
        # mean = self.find_intersections_mean(left_predictions, right_predictions)
        # print(mean)
        for i in range(num_predictions):
            if i == 0:
                pred_time_indices.extend(range(stride))
            else:
                pred_time_indices.append(i * stride)
        
        pred_time_axis = np.array(pred_time_indices[:num_predictions])
        
        # Plot original data and predictions
        fig, axes = plt.subplots(3, 1, figsize=(15, 16),sharex=True)
        
        # 1. Original angle data
        if original_data['left_angle'] is not None:
            axes[0].plot(time_axis_original, original_data['left_angle'], 'b-', alpha=0.7, label='Left Angle', linewidth=1)
        if original_data['right_angle'] is not None:
            axes[0].plot(time_axis_original, original_data['right_angle'], 'r-', alpha=0.7, label='Right Angle', linewidth=1)
        axes[0].set_ylabel('Angle (deg)')
        axes[0].set_title('Original Angle Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Original angular velocity data
        if original_data['left_velocity'] is not None:
            axes[1].plot(time_axis_original, original_data['left_velocity'], 'g-', alpha=0.7, label='Left Angular Velocity', linewidth=1)
        if original_data['right_velocity'] is not None:
            axes[1].plot(time_axis_original, original_data['right_velocity'], 'm-', alpha=0.7, label='Right Angular Velocity', linewidth=1)
        axes[1].set_ylabel('Angular Velocity (deg/s)')
        axes[1].set_title('Original Angular Velocity Data')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Predicted torque data for both legs
        if left_predictions is not None and len(left_predictions) > 0:
            left_torque = left_predictions.flatten() if len(left_predictions.shape) > 1 else left_predictions
            axes[2].plot(pred_time_axis + 218, left_torque , 'b-', alpha=0.8, label='Left Predicted Torque', linewidth=1.5)
        
        if right_predictions is not None and len(right_predictions) > 0:
            right_torque = right_predictions.flatten() if len(right_predictions.shape) > 1 else right_predictions
            axes[2].plot(pred_time_axis + 218, right_torque , 'r-', alpha=0.8, label='Right Predicted Torque', linewidth=1.5)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('Torque')
        axes[2].set_title('Predicted Torque Data - Separate')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        

        
        # Set consistent x-axis range for easy comparison
        x_min = 0
        x_max = max(time_axis_original[-1], pred_time_axis[-1] + 218 if len(pred_time_axis) > 0 else time_axis_original[-1])
        for ax in axes:
            ax.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'torque_predictions_both_legs.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save prediction data to CSV
        self.save_predictions(left_predictions, right_predictions, pred_time_axis)
        
        print(f"Prediction data time axis range: {pred_time_axis[0] if len(pred_time_axis) > 0 else 0} - {pred_time_axis[-1] if len(pred_time_axis) > 0 else 0}")
        print(f"Original data time axis range: {time_axis_original[0]} - {time_axis_original[-1]}")
        
    def save_predictions(self, left_predictions, right_predictions, time_axis):
        """Save prediction results to CSV file"""
        data_dict = {'time_step': time_axis}
        
        if left_predictions is not None and len(left_predictions) > 0:
            left_torque = left_predictions.flatten() if len(left_predictions.shape) > 1 else left_predictions
            # Ensure same length
            min_length = min(len(time_axis), len(left_torque))
            data_dict['left_torque_prediction'] = left_torque[:min_length]
        
        if right_predictions is not None and len(right_predictions) > 0:
            right_torque = right_predictions.flatten() if len(right_predictions.shape) > 1 else right_predictions
            # Ensure same length
            min_length = min(len(time_axis), len(right_torque))
            data_dict['right_torque_prediction'] = right_torque[:min_length]
        
        df = pd.DataFrame(data_dict)
        csv_path = self.output_dir / 'torque_predictions_both_legs.csv'
        df.to_csv(csv_path, index=False)
        print(f"Prediction results saved to: {csv_path}")

def apply_filter(buffer_array):
    """
    对缓冲区数据应用巴特沃斯滤波器
    buffer_array: shape (n, 2), n <= 30
    """
    # 滤波器参数
    cutoff_freq = 10.0  # 截止频率 10 Hz
    sampling_rate = 200.0  # 采样率 200 Hz
    filter_order = 2  # 二阶滤波器
    
    # 设计巴特沃斯滤波器
    nyquist_freq = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(filter_order, normalized_cutoff, btype='low', analog=False)
    if buffer_array.shape[0] < 3:  # filtfilt需要至少3个样本点
        return buffer_array
    
    filtered_data = np.zeros_like(buffer_array)
    
    # 对左右髋关节力矩分别进行滤波
    try:
        # 滤波左髋关节力矩
        filtered_data[:, 0] = signal.filtfilt(b, a, buffer_array[:, 0])
        # 滤波右髋关节力矩
        filtered_data[:, 1] = signal.filtfilt(b, a, buffer_array[:, 1])
    except Exception as e:
        print(f"Filter failed, using original data: {e}")
        return buffer_array
    
    return filtered_data
    
def main():
    # Configuration parameters
    csv_path = 'wgg_sensor_data_20251027_171509.csv'  # 替换为您的CSV文件路径
    model_path = 'checkpoint_and_model/best_model1021.pth'  # 替换为您的模型路径
    window_size = 109  # 修改为109
    target_size = 218  # 模型期望的输入大小
    stride = 1
    
    # 1. Data processing
    processor = DataProcessor(csv_path, window_size, stride, target_size)
    left_angle_data, left_velocity_data, right_angle_data, right_velocity_data = processor.load_data()

    left_windows, right_windows = processor.create_sliding_windows(
        left_angle_data, left_velocity_data, right_angle_data, right_velocity_data
    )
    
    # 2. Model prediction
    predictor = TorquePredictor(model_path)
    
    # 预测左腿扭矩
    left_predictions = None
    if len(left_windows) > 0:
        left_windows_data = np.array([window['data'] for window in left_windows])
        print(f"Left leg batch data shape: {left_windows_data.shape}")
        left_batch_predictions = predictor.predict_batch(left_windows_data)
        
        # Process left leg prediction results
        left_all_predictions = []
        for i, prediction in enumerate(left_batch_predictions):
            selected_pred = prediction[:, -2:-1].T  # 取最后一个点
            selected_pred = selected_pred * 1     # 应用缩放因子
            left_all_predictions.append(selected_pred)
        
        left_predictions = np.vstack(left_all_predictions)
        print(f"Left leg merged prediction data shape: {left_predictions.shape}")
    
    # 预测右腿扭矩
    right_predictions = None
    if len(right_windows) > 0:
        right_windows_data = np.array([window['data'] for window in right_windows])
        print(f"Right leg batch data shape: {right_windows_data.shape}")
        right_batch_predictions = predictor.predict_batch(right_windows_data)
        
        # Process right leg prediction results
        right_all_predictions = []
        for i, prediction in enumerate(right_batch_predictions):
            selected_pred = prediction[:, -1:].T  # 取最后一个点
            selected_pred = selected_pred * 1     # 应用缩放因子
            right_all_predictions.append(selected_pred)
        
        right_predictions = np.vstack(right_all_predictions)
        print(f"Right leg merged prediction data shape: {right_predictions.shape}")
    
    # 3. Visualization
    original_data = {
        'left_angle': left_angle_data,
        'left_velocity': left_velocity_data,
        'right_angle': right_angle_data,
        'right_velocity': right_velocity_data
    }
    
    visualizer = Visualizer()
    visualizer.visualize_predictions(left_predictions, right_predictions, original_data, stride, window_size)
    
    print("Program execution completed!")

if __name__ == "__main__":
    main()