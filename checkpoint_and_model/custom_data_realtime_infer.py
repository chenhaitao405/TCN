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

# Add model path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.ConvTimeNet_backbone import ConvTimeNet_backbone

class DataProcessor:
    """Data processing class for reading CSV and generating sliding window data"""
    
    def __init__(self, csv_path, window_size=218, stride=5):
        self.csv_path = csv_path
        self.window_size = window_size
        self.stride = stride
        self.data = None
        self.windows = []
        
    def load_data(self):
        """Load CSV data"""
        print(f"Loading data: {self.csv_path}")
        # 读取数据
        self.data = pd.read_csv(self.csv_path)

        # 检查CSV文件中存在的列
        available_columns = self.data.columns.tolist()

        # 定义两对可能的键值
        key_pairs = [
            ['hip_angle_l', 'hip_angle_l_velocity'],
            ['motor_angle_L_float', 'motor_vel_L_float']
        ]

        # 查找存在的键值对
        selected_pair = None
        for pair in key_pairs:
            if all(key in available_columns for key in pair):
                selected_pair = pair
                break

        if selected_pair is None:
            raise ValueError("未找到可用的数据列，请检查CSV文件格式")

        print(f"使用数据列: {selected_pair}")

        # 提取数据
        angle_data = self.data[selected_pair[0]].values
        velocity_data = self.data[selected_pair[1]].values

        # 如果是第二对键值，进行单位转换和归一化
        if selected_pair == ['motor_angle_L_float', 'motor_vel_L_float']:
            # 单位转换
            angle_data = (angle_data*180/np.pi) * -1
            velocity_data = (velocity_data*180/np.pi) * -1
        
        global norm_stats
        # 归一化
        path = 'checkpoint_and_model/norm_stats_5sensors.pkl'
        with open(path, 'rb') as f:
            norm_stats = pickle.load(f)
        
        # angle_data = (angle_data - norm_stats['input_mean'][0]) / norm_stats['input_std'][0]
        # velocity_data = (velocity_data - norm_stats['input_mean'][1]) / norm_stats['input_std'][1]
    
        angle_data = (angle_data ) / norm_stats['input_std'][0]
        velocity_data = (velocity_data) / norm_stats['input_std'][1]

        print(f"Total data length: {len(angle_data)}")
        print(f"Angle data range: [{angle_data.min():.2f}, {angle_data.max():.2f}]")
        print(f"Angular velocity data range: [{velocity_data.min():.2f}, {velocity_data.max():.2f}]")
        
        return angle_data, velocity_data
    
    def load_ground_truth(self, moment_csv_path):
        """Load ground truth torque data from moment_filt.csv"""
        try:
            print(f"Loading ground truth data: {moment_csv_path}")
            moment_data = pd.read_csv(moment_csv_path)
            
            # 检查是否存在hip_flexion_l_moment列
            if 'hip_flexion_l_moment' not in moment_data.columns:
                print(f"Warning: 'hip_flexion_l_moment' column not found in {moment_csv_path}")
                return None
            
            gt_torque = moment_data['hip_flexion_l_moment'].values
            print(f"Ground truth torque data length: {len(gt_torque)}")
            print(f"Ground truth torque range: [{gt_torque.min():.2f}, {gt_torque.max():.2f}]")
            
            return gt_torque
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            return None
    
    def create_sliding_windows(self, angle_data, velocity_data):
        """Create sliding windows"""
        print(f"Creating sliding windows: window size={self.window_size}, stride={self.stride}")
        
        data_length = len(angle_data)
        self.windows = []
        
        for start_idx in range(0, data_length - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            
            # Extract window data
            window_angle = angle_data[start_idx:end_idx]
            window_velocity = velocity_data[start_idx:end_idx]
            
            # Combine into (2, window_size) shape
            window_data = np.stack([window_angle, window_velocity], axis=0)
            
            self.windows.append({
                'data': window_data,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        print(f"Generated {len(self.windows)} windows")
        return self.windows

class TorquePredictor:
    """Torque prediction class"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Butterworth filter parameters
        self.cutoff_freq = 10.0  # 截止频率 3 Hz
        self.sampling_rate = 200.0  # 采样率 200 Hz
        self.filter_order = 2  # 二阶滤波器
        
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
    
    def apply_butterworth_filter(self, data):
        """
        对预测的扭矩数据应用巴特沃斯滤波器
        data: numpy array, shape (n,) or (n, 1)
        """
        print("Applying Butterworth filter to predictions...")
        
        # 确保数据是1D数组
        if len(data.shape) > 1:
            data = data.flatten()
        
        # 设计巴特沃斯滤波器
        nyquist_freq = self.sampling_rate / 2
        normalized_cutoff = self.cutoff_freq / nyquist_freq
        b, a = signal.butter(self.filter_order, normalized_cutoff, btype='low', analog=False)
        
        # 检查数据长度
        if len(data) < 3:  # filtfilt需要至少3个样本点
            print("Warning: Data length too short for filtering, returning original data")
            return data
        
        try:
            # 应用零相位滤波
            filtered_data = signal.filtfilt(b, a, data)
            print(f"Filtering complete. Original range: [{data.min():.2f}, {data.max():.2f}], "
                  f"Filtered range: [{filtered_data.min():.2f}, {filtered_data.max():.2f}]")
            return filtered_data
        except Exception as e:
            print(f"Filter failed, using original data: {e}")
            return data
    
    def predict_batch(self, windows_data):
        """Batch predict torque"""
        print("Starting batch prediction...")
        
        # Convert data to tensor
        batch_data = torch.FloatTensor(windows_data).to(self.device)
        
        print(f"Input data shape: {batch_data.shape}")
        
        with torch.no_grad():
            predictions = self.model(batch_data)
            print(f"Prediction output shape: {predictions.shape}")
            
        return predictions.cpu().numpy()

class Visualizer:
    """Visualization class"""
    
    def __init__(self, output_dir="output_visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def visualize_predictions(self, all_predictions, original_data, stride, window_size, gt_torque=None):
        """Visualize prediction results"""
        print("Generating visualization...")
        
        # Create time axis - fix the issue here
        time_axis_original = np.arange(len(original_data['angle']))
        
        # Create time axis for prediction data
        # Since we take the first stride points from each window, we need to build the corresponding time axis
        pred_time_indices = []
        for i in range(len(all_predictions)):
            if i == 0:
                # First window takes all first stride points
                pred_time_indices.extend(range(stride))
            else:
                # Subsequent windows only take one point at position i * stride
                pred_time_indices.append(i * stride)
        
        # Ensure time axis length matches prediction data
        pred_time_axis = np.array(pred_time_indices[:len(all_predictions)])
        
        # Plot original data and predictions
        fig, axes = plt.subplots(3, 1, figsize=(15, 12),sharex=True)
        
        # 1. Original angle data
        axes[0].plot(time_axis_original, original_data['angle'], 'b-', alpha=0.7, label='Angle', linewidth=1)
        axes[0].set_ylabel('Angle (deg)')
        axes[0].set_title('Original Angle Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Original angular velocity data
        axes[1].plot(time_axis_original, original_data['velocity'], 'g-', alpha=0.7, label='Angular Velocity', linewidth=1)
        axes[1].set_ylabel('Angular Velocity (deg/s)')
        axes[1].set_title('Original Angular Velocity Data')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Predicted torque data (filtered) and ground truth
        # Flatten prediction data
        if len(all_predictions.shape) > 1 and all_predictions.shape[1] == 1:
            torque_predictions = all_predictions.flatten()
        else:
            torque_predictions = all_predictions
        
        axes[2].plot(pred_time_axis+217, torque_predictions, 'r-', alpha=0.8, label='Predicted Torque (Filtered)', linewidth=1.5)
        
        # Plot ground truth if available
        if gt_torque is not None:
            gt_time_axis = np.arange(len(gt_torque))
            axes[2].plot(gt_time_axis, gt_torque, 'b--', alpha=0.6, label='Ground Truth Torque', linewidth=1.5)
        
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Torque (Nm)')
        axes[2].set_title('Predicted Torque vs Ground Truth')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Set consistent x-axis range for easy comparison
        x_min = 0
        x_max = max(time_axis_original[-1], pred_time_axis[-1])
        if gt_torque is not None:
            x_max = max(x_max, len(gt_torque) - 1)
        for ax in axes:
            ax.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'torque_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save prediction data to CSV
        self.save_predictions(all_predictions, pred_time_axis, gt_torque)
        
        print(f"Prediction data time axis range: {pred_time_axis[0]} - {pred_time_axis[-1]}")
        print(f"Original data time axis range: {time_axis_original[0]} - {time_axis_original[-1]}")
        if gt_torque is not None:
            print(f"Ground truth data time axis range: 0 - {len(gt_torque) - 1}")
        
    def save_predictions(self, predictions, time_axis, gt_torque=None):
        """Save prediction results to CSV file"""
        data_dict = {
            'time_step': time_axis,
            'torque_prediction_filtered': predictions.flatten() if len(predictions.shape) > 1 else predictions
        }
        
        # Add ground truth if available
        if gt_torque is not None:
            # Align ground truth with prediction time steps
            gt_aligned = np.full(len(time_axis), np.nan)
            for i, t in enumerate(time_axis):
                if t < len(gt_torque):
                    gt_aligned[i] = gt_torque[int(t)]
            data_dict['ground_truth_torque'] = gt_aligned
        
        df = pd.DataFrame(data_dict)
        csv_path = self.output_dir / 'torque_predictions.csv'
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
    filter_order = 1  # 二阶滤波器
    
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
    csv_path = "outputs_csv/hip_moment_data_20251027_160915.csv"  # Replace with your CSV file path
    # csv_path = 'wgg_sensor_data_20251017_161510.csv'
    # csv_path = '/home/zbb/Documents/xwechat_files/wxid_nzwhj3rjmx7l12_31e6/msg/file/2025-10/25.9.28.端到端数据采集/kuaizou/motorAngle.csv'
    
    csv_path ='checkpoint_and_model/BT01_normal_walk_1_1-2_on_exo.csv'
    # csv_path = '/home/xietao/下载/knee_e2e/capsule-5421243-code/data_raw/Phase1And2_Parsed/Parsed/BT01/sit_to_stand_2_tall-noarm_on/BT01_sit_to_stand_2_tall-noarm_on_exo.csv'
    model_path = 'checkpoint_and_model/best_model1021.pth'  # Replace with your model path, keep None if not available
    window_size = 218
    stride = 1
    
    # Generate moment_filt.csv path
    moment_csv_path = csv_path.replace('_exo.csv', '_moment_filt.csv')
    print(f"Ground truth CSV path: {moment_csv_path}")
    
    # 1. Data processing
    processor = DataProcessor(csv_path, window_size, stride)
    angle_data, velocity_data = processor.load_data()
    
    # Load ground truth torque data
    gt_torque = processor.load_ground_truth(moment_csv_path)

    windows = processor.create_sliding_windows(angle_data, velocity_data)
    
    # Prepare batch data
    windows_data = np.array([window['data'] for window in windows])
    print(f"Batch data shape: {windows_data.shape}")
    
    # 2. Model prediction
    predictor = TorquePredictor(model_path)
    batch_predictions = predictor.predict_batch(windows_data)
    
    # 3. Process prediction results - take the first stride prediction points from each window
    all_predictions = []
    all_predictions = []
    for i, prediction in enumerate(batch_predictions):
        # 取最后一个时间步，对应当前时刻预测
        selected_pred = prediction[:, -2:-1].T  # ✅ 取最后一个点
        
        all_predictions.append(selected_pred)
    
    # Merge all predictions
    all_predictions = np.vstack(all_predictions)
    print(f"Merged prediction data shape: {all_predictions.shape}")
    
    # 4. Apply Butterworth filter to predictions
    all_predictions_filtered = predictor.apply_butterworth_filter(all_predictions)
    
    # 5. Visualization
    original_data = {
        'angle': angle_data,
        'velocity': velocity_data
    }
    
    visualizer = Visualizer()
    visualizer.visualize_predictions(all_predictions_filtered, original_data, stride, window_size, gt_torque)
    
    print("Program execution completed!")

if __name__ == "__main__":
    main()