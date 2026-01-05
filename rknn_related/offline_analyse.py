import numpy as np
import sys
sys.path.append(".")
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from scipy import signal
from utils.config_utils import ConfigManager
from utils.model_loader import ModelLoader


def create_argument_parser():
    """Create and configure argument parser for validation."""
    parser = argparse.ArgumentParser(description='Offline analysis of recorded IMU data')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='rknn_related/record_data.bin',
                        help='Path to recorded binary data file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    parser.add_argument('--interval', type=int, default=10,
                        help='Animation interval in ms (simulates real-time)')
    parser.add_argument('--history_len', type=int, default=500,
                        help='Number of frames to show in plot history')
    return parser


class RealtimeButterworthFilter:
    """实时巴特沃斯低通滤波器"""
    def __init__(self, order=2, cutoff_freq=10.0, sample_rate=100.0):
        """
        初始化滤波器
        Args:
            order: 滤波器阶数
            cutoff_freq: 截止频率 (Hz)
            sample_rate: 采样率 (Hz)
        """
        # 计算归一化截止频率 (Nyquist频率的比例)
        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff_freq / nyquist
        
        # 设计巴特沃斯滤波器
        self.b, self.a = signal.butter(order, normalized_cutoff, btype='low')
        
        # 初始化滤波器状态 (用于实时滤波)
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False
        
    def filter(self, value):
        """
        对单个值进行滤波
        Args:
            value: 输入值
        Returns:
            滤波后的值
        """
        if not self.initialized:
            # 第一个值，用它来初始化滤波器状态
            self.zi = self.zi * value
            self.initialized = True
        
        # 使用lfilter进行实时滤波
        filtered, self.zi = signal.lfilter(self.b, self.a, [value], zi=self.zi)
        return filtered[0]
    
    def reset(self):
        """重置滤波器状态"""
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load recorded data
    print(f"\nLoading data from: {args.data_path}")
    with open(args.data_path, "rb") as f:
        raw_data = np.fromfile(f, dtype=np.float32)
    
    elements_per_frame = 8 * 280
    inputs = raw_data.reshape(-1, elements_per_frame+1)[:, :elements_per_frame]
    quan_output = raw_data.reshape(-1, elements_per_frame+1)[:, elements_per_frame:]
    # 计算帧数
    
    data = inputs.reshape(-1, 8, 280)
    print(f"Data shape: {data.shape} (frames, channels, window_size)")
    
    # 打印数据统计信息
    print("\nData statistics (per channel, last time step):")
    channel_names_all = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z', 'motorPos', 'motorVel']
    for i, name in enumerate(channel_names_all):
        ch_data = data[:, i, -1]
        print(f"  {name}: min={ch_data.min():.3f}, max={ch_data.max():.3f}, mean={ch_data.mean():.3f}")
    
    # Load and process config
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config_path)
    config = config_manager.apply_sensor_selection(config)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model_loader = ModelLoader()
    model, model_info = model_loader.load_model(
        args.model_path, device, config, load_weights=True
    )
    model.eval()
    print("Model loaded successfully!")
    
    # 初始化输出滤波器：2阶巴特沃斯低通，截止频率10Hz，采样率100Hz
    output_filter = RealtimeButterworthFilter(order=2, cutoff_freq=10.0, sample_rate=100.0)
    print("\nOutput filter initialized: 2nd order Butterworth lowpass, cutoff=10Hz, sample_rate=100Hz")

    # 选择的输入通道索引 (0,1,2,6) -> gyro_x, gyro_y, gyro_z, motorPos
    display_channels = [0, 1, 2, 6]
    channel_names = ['gyro_x', 'gyro_y', 'gyro_z', 'motorPos']
    
    # 历史数据缓存
    history_len = args.history_len
    raw_torque_history = deque(maxlen=history_len)      # 原始输出
    filtered_torque_history = deque(maxlen=history_len)  # 滤波后输出
    quan_torque_history = deque(maxlen=history_len)
    input_histories = {ch: deque(maxlen=history_len) for ch in display_channels}
    frame_indices = deque(maxlen=history_len)
    
    # 创建绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Offline IMU Data Analysis - Real-time Simulation')
    
    # 上图：输入通道（原始数据）
    ax_input = axes[0]
    ax_input.set_xlabel('Frame')
    ax_input.set_ylabel('Raw Sensor Value')
    ax_input.set_title('Input Channels (Raw Data): gyro_x, gyro_y, gyro_z, motorPos')
    ax_input.grid(True, alpha=0.3)
    input_lines = {}
    colors = ['r', 'g', 'b', 'm']
    for i, (ch, name) in enumerate(zip(display_channels, channel_names)):
        line, = ax_input.plot([], [], colors[i], label=name, linewidth=1)
        input_lines[ch] = line
    ax_input.legend(loc='upper right')
    
    # 下图：输出扭矩（原始和滤波后）
    ax_torque = axes[1]
    ax_torque.set_xlabel('Frame')
    ax_torque.set_ylabel('Torque (Nm/kg)')
    ax_torque.set_title('Predicted Knee Moment (Raw vs Filtered vs Quan)')
    ax_torque.grid(True, alpha=0.3)
    raw_torque_line, = ax_torque.plot([], [], 'b-', linewidth=1, alpha=0.5, label='Raw')
    filtered_torque_line, = ax_torque.plot([], [], 'r-', linewidth=1.5, label='Filtered (10Hz LP)')
    quan_torque_line, = ax_torque.plot([], [], 'g-', linewidth=1.5, label='quantize model')
    ax_torque.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 当前帧索引
    current_frame = [0]
    total_frames = len(data)
    
    def init():
        """初始化动画"""
        for line in input_lines.values():
            line.set_data([], [])
        raw_torque_line.set_data([], [])
        filtered_torque_line.set_data([], [])
        quan_torque_line.set_data([], [])
        return list(input_lines.values()) + [raw_torque_line, filtered_torque_line, quan_torque_line]
    
    def update(frame):
        """更新每一帧"""
        idx = current_frame[0]
        if idx >= total_frames:
            return list(input_lines.values()) + [raw_torque_line, filtered_torque_line]
        
        # 获取当前帧数据 [8, 280]（原始传感器数据）
        frame_data = data[idx]
        
        # 记录输入通道的最后一个时间点的值（原始数据）
        for ch in display_channels:
            input_histories[ch].append(frame_data[ch, -1])
        
        # 直接将原始数据送入模型（模型内部会自动归一化）
        input_tensor = torch.from_numpy(frame_data).float().unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            output = model(input_tensor)
            # 输出形状: (N, 1, 1, T) -> 获取最后一个时间点
            if len(output.shape) == 4:
                raw_torque = output[0, 0, 0, -1].item()
            elif len(output.shape) == 3:
                raw_torque = output[0, 0, -1].item()
            else:
                raw_torque = output[0, -1].item()
        
        # 对输出进行巴特沃斯低通滤波
        filtered_torque = output_filter.filter(raw_torque)
        
        quan_torque = quan_output[idx].item()
        
        raw_torque_history.append(raw_torque)
        filtered_torque_history.append(filtered_torque)
        quan_torque_history.append(quan_torque)
        frame_indices.append(idx)
        
        # 更新绘图数据
        x_data = list(frame_indices)
        
        for ch in display_channels:
            input_lines[ch].set_data(x_data, list(input_histories[ch]))
        
        raw_torque_line.set_data(x_data, list(raw_torque_history))
        filtered_torque_line.set_data(x_data, list(filtered_torque_history))
        quan_torque_line.set_data(x_data, list(quan_torque_history))
        
        # 调整坐标轴范围
        if len(x_data) > 1:
            ax_input.set_xlim(x_data[0], x_data[-1] + 1)
            ax_torque.set_xlim(x_data[0], x_data[-1] + 1)
            
            # 自动调整Y轴
            all_input_vals = []
            for ch in display_channels:
                all_input_vals.extend(list(input_histories[ch]))
            if all_input_vals:
                y_min, y_max = min(all_input_vals), max(all_input_vals)
                margin = (y_max - y_min) * 0.1 + 0.1
                ax_input.set_ylim(y_min - margin, y_max + margin)
            
            if raw_torque_history and filtered_torque_history:
                all_torque = list(raw_torque_history) + list(filtered_torque_history)
                t_min, t_max = min(all_torque), max(all_torque)
                margin = (t_max - t_min) * 0.1 + 0.1
                ax_torque.set_ylim(t_min - margin, t_max + margin)
        
        # 更新标题显示当前进度
        fig.suptitle(f'Offline Analysis - Frame {idx+1}/{total_frames} | Raw: {raw_torque:.4f} | '
                     f'Filtered: {filtered_torque:.4f} | Quantize: {quan_torque:.4f}Nm/kg')
        
        current_frame[0] += 1
        
        return list(input_lines.values()) + [raw_torque_line, filtered_torque_line, quan_torque_line]
    
    print(f"\nStarting real-time simulation ({total_frames} frames, {args.interval}ms interval)...")
    print("Close the plot window to exit.")
    
    ani = FuncAnimation(
        fig, update, init_func=init,
        frames=total_frames, interval=args.interval,
        blit=False, repeat=False
    )
    
    plt.show()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

