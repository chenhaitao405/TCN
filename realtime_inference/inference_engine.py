import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_utils import ConfigManager
from utils.model_loader import ModelLoader
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import time
from scipy import signal

class InferenceEngine:
    """实时推理引擎，复用现有的模型加载和配置管理代码"""

    def __init__(self, config_path: str):
        """
        初始化推理引擎
        Args:
            config_path: 配置文件路径
        """
        # 复用现有的配置加载器
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)
        self.config = self.config_manager.apply_sensor_selection(self.config)

        # 复用现有的模型加载器
        self.model_loader = ModelLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载模型
        self.model, self.model_info = self.model_loader.load_pretrained_model(
            self.config.model_path,
            self.device,
            self.config,
            load_weights=True
        )
        self.model.eval()
        print(f"Model loaded successfully from {self.config.model_path}")

        # 初始化数据缓冲区
        self.history_window = self.model.get_effective_history()
        self.buffer_size = self.history_window + 500  # 额外缓冲
        self.data_buffer = deque(maxlen=self.buffer_size)

        # 获取输入输出配置
        self.input_names = self.config.input_names
        self.input_rate = self.config.input_rate
        self.label_names = self.config.label_names
        self.model_delays = self.config.model_delays if hasattr(self.config, 'model_delays') else [0] * len(
            self.label_names)

        self.input_frames_needed = int(self.history_window * self.input_rate / 200)


        # 性能监控
        self.inference_times = deque(maxlen=100)
        self.last_inference_time = 0

        print(f"Inference engine initialized:")
        print(f"  - Input dimensions: {len(self.input_names)}")
        print(f"  - Output dimensions: {len(self.label_names)}")
        print(f"  - History window: {self.history_window}")
        print(f"  - Model delays: {self.model_delays}")

        # 初始化巴特沃斯滤波器（用于motorVel）
        if hasattr(self.config, 'vel_filter_cutoff'):
            self.init_butterworth_filter(self.config.vel_filter_cutoff, self.config.vel_filter_sampling_rate)
            print(f"已启用motorVel滤波器 (截止频率: {self.config.vel_filter_cutoff} Hz)")

    def init_butterworth_filter(self, cutoff_freq: float, sampling_rate: float):
        """
        初始化巴特沃斯低通滤波器
        Args:
            cutoff_freq: 截止频率 (Hz)
            sampling_rate: 采样率 (Hz)
        """
        # 计算归一化截止频率
        nyquist = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # 设计2阶巴特沃斯滤波器
        self.filter_order = 2
        self.b, self.a = signal.butter(self.filter_order, normalized_cutoff, btype='low', analog=False)

        # 初始化滤波器状态
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.filter_state = None

        # 打印滤波器参数（调试用）
        print(f"巴特沃斯滤波器参数:")
        print(f"  - 截止频率: {cutoff_freq} Hz")
        print(f"  - 采样率: {sampling_rate} Hz")
        print(f"  - 归一化截止频率: {normalized_cutoff:.4f}")
        print(f"  - 滤波器阶数: {self.filter_order}")

    def reset_filter(self):
        """重置滤波器状态"""
        if self.enable_vel_filter:
            self.filter_state = None

    def filter_velocity(self, velocity: float) -> float:
        """
        对速度值进行滤波
        Args:
            velocity: 原始速度值
        Returns:
            滤波后的速度值
        """
        if not self.enable_vel_filter:
            return velocity

        # 如果滤波器状态未初始化，使用当前值初始化
        if self.filter_state is None:
            self.filter_state = self.zi * velocity

        # 应用滤波
        filtered_value, self.filter_state = signal.lfilter(
            self.b, self.a, [velocity], zi=self.filter_state
        )

        return filtered_value[0]

    def process_frame(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """
        处理单帧传感器数据
        Args:
            sensor_data: 传感器数据字典 {sensor_name: value}
        Returns:
            关节力矩字典 {joint_name: moment_value}
        """
        start_time = time.time()

        # 将传感器数据按配置顺序排列
        frame_data = np.array([sensor_data.get(name, 0.0) for name in self.input_names])
        self.data_buffer.append(frame_data)

        # 检查是否有足够的历史数据
        if len(self.data_buffer) < self.input_frames_needed:
            return {}

        # 准备输入张量 - shape: [1, features, time]
        input_window = np.array(list(self.data_buffer))[-self.input_frames_needed:]
        if self.input_rate != 200:
            # 使用scipy的resample函数将数据重采样到200Hz（history_window帧）
            input_window = signal.resample(input_window, self.history_window, axis=0)
        else:
            # 如果已经是200Hz，直接使用原始数据
            input_window = np.array(list(self.data_buffer))[-self.history_window:]

        input_tensor = torch.tensor(input_window.T, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)

        # 提取输出 - 模型输出的最后一个时刻是对"当前-delay"时刻的预测
        moments = {}
        for i, label_name in enumerate(self.label_names):
            # 获取最新的预测值（这是对past时刻的预测）
            moment_value = output[0, i, -1].item()
            if hasattr(self.config, 'vel_filter_cutoff'):
                moment_value = self.filter_velocity(moment_value)
            moments[label_name] = moment_value

        # 记录推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        self.inference_times.append(inference_time)
        self.last_inference_time = inference_time

        return moments

    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if len(self.inference_times) > 0:
            return {
                'avg_inference_time': np.mean(self.inference_times),
                'max_inference_time': np.max(self.inference_times),
                'min_inference_time': np.min(self.inference_times),
                'last_inference_time': self.last_inference_time,
                'buffer_size': len(self.data_buffer)
            }
        else:
            return {
                'avg_inference_time': 0,
                'max_inference_time': 0,
                'min_inference_time': 0,
                'last_inference_time': 0,
                'buffer_size': 0
            }

    def reset_buffer(self):
        """重置数据缓冲区"""
        self.data_buffer.clear()
        self.inference_times.clear()
        print("Buffer reset")