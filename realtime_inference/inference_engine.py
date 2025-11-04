import sys
import os
from ImpactAttenuator import ImpactAttenuator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_utils import ConfigManager
from utils.model_loader import ModelLoader
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import time
from scipy import signal


class ButterworthFilter:
    """巴特沃斯低通滤波器类"""

    def __init__(self, cutoff_freq: float, sampling_rate: float, order: int = 2):
        """
        初始化巴特沃斯低通滤波器
        Args:
            cutoff_freq: 截止频率 (Hz)
            sampling_rate: 采样率 (Hz)
            order: 滤波器阶数，默认为2
        """
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order

        # 计算归一化截止频率
        nyquist = sampling_rate / 2
        self.normalized_cutoff = cutoff_freq / nyquist

        # 设计巴特沃斯滤波器
        self.b, self.a = signal.butter(self.order, self.normalized_cutoff, btype='low', analog=False)

        # 初始化滤波器状态
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.filter_state = None

    def reset(self):
        """重置滤波器状态"""
        self.filter_state = None

    def filter(self, value: float) -> float:
        """
        对单个值进行滤波
        Args:
            value: 原始值
        Returns:
            滤波后的值
        """
        # 如果滤波器状态未初始化，使用当前值初始化
        if self.filter_state is None:
            self.filter_state = self.zi * value

        # 应用滤波
        filtered_value, self.filter_state = signal.lfilter(
            self.b, self.a, [value], zi=self.filter_state
        )

        return filtered_value[0]

    def get_params(self) -> Dict[str, float]:
        """获取滤波器参数"""
        return {
            'cutoff_freq': self.cutoff_freq,
            'sampling_rate': self.sampling_rate,
            'normalized_cutoff': self.normalized_cutoff,
            'order': self.order
        }


class TorqueNonlinearFilter:
    """力矩系数非线性滤波器"""

    def __init__(self,
                 power_pos=1.5,  # 正向幂次
                 power_neg=2.5,  # 负向幂次
                 gain_pos=1.8,  # 正向增益（>1为放大）
                 gain_neg=0.8,  # 负向增益（<1为抑制）
                 input_limit=0.4,  # 输入限制（修改为0.4）
                 output_limit=0.6):  # 输出限制（修改为0.6，1.5倍）

        self.power_pos = power_pos
        self.power_neg = power_neg
        self.gain_pos = gain_pos
        self.gain_neg = gain_neg
        self.input_limit = input_limit
        self.output_limit = output_limit

    def filter(self, torque_coefficient):
        """
        对力矩系数进行非线性滤波
        输入范围：-0.4 到 +0.4
        输出范围：-0.6 到 +0.6
        """
        # 输入限幅
        torque_coefficient = np.clip(torque_coefficient,
                                     -self.input_limit,
                                     self.input_limit)

        # 归一化到[-1, 1]
        normalized = torque_coefficient / self.input_limit

        # 应用非线性变换
        if torque_coefficient >= 0:
            # 正向：放大
            filtered_norm = np.sign(normalized) * \
                            (abs(normalized) ** self.power_pos) * \
                            self.gain_pos
        else:
            # 负向：抑制
            filtered_norm = -(abs(normalized) ** self.power_neg) * \
                            self.gain_neg

        # 反归一化并限幅
        output = filtered_norm * self.input_limit
        return np.clip(output, -self.output_limit, self.output_limit)


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
        self.model, self.model_info = self.model_loader.load_model(
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
        self.data_buffer_l = deque(maxlen=self.buffer_size)
        self.data_buffer_r = deque(maxlen=self.buffer_size)

        self.enable_vel_filter = False
        if hasattr(self.config, 'vel_filter_cutoff'):
            self.enable_vel_filter = True

        # 获取输入输出配置
        self.input_names = self.config.input_names
        self.input_rate = self.config.input_rate
        self.sides = self.config.side
        self.label_names = self.config.label_names

        self.label_dir_names = []

        for i, label_name in enumerate(self.config.label_names):
            for j, side in enumerate(self.sides):
                self.label_dir_names.append(label_name.replace("*", side))

        self.model_delays = self.config.model_delays if hasattr(self.config, 'model_delays') else [0] * len(
            self.label_names)

        self.input_frames_needed = int(self.history_window * self.input_rate / 200)

        # 初始化冲击衰减器

        # 初始化力矩非线性滤波器
        self.torque_nonlinear_filter = None
        if hasattr(self.config, 'enable_torque_nonlinear_filter') and self.config.enable_torque_nonlinear_filter:
            # 从配置文件读取参数，或使用默认值
            filter_params = {
                'power_pos': getattr(self.config, 'torque_filter_power_pos', 1.5),
                'power_neg': getattr(self.config, 'torque_filter_power_neg', 2.5),
                'gain_pos': getattr(self.config, 'torque_filter_gain_pos', 1.8),
                'gain_neg': getattr(self.config, 'torque_filter_gain_neg', 0.8),
                'input_limit': getattr(self.config, 'torque_filter_input_limit', 0.4),  # 默认0.4
                'output_limit': getattr(self.config, 'torque_filter_output_limit', 0.6)  # 默认0.6
            }

            self.torque_nonlinear_filter = TorqueNonlinearFilter(**filter_params)
            print(f"力矩非线性滤波器已启用:")
            print(f"  - 正向幂次: {filter_params['power_pos']}")
            print(f"  - 负向幂次: {filter_params['power_neg']}")
            print(f"  - 正向增益: {filter_params['gain_pos']}")
            print(f"  - 负向增益: {filter_params['gain_neg']}")
            print(f"  - 输入限制: ±{filter_params['input_limit']}")
            print(f"  - 输出限制: ±{filter_params['output_limit']}")

        # 性能监控
        self.inference_times = deque(maxlen=100)
        self.last_inference_time = 0

        print(f"Inference engine initialized:")
        print(f"  - Input dimensions: {len(self.input_names)}")
        print(f"  - Output dimensions: {len(self.label_names)}")
        print(f"  - History window: {self.history_window}")
        print(f"  - Model delays: {self.model_delays}")

        # 初始化巴特沃斯滤波器（为每条腿创建独立的滤波器）
        self.butterworth_filters = {}
        if hasattr(self.config, 'vel_filter_cutoff'):
            # 获取滤波器参数
            cutoff_freq = self.config.vel_filter_cutoff
            sampling_rate = self.config.vel_filter_sampling_rate
            filter_order = getattr(self.config, 'vel_filter_order', 2)  # 默认2阶

            # 为每个输出标签和每侧腿创建独立的滤波器
            for label_name in self.label_names:
                for side in self.sides:
                    filter_key = label_name.replace("*", side)
                    self.butterworth_filters[filter_key] = ButterworthFilter(
                        cutoff_freq, sampling_rate, filter_order
                    )

            print(f"已启用巴特沃斯滤波器:")
            print(f"  - 截止频率: {cutoff_freq} Hz")
            print(f"  - 采样率: {sampling_rate} Hz")
            print(f"  - 滤波器阶数: {filter_order}")
            print(f"  - 创建了 {len(self.butterworth_filters)} 个独立滤波器")

    def reset_filters(self):
        """重置所有滤波器状态"""
        for filter_instance in self.butterworth_filters.values():
            filter_instance.reset()
        print(f"已重置 {len(self.butterworth_filters)} 个滤波器")

    def process_frame_hip(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """
        处理单帧传感器数据
        Args:
            sensor_data: 传感器数据字典 {sensor_name: value}
        Returns:
            关节力矩字典 {joint_name: moment_value}
        """
        start_time = time.time()

        # 推理时的修改
        # 分别提取左右侧数据
        frame_data_l = np.array([sensor_data['hip_angle_l'], sensor_data['hip_vel_l']])
        frame_data_r = np.array([sensor_data['hip_angle_r'], sensor_data['hip_vel_r']])

        # 分别添加到各自的buffer
        self.data_buffer_l.append(frame_data_l)
        self.data_buffer_r.append(frame_data_r)

        # 检查是否有足够的历史数据
        if len(self.data_buffer_l) < self.input_frames_needed or len(self.data_buffer_r) < self.input_frames_needed:
            print("skip infer")
            return {}

        # 准备左右两侧的输入窗口
        input_window_l = np.array(list(self.data_buffer_l))[-self.input_frames_needed:]
        input_window_r = np.array(list(self.data_buffer_r))[-self.input_frames_needed:]

        # 重采样处理
        if self.input_rate != 200:
            input_window_l = signal.resample(input_window_l, self.history_window, axis=0)
            input_window_r = signal.resample(input_window_r, self.history_window, axis=0)
        else:
            input_window_l = np.array(list(self.data_buffer_l))[-self.history_window:]
            input_window_r = np.array(list(self.data_buffer_r))[-self.history_window:]

        # 构建输入张量 - shape: [2, 2, 248]
        # 方式1: 使用stack
        input_tensor_l = torch.tensor(input_window_l.T, dtype=torch.float32).unsqueeze(0)  # [1, 2, 248]
        input_tensor_r = torch.tensor(input_window_r.T, dtype=torch.float32).unsqueeze(0)  # [1, 2, 248]
        input_tensor = torch.cat([input_tensor_l, input_tensor_r], dim=0).to(self.device)  # [2, 2, 248]

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
        #先左后右
        # 提取输出 - 模型输出的最后一个时刻是对"当前-delay"时刻的预测
        moments = {}
        for i, label_name in enumerate(self.label_names):
            for j, side in enumerate(self.sides):
                # 获取最新的预测值（这是对past时刻的预测）
                moment_value = output[j, i, -1].item()
                # 构建当前输出的键
                output_key = label_name.replace("*", side)
                # 应用巴特沃斯滤波器（如果启用，使用对应腿的滤波器）
                if self.enable_vel_filter and output_key in self.butterworth_filters:
                    moment_value = self.butterworth_filters[output_key].filter(moment_value)

                # 应用力矩非线性滤波器（如果启用）
                if self.torque_nonlinear_filter is not None:
                    moment_value = self.torque_nonlinear_filter.filter(moment_value)

                moments[output_key] = moment_value

        # 记录推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        self.inference_times.append(inference_time)
        self.last_inference_time = inference_time

        return moments

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
            print("skip infer")
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
            for j, side in enumerate(self.sides):
                # 获取最新的预测值（这是对past时刻的预测）
                moment_value = output[j, i, -1].item()
                # 构建当前输出的键
                output_key = label_name.replace("*", side)
                # 应用巴特沃斯滤波器（如果启用，使用对应腿的滤波器）
                if self.enable_vel_filter and output_key in self.butterworth_filters:
                    moment_value = self.butterworth_filters[output_key].filter(moment_value)
                # 应用力矩非线性滤波器（如果启用）
                if self.torque_nonlinear_filter is not None:
                    moment_value = self.torque_nonlinear_filter.filter(moment_value)

                moments[output_key] = moment_value

        # 记录推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        self.inference_times.append(inference_time)
        self.last_inference_time = inference_time

        return moments

    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if len(self.inference_times) > 0:
            stats = {
                'avg_inference_time': np.mean(self.inference_times),
                'max_inference_time': np.max(self.inference_times),
                'min_inference_time': np.min(self.inference_times),
                'last_inference_time': self.last_inference_time,
                'buffer_size': len(self.data_buffer)
            }

            # 添加滤波器信息
            if self.butterworth_filters:
                stats['num_butterworth_filters'] = len(self.butterworth_filters)

            return stats
        else:
            return {
                'avg_inference_time': 0,
                'max_inference_time': 0,
                'min_inference_time': 0,
                'last_inference_time': 0,
                'buffer_size': 0,
                'num_butterworth_filters': len(self.butterworth_filters)
            }

    def reset_buffer(self):
        """重置数据缓冲区"""
        self.data_buffer.clear()
        self.data_buffer_l.clear()
        self.data_buffer_r.clear()
        self.inference_times.clear()

        # 重置所有滤波器
        if self.butterworth_filters:
            self.reset_filters()

        print("Buffer and filters reset")