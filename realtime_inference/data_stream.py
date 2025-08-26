import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataManager
import torch
import numpy as np
from typing import Dict, Optional, Generator, List, Tuple
import time
from abc import ABC, abstractmethod


class DataSource(ABC):
    """数据源基类"""

    @abstractmethod
    def get_next_frame(self) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """获取下一帧数据
        Returns:
            (sensor_data, ground_truth) - 如果没有ground truth则返回None
        """
        pass

    @abstractmethod
    def reset(self):
        """重置数据源"""
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """检查数据是否已经结束"""
        pass


class DatasetSource(DataSource):
    """从数据集读取数据的数据源"""

    def __init__(self, config):
        self.config = config
        self.data_manager = DataManager()
        self.device = torch.device('cpu')

        # 加载数据集
        print("Loading dataset...")
        self.full_dataset = self.data_manager.load_datasets(self.config, self.device)
        self.valid_indices = self.data_manager.get_or_compute_valid_indices(
            self.full_dataset, self.config
        )
        print(f"Loaded {len(self.valid_indices)} valid trials")

        # 初始化状态
        self.current_trial_idx = None
        self.current_inputs = None
        self.current_labels = None
        self.current_seq_length = 0
        self.current_frame_idx = 0
        self.current_trial_name = ""

    def select_trial(self, trial_idx: int) -> bool:
        """选择特定试验"""
        if trial_idx < len(self.valid_indices):
            actual_idx = self.valid_indices[trial_idx]
            # 获取数据
            inputs, labels, seq_lengths, trial_names = self.full_dataset[actual_idx]

            self.current_trial_idx = trial_idx
            self.current_inputs = inputs.squeeze(0).numpy()  # [features, time]
            self.current_labels = labels.squeeze(0).numpy()  # [labels, time]

            # 处理序列长度
            if isinstance(seq_lengths, list):
                self.current_seq_length = seq_lengths[0]
            else:
                self.current_seq_length = seq_lengths

            # 处理试验名称
            if isinstance(trial_names, list):
                self.current_trial_name = trial_names[0]
            else:
                self.current_trial_name = trial_names

            self.current_frame_idx = 0

            print(f"Selected trial: {self.current_trial_name} (length: {self.current_seq_length})")
            return True
        return False

    def get_trial_list(self) -> List[str]:
        """获取所有可用试验名称"""
        trial_names = []
        for idx in self.valid_indices:
            _, _, _, names = self.full_dataset[idx]
            name = names[0] if isinstance(names, list) else names
            trial_names.append(name)
        return trial_names

    def get_next_frame(self) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """获取下一帧数据"""
        if self.current_inputs is None or self.current_frame_idx >= self.current_seq_length:
            return None, None

        # 准备传感器数据字典
        sensor_data = {}
        for i, name in enumerate(self.config.input_names):
            sensor_data[name] = float(self.current_inputs[i, self.current_frame_idx])

        # 准备真实标签
        ground_truth = {}
        for i, name in enumerate(self.config.label_names):
            ground_truth[name] = float(self.current_labels[i, self.current_frame_idx])

        self.current_frame_idx += 1
        return sensor_data, ground_truth

    def reset(self):
        """重置到当前试验的开始"""
        self.current_frame_idx = 0

    def is_finished(self) -> bool:
        """检查当前试验是否结束"""
        return self.current_inputs is None or self.current_frame_idx >= self.current_seq_length

    def get_progress(self) -> float:
        """获取当前进度百分比"""
        if self.current_seq_length > 0:
            return (self.current_frame_idx / self.current_seq_length) * 100
        return 0


class DeviceSource(DataSource):
    """从实时设备读取数据的数据源（预留接口）"""

    def __init__(self, config):
        self.config = config
        self.is_connected = False

    def connect(self, device_address: str) -> bool:
        """连接到设备"""
        # TODO: 实现设备连接逻辑
        print(f"Device connection not implemented yet. Address: {device_address}")
        return False

    def get_next_frame(self) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """从设备获取实时数据"""
        if not self.is_connected:
            return None, None

        # TODO: 实现从设备读取数据的逻辑
        # sensor_data = self.read_from_device()
        # return sensor_data, None  # 实时设备没有ground truth

        return None, None

    def reset(self):
        """重置设备连接"""
        pass

    def is_finished(self) -> bool:
        """设备数据流永不结束"""
        return False


class DataStreamManager:
    """数据流管理器"""

    def __init__(self, config):
        self.config = config
        self.dataset_source = DatasetSource(config)
        self.device_source = DeviceSource(config)

        # 添加自定义设备源
        from devices.custom_device_source import CustomDeviceSource
        self.custom_device_source = CustomDeviceSource(config)

        self.current_source = self.dataset_source
        self.use_device = False
        self.use_custom_device = False  # 新增标志

        # 播放控制
        self.playback_speed = 1.0
        self.is_paused = False
        self.is_playing = False
        self.sampling_rate = 200  # Hz

    def set_source(self, use_device: bool, use_custom: bool = False):
        """切换数据源（修改后的版本）"""
        self.use_device = use_device
        self.use_custom_device = use_custom

        if use_device and use_custom:
            self.current_source = self.custom_device_source
            print("Switched to custom device source")
        elif use_device:
            self.current_source = self.device_source
            print("Switched to device source")
        else:
            self.current_source = self.dataset_source
            print("Switched to dataset source")

    def set_playback_speed(self, speed: float):
        """设置播放速度"""
        self.playback_speed = speed
        print(f"Playback speed set to {speed}x")

    def pause(self):
        """暂停播放"""
        self.is_paused = True

    def resume(self):
        """恢复播放"""
        self.is_paused = False

    def stop(self):
        """停止播放"""
        self.is_playing = False
        self.current_source.reset()

    def stream_frames(self) -> Generator[Tuple[Dict, Optional[Dict], float], None, None]:
        """流式输出数据帧
        Yields:
            (sensor_data, ground_truth, timestamp)
        """
        self.is_playing = True
        frame_interval = 1.0 / (self.sampling_rate * self.playback_speed)
        start_time = time.time()
        frame_count = 0

        while self.is_playing and not self.current_source.is_finished():
            if self.is_paused:
                time.sleep(0.01)
                continue

            # 获取下一帧
            sensor_data, ground_truth = self.current_source.get_next_frame()

            if sensor_data is None:
                break

            # 计算时间戳
            timestamp = frame_count / self.sampling_rate

            yield sensor_data, ground_truth, timestamp

            frame_count += 1

            # 控制播放速度
            expected_time = start_time + frame_count * frame_interval
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)

    def configure_custom_device(self, config: dict) -> bool:
        """
        配置自定义设备源
        Args:
            config: 配置字典
        Returns:
            是否配置成功
        """
        if config['source_type'] == 'offline':
            # 配置离线CSV模式
            success = self.custom_device_source.load_csv_file(
                config['csv_path'],
                config.get('label_filter', None),
                config.get('side', 'r')
            )

            if success:
                # 设置预处理参数
                self.custom_device_source.set_preprocessing_params(
                    config.get('preprocessing', {})
                )

                # 启动数据流
                self.custom_device_source.start_streaming()

            return success

        elif config['source_type'] == 'ros':
            # 配置ROS模式
            return self.custom_device_source.connect_ros(config.get('topic', '/exo_sensor_data'))

        return False

    def get_custom_device_status(self) -> dict:
        """获取自定义设备状态"""
        if hasattr(self, 'custom_device_source'):
            return self.custom_device_source.get_status()
        return {}

    def stream_frames_with_debug(self):
        """
        流式输出数据帧（带调试信息）
        专门用于自定义设备源
        """
        self.is_playing = True
        frame_interval = 1.0 / (self.sampling_rate * self.playback_speed)
        start_time = time.time()
        frame_count = 0

        while self.is_playing:
            if self.is_paused:
                time.sleep(0.01)
                continue

            # 根据源类型获取数据
            if self.use_custom_device and hasattr(self.custom_device_source, 'get_next_frame_with_debug'):
                # 获取带调试信息的数据
                sensor_data, raw_data, debug_info = self.custom_device_source.get_next_frame_with_debug()

                if sensor_data is None:
                    # 检查是否结束
                    if self.custom_device_source.is_finished():
                        break
                    else:
                        # 等待数据
                        time.sleep(0.001)
                        continue

                # 计算时间戳
                timestamp = frame_count / self.sampling_rate

                yield sensor_data, None, timestamp, raw_data, debug_info

            else:
                # 使用原始的stream_frames逻辑
                sensor_data, ground_truth = self.current_source.get_next_frame()

                if sensor_data is None:
                    break

                timestamp = frame_count / self.sampling_rate
                yield sensor_data, ground_truth, timestamp, None, None

            frame_count += 1

            # 控制播放速度
            expected_time = start_time + frame_count * frame_interval
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)