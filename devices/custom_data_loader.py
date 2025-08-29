"""
自定义数据加载器，用于读取和处理自定义格式的离线数据
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from collections import deque
import threading
import time
from scipy import signal

class CustomDataLoader:
    """从CSV文件加载自定义格式的数据"""

    def __init__(self, csv_path: str, label_filter: Optional[int] = None, sampling_rate: float = 200.0):
        """
        初始化数据加载器
        Args:
            csv_path: CSV文件路径
            label_filter: 标签过滤器，只保留特定label的数据
            sampling_rate: 采样率（Hz）
        """
        self.csv_path = csv_path
        self.label_filter = label_filter
        self.sampling_rate = sampling_rate
        self.frame_interval = 1.0 / sampling_rate

        # 列名映射
        self.column_names = [
            'motorPos', 'motorVel',
            'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'label'
        ]

        # 加载数据
        self.data = None
        self.filtered_data = None
        self.current_index = 0
        self.load_data()

    def load_data(self):
        """加载CSV数据"""
        try:
            # 尝试读取CSV文件，自动检测是否有表头
            try:
                # 先尝试读取第一行
                first_line = pd.read_csv(self.csv_path, nrows=1, header=None)
                # 检查第一行是否包含字符串（表头）
                if first_line.iloc[0].dtype == object and 'motorPos' in str(first_line.iloc[0, 0]):
                    # 有表头，使用第一行作为列名
                    self.data = pd.read_csv(self.csv_path)
                    # 确保列名正确
                    self.data.columns = self.column_names
                    print(f"Loaded {len(self.data)} rows from {self.csv_path} (with header)")
                else:
                    # 没有表头
                    self.data = pd.read_csv(self.csv_path, header=None, names=self.column_names)
                    print(f"Loaded {len(self.data)} rows from {self.csv_path} (no header)")
            except:
                # 如果检测失败，尝试两种方式
                try:
                    # 假设有表头
                    self.data = pd.read_csv(self.csv_path)
                    if len(self.data.columns) == len(self.column_names):
                        self.data.columns = self.column_names
                        print(f"Loaded {len(self.data)} rows from {self.csv_path} (detected header)")
                except:
                    # 假设没有表头
                    self.data = pd.read_csv(self.csv_path, header=None, names=self.column_names)
                    print(f"Loaded {len(self.data)} rows from {self.csv_path} (no header assumed)")

            # 验证数据类型，确保都是数值
            for col in self.column_names[:-1]:  # 除了label列
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # label列转换为整数
            if 'label' in self.data.columns:
                self.data['label'] = pd.to_numeric(self.data['label'], errors='coerce').astype('Int64')

            # 删除包含NaN的行
            self.data = self.data.dropna()

            # 应用标签过滤
            if self.label_filter is not None:
                self.filtered_data = self.data[self.data['label'] == self.label_filter].reset_index(drop=True)
                print(f"Filtered to {len(self.filtered_data)} rows with label={self.label_filter}")
            else:
                self.filtered_data = self.data

            self.current_index = 0

        except Exception as e:
            print(f"Error loading CSV file: {e}")
            import traceback
            traceback.print_exc()
            self.filtered_data = pd.DataFrame()

    def get_next_frame(self) -> Optional[Dict[str, float]]:
        """
        获取下一帧数据
        Returns:
            数据字典或None（如果已结束）
        """
        if self.filtered_data is None or self.current_index >= len(self.filtered_data):
            return None

        row = self.filtered_data.iloc[self.current_index]
        self.current_index += 1

        # 转换为字典格式
        frame_data = {
            'motorPos': float(row['motorPos']),
            'motorVel': float(row['motorVel']),
            'acc_x': float(row['acc_x']),
            'acc_y': float(row['acc_y']),
            'acc_z': float(row['acc_z']),
            'gyro_x': float(row['gyro_x']),
            'gyro_y': float(row['gyro_y']),
            'gyro_z': float(row['gyro_z']),
            'label': int(row['label'])
        }

        return frame_data

    def reset(self):
        """重置到数据开始"""
        self.current_index = 0

    def is_finished(self) -> bool:
        """检查是否已经读完所有数据"""
        return self.filtered_data is None or self.current_index >= len(self.filtered_data)

    def get_progress(self) -> float:
        """获取当前进度百分比"""
        if self.filtered_data is not None and len(self.filtered_data) > 0:
            return (self.current_index / len(self.filtered_data)) * 100
        return 0

    def get_total_frames(self) -> int:
        """获取总帧数"""
        if self.filtered_data is not None:
            return len(self.filtered_data)
        return 0


class DataPreprocessor:
    """数据预处理器，将自定义数据格式转换为官方格式"""

    def __init__(self, config, side: str = 'l',
                 enable_vel_filter: bool = True,
                 vel_filter_cutoff: float = 10.0,
                 sampling_rate: float = 200.0):
        """
        初始化预处理器
        Args:
            config: 配置对象，包含input_names等信息
            side: 腿部侧面 ('r' 或 'l')
            enable_vel_filter: 是否启用速度滤波
            vel_filter_cutoff: 速度滤波器截止频率 (Hz)
            sampling_rate: 数据采样率 (Hz)
        """
        self.config = config
        self.side = side
        self.enable_vel_filter = enable_vel_filter
        self.sampling_rate = sampling_rate

        # 获取官方数据格式的输入名称
        self.official_input_names = config.input_names

        # 创建映射关系
        self.create_mapping()

        # 缺失传感器的默认值
        self.default_values = self.get_default_values()

        # 初始化巴特沃斯滤波器（用于motorVel）
        if self.enable_vel_filter:
            self.init_butterworth_filter(vel_filter_cutoff, sampling_rate)
            print(f"已启用motorVel滤波器 (截止频率: {vel_filter_cutoff} Hz)")

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

    def create_mapping(self):
        """创建自定义数据到官方格式的映射"""
        self.mapping = {}

        # 根据配置文件中的sensor_pick，我们知道使用的是：
        # [12,13,14,15,16,17,23,24] 对应：
        # thigh_imu_*_gyro_x, thigh_imu_*_gyro_y, thigh_imu_*_gyro_z,
        # thigh_imu_*_accel_x, thigh_imu_*_accel_y, thigh_imu_*_accel_z,
        # knee_angle_*, knee_angle_*_velocity_filt

        side = self.side
        self.direct_mapping = {
            'motorPos': f'knee_angle_*',
            'motorVel': f'knee_angle_*_velocity_filt',
            'gyro_x': f'thigh_imu_*_gyro_x',
            'gyro_y': f'thigh_imu_*_gyro_y',
            'gyro_z': f'thigh_imu_*_gyro_z',
            'acc_x': f'thigh_imu_*_accel_x',
            'acc_y': f'thigh_imu_*_accel_y',
            'acc_z': f'thigh_imu_*_accel_z',
        }

    def get_default_values(self) -> Dict[str, float]:
        """获取缺失传感器的默认值"""
        defaults = {}
        side = self.side

        # 为所有官方格式的传感器设置默认值
        for name in self.official_input_names:
            if 'gyro' in name:
                defaults[name] = 0.0
            elif 'accel' in name:
                if 'accel_z' in name:
                    defaults[name] = -9.81  # 重力加速度
                else:
                    defaults[name] = 0.0
            elif 'angle' in name and 'velocity' not in name:
                defaults[name] = 0.0
            elif 'velocity' in name:
                defaults[name] = 0.0
            elif 'cop' in name:
                defaults[name] = 0.0
            elif 'force' in name:
                defaults[name] = 0.0
            else:
                defaults[name] = 0.0

        return defaults

    def process(self, custom_data: Dict[str, float]) -> Dict[str, float]:
        """
        处理自定义格式数据，转换为官方格式
        Args:
            custom_data: 自定义格式的数据字典
        Returns:
            官方格式的数据字典
        """
        import numpy as np

        # 初始化输出，使用默认值
        processed_data = self.default_values.copy()

        # 始终应用坐标系转换（使用旋转矩阵）
        # 定义旋转矩阵：将你的坐标系转换到论文坐标系
        # 你的设备：X(下) Y(前) Z(内) -> 论文：X(前) Y(上) Z(内)
        R = np.array([[0, 1, 0],  # paper_x = device_y
                      [-1, 0, 0],  # paper_y = -device_x
                      [0, 0, 1]])  # paper_z = device_z

        # 提取IMU数据
        acc_vec = np.array([
            custom_data.get('acc_x', 0.0),
            custom_data.get('acc_y', 0.0),
            custom_data.get('acc_z', 0.0)
        ])
        gyro_vec = np.array([
            custom_data.get('gyro_x', 0.0),
            custom_data.get('gyro_y', 0.0),
            custom_data.get('gyro_z', 0.0)
        ])

        # 应用旋转矩阵
        acc_transformed = R @ acc_vec
        gyro_transformed = R @ gyro_vec

        # 创建转换后的数据字典
        transformed_data = custom_data.copy()
        transformed_data['acc_x'] = acc_transformed[0]
        transformed_data['acc_y'] = acc_transformed[1]
        transformed_data['acc_z'] = acc_transformed[2]
        transformed_data['gyro_x'] = gyro_transformed[0]
        transformed_data['gyro_y'] = gyro_transformed[1]
        transformed_data['gyro_z'] = gyro_transformed[2]

        # 映射数据到官方格式
        for custom_key, official_key in self.direct_mapping.items():
            if custom_key in transformed_data and official_key in processed_data:
                value = transformed_data[custom_key]

                # 电机角度减180度
                if custom_key == 'motorPos':
                    value -= 180

                if custom_key == 'motorVel':
                    value = self.filter_velocity(value)

                # 左腿镜像处理（如果需要）
                if self.side == 'l':
                    # 左腿需要反转某些轴（基于论文的坐标系）
                    if 'gyro_x' in official_key or 'gyro_y' in official_key:
                        value *= -1.0
                    elif 'accel_z' in official_key:
                        value *= -1.0

                processed_data[official_key] = value

        return processed_data