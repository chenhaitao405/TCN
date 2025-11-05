"""
巴特沃斯低通滤波器模块
用于信号处理的统一滤波器实现
"""
from scipy import signal
from typing import Dict, Optional


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

        # 打印滤波器参数（可选）
        self.print_params()

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

    def print_params(self, verbose: bool = False):
        """
        打印滤波器参数
        Args:
            verbose: 是否打印详细信息
        """
        if verbose:
            print(f"巴特沃斯滤波器参数:")
            print(f"  - 截止频率: {self.cutoff_freq} Hz")
            print(f"  - 采样率: {self.sampling_rate} Hz")
            print(f"  - 归一化截止频率: {self.normalized_cutoff:.4f}")
            print(f"  - 滤波器阶数: {self.order}")