"""
motorVel滤波效果对比分析脚本
对比DataPreprocessor.filter_velocity滤波前后的速度值
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from typing import List, Tuple
import os

# 简化版的DataPreprocessor类，只包含滤波功能
class SimpleVelocityFilter:
    def __init__(self, cutoff_freq: float = 10.0, sampling_rate: float = 200.0):
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

        print(f"巴特沃斯滤波器参数:")
        print(f"  - 截止频率: {cutoff_freq} Hz")
        print(f"  - 采样率: {sampling_rate} Hz")
        print(f"  - 归一化截止频率: {normalized_cutoff:.4f}")
        print(f"  - 滤波器阶数: {self.filter_order}")

    def reset_filter(self):
        """重置滤波器状态"""
        self.filter_state = None

    def filter_velocity(self, velocity: float) -> float:
        """
        对速度值进行滤波
        Args:
            velocity: 原始速度值
        Returns:
            滤波后的速度值
        """
        # 如果滤波器状态未初始化，使用当前值初始化
        if self.filter_state is None:
            self.filter_state = self.zi * velocity

        # 应用滤波
        filtered_value, self.filter_state = signal.lfilter(
            self.b, self.a, [velocity], zi=self.filter_state
        )

        return filtered_value[0]


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    加载CSV文件
    Args:
        csv_path: CSV文件路径
    Returns:
        pandas DataFrame
    """
    # 列名定义
    column_names = [
        'motorPos', 'motorVel',
        'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'label'
    ]

    try:
        # 尝试读取CSV文件，自动检测是否有表头
        first_line = pd.read_csv(csv_path, nrows=1, header=None)

        # 检查第一行是否包含字符串（表头）
        if first_line.iloc[0].dtype == object and 'motorPos' in str(first_line.iloc[0, 0]):
            # 有表头
            data = pd.read_csv(csv_path)
            data.columns = column_names
            print(f"加载 {len(data)} 行数据 (含表头)")
        else:
            # 没有表头
            data = pd.read_csv(csv_path, header=None, names=column_names)
            print(f"加载 {len(data)} 行数据 (无表头)")

        # 转换为数值类型
        for col in column_names[:-1]:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data['label'] = pd.to_numeric(data['label'], errors='coerce').astype('Int64')

        # 删除NaN行
        data = data.dropna()

        return data

    except Exception as e:
        print(f"加载CSV文件出错: {e}")
        return pd.DataFrame()


def compare_filtering(data: pd.DataFrame, cutoff_freq: float = 10.0, sampling_rate: float = 200.0) -> Tuple[List[float], List[float]]:
    """
    对比滤波前后的motorVel值
    Args:
        data: 数据DataFrame
        cutoff_freq: 滤波器截止频率
        sampling_rate: 采样率
    Returns:
        原始值列表，滤波后值列表
    """
    # 创建滤波器
    vel_filter = SimpleVelocityFilter(cutoff_freq, sampling_rate)

    original_values = []
    filtered_values = []

    # 处理每一帧数据
    for idx, row in data.iterrows():
        original_vel = row['motorVel']
        filtered_vel = vel_filter.filter_velocity(original_vel)

        # 注意：根据DataPreprocessor的代码，还需要除以2
        filtered_vel_final = filtered_vel / 2
        original_vel_final = original_vel / 2

        original_values.append(original_vel_final)
        filtered_values.append(filtered_vel_final)

    return original_values, filtered_values


def plot_comparison(original: List[float], filtered: List[float], sampling_rate: float = 200.0):
    """
    绘制对比图
    Args:
        original: 原始值列表
        filtered: 滤波后值列表
        sampling_rate: 采样率
    """
    # 创建时间轴
    time_axis = np.arange(len(original)) / sampling_rate

    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 子图1：原始信号和滤波后信号对比
    axes[0].plot(time_axis, original, 'b-', alpha=0.7, linewidth=0.8, label='原始motorVel')
    axes[0].plot(time_axis, filtered, 'r-', linewidth=1.2, label='滤波后motorVel')
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('速度 (单位/s)')
    axes[0].set_title('motorVel滤波前后对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：差值
    difference = np.array(original) - np.array(filtered)
    axes[1].plot(time_axis, difference, 'g-', linewidth=0.8)
    axes[1].set_xlabel('时间 (s)')
    axes[1].set_ylabel('差值')
    axes[1].set_title('滤波前后差值 (原始 - 滤波后)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # 子图3：频谱分析
    if len(original) > 256:  # 确保有足够的数据点进行FFT
        # 计算FFT
        n = len(original)
        freqs = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]

        # 原始信号的FFT
        fft_original = np.abs(np.fft.fft(original))[:n//2]
        # 滤波后信号的FFT
        fft_filtered = np.abs(np.fft.fft(filtered))[:n//2]

        axes[2].semilogy(freqs, fft_original, 'b-', alpha=0.7, linewidth=0.8, label='原始信号频谱')
        axes[2].semilogy(freqs, fft_filtered, 'r-', linewidth=1.2, label='滤波后频谱')
        axes[2].axvline(x=10, color='k', linestyle='--', alpha=0.5, label='截止频率 (10Hz)')
        axes[2].set_xlabel('频率 (Hz)')
        axes[2].set_ylabel('幅值')
        axes[2].set_title('频谱分析')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([0, 50])  # 只显示0-50Hz的频率范围

    plt.tight_layout()
    plt.show()


def print_statistics(original: List[float], filtered: List[float]):
    """
    打印统计信息
    Args:
        original: 原始值列表
        filtered: 滤波后值列表
    """
    orig_array = np.array(original)
    filt_array = np.array(filtered)

    print("\n" + "="*50)
    print("统计信息对比")
    print("="*50)

    print(f"\n样本数量: {len(original)}")

    print(f"\n原始motorVel统计:")
    print(f"  - 平均值: {np.mean(orig_array):.4f}")
    print(f"  - 标准差: {np.std(orig_array):.4f}")
    print(f"  - 最小值: {np.min(orig_array):.4f}")
    print(f"  - 最大值: {np.max(orig_array):.4f}")
    print(f"  - 范围: {np.max(orig_array) - np.min(orig_array):.4f}")

    print(f"\n滤波后motorVel统计:")
    print(f"  - 平均值: {np.mean(filt_array):.4f}")
    print(f"  - 标准差: {np.std(filt_array):.4f}")
    print(f"  - 最小值: {np.min(filt_array):.4f}")
    print(f"  - 最大值: {np.max(filt_array):.4f}")
    print(f"  - 范围: {np.max(filt_array) - np.min(filt_array):.4f}")

    # 计算滤波效果
    print(f"\n滤波效果:")
    print(f"  - 标准差减少: {(1 - np.std(filt_array)/np.std(orig_array))*100:.2f}%")
    print(f"  - 范围减少: {(1 - (np.max(filt_array)-np.min(filt_array))/(np.max(orig_array)-np.min(orig_array)))*100:.2f}%")

    # 计算延迟（使用互相关）
    if len(original) > 100:
        correlation = np.correlate(orig_array, filt_array, mode='same')
        lag = len(correlation)//2 - np.argmax(correlation)
        print(f"  - 估计延迟: {lag} 采样点 ({lag/200*1000:.2f} ms @ 200Hz)")


def main():
    """主函数"""
    # CSV文件路径
    csv_path = 'devices/kneeData_left.csv'

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        print("请确保文件路径正确")
        return

    print(f"正在加载数据文件: {csv_path}")

    # 加载数据
    data = load_csv_data(csv_path)

    if data.empty:
        print("数据加载失败")
        return

    print(f"成功加载 {len(data)} 行数据")

    # 滤波参数
    cutoff_freq = 10.0  # Hz
    sampling_rate = 200.0  # Hz

    print(f"\n开始滤波处理...")
    print(f"滤波器参数: 截止频率={cutoff_freq}Hz, 采样率={sampling_rate}Hz")

    # 对比滤波前后
    original_values, filtered_values = compare_filtering(data, cutoff_freq, sampling_rate)

    # 打印统计信息
    print_statistics(original_values, filtered_values)

    # 绘制对比图
    print("\n正在生成对比图表...")
    plot_comparison(original_values, filtered_values, sampling_rate)

    # 保存结果到CSV（可选）
    save_results = input("\n是否保存对比结果到CSV文件? (y/n): ")
    if save_results.lower() == 'y':
        output_df = pd.DataFrame({
            'time_s': np.arange(len(original_values)) / sampling_rate,
            'original_motorVel': original_values,
            'filtered_motorVel': filtered_values,
            'difference': np.array(original_values) - np.array(filtered_values)
        })
        output_path = 'motorVel_comparison_results.csv'
        output_df.to_csv(output_path, index=False)
        print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()