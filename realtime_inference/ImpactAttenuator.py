import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple


class ImpactAttenuator:
    """
    冲击检测与力矩衰减器
    用于检测落地冲击并计算相应的力矩衰减系数
    """

    def __init__(self,
                 frame_rate: int = 200,
                 acc_threshold: float = 30.0,
                 min_attenuation: float = -0.7,
                 attenuation_duration_ms: float = 300.0,
                 strong_attenuation_ms: float = 50.0,
                 mid_recovery_ms: float = 200.0,
                 mid_attenuation: float = 0,
                 min_impact_interval_ms: float = 200.0,
                 acc_rate_threshold: float = 30.0,
                 enable_rate_detection: bool = True,
                 debug: bool = False):
        """
        初始化冲击衰减器

        Args:
            frame_rate: 系统调用频率 (Hz)
            acc_threshold: ACC_X冲击检测阈值
            min_attenuation: 最小衰减系数 (0-1)
            attenuation_duration_ms: 总衰减持续时间 (ms)
            strong_attenuation_ms: 强衰减期持续时间 (ms)
            mid_recovery_ms: 中间恢复点时间 (ms)
            mid_attenuation: 中间恢复点衰减系数
            min_impact_interval_ms: 最小冲击间隔 (ms)，用于防抖
            acc_rate_threshold: ACC_X变化率阈值
            enable_rate_detection: 是否启用变化率检测
            debug: 是否输出调试信息
        """
        self.frame_rate = frame_rate
        self.debug = debug

        # 将时间参数转换为帧数
        self.config = {
            'acc_threshold': acc_threshold,
            'acc_rate_threshold': acc_rate_threshold,
            'enable_rate_detection': enable_rate_detection,
            'attenuation_frames': int(attenuation_duration_ms * frame_rate / 1000),
            'strong_attenuation_frames': int(strong_attenuation_ms * frame_rate / 1000),
            'mid_recovery_frames': int(mid_recovery_ms * frame_rate / 1000),
            'min_attenuation': min_attenuation,
            'mid_attenuation': mid_attenuation,
            'min_impact_interval': int(min_impact_interval_ms * frame_rate / 1000),
        }

        # 状态变量
        self.state = {
            'is_impact': False,
            'frames_since_impact': 0,
            'frames_since_last_impact': 999,
            'last_acc_x': 0.0,
            'acc_x_buffer': deque(maxlen=5),
        }

        # 统计信息
        self.stats = {
            'impact_count': 0,
            'total_frames': 0,
            'last_impact_frame': -1,
            'last_impact_acc': 0.0,
            'max_impact_acc': 0.0,
        }

        self._print_config()

    def _print_config(self):
        """打印配置信息"""
        if self.debug:
            print(f"[ImpactAttenuator] 配置:")
            print(f"  - 帧率: {self.frame_rate} Hz")
            print(f"  - ACC阈值: {self.config['acc_threshold']}")
            print(f"  - 衰减持续: {self.config['attenuation_frames']}帧 "
                  f"({self.config['attenuation_frames'] * 1000 / self.frame_rate:.1f}ms)")
            print(f"  - 强衰减期: {self.config['strong_attenuation_frames']}帧 "
                  f"({self.config['strong_attenuation_frames'] * 1000 / self.frame_rate:.1f}ms)")
            print(f"  - 最小衰减系数: {self.config['min_attenuation']}")
            print(f"  - 变化率检测: {'启用' if self.config['enable_rate_detection'] else '禁用'}")

    def detect_impact(self, acc_x: float) -> bool:
        """
        检测是否发生冲击

        Args:
            acc_x: 当前ACC_X值

        Returns:
            是否检测到新的冲击
        """
        # 更新ACC历史缓冲
        self.state['acc_x_buffer'].append(acc_x)

        # 1. 幅值检测
        amplitude_trigger = abs(acc_x) > self.config['acc_threshold']

        # 2. 变化率检测（可选）
        rate_trigger = False
        if self.config['enable_rate_detection'] and len(self.state['acc_x_buffer']) > 1:
            acc_rate = abs(acc_x - self.state['last_acc_x'])
            rate_trigger = acc_rate > self.config['acc_rate_threshold']

        # 3. 防抖检查
        debounce_ok = self.state['frames_since_last_impact'] > self.config['min_impact_interval']

        # 4. 综合判断
        if self.config['enable_rate_detection']:
            impact_detected = (amplitude_trigger or rate_trigger) and debounce_ok
        else:
            impact_detected = amplitude_trigger and debounce_ok

        # 5. 更新状态
        if impact_detected:
            self.state['is_impact'] = True
            self.state['frames_since_impact'] = 0
            self.state['frames_since_last_impact'] = 0

            # 更新统计
            self.stats['impact_count'] += 1
            self.stats['last_impact_frame'] = self.stats['total_frames']
            self.stats['last_impact_acc'] = acc_x
            self.stats['max_impact_acc'] = max(self.stats['max_impact_acc'], abs(acc_x))

            if self.debug:
                print(f"[Impact #{self.stats['impact_count']}] "
                      f"Frame={self.stats['total_frames']}, "
                      f"ACC_X={acc_x:.1f}, "
                      f"Rate={abs(acc_x - self.state['last_acc_x']):.1f}")

            self.state['last_acc_x'] = acc_x
            print("检测到新的冲击")
            return True

        self.state['last_acc_x'] = acc_x
        return False

    def calculate_attenuation(self) -> float:
        """
        计算当前衰减系数

        Returns:
            衰减系数 (0-1)
        """
        if not self.state['is_impact']:
            return 1.0

        frames = self.state['frames_since_impact']

        # 检查是否超过衰减期
        if frames >= self.config['attenuation_frames']:
            self.state['is_impact'] = False
            return 1.0

        # 分段衰减策略
        if frames < self.config['strong_attenuation_frames']:
            # 阶段1：强衰减期
            factor = self.config['min_attenuation']

        elif frames < self.config['mid_recovery_frames']:
            # 阶段2：快速恢复期
            t1 = self.config['strong_attenuation_frames']
            t2 = self.config['mid_recovery_frames']
            progress = (frames - t1) / (t2 - t1)

            factor = self.config['min_attenuation'] + \
                     (self.config['mid_attenuation'] - self.config['min_attenuation']) * progress

        else:
            # 阶段3：缓慢恢复期
            t2 = self.config['mid_recovery_frames']
            t3 = self.config['attenuation_frames']
            progress = (frames - t2) / (t3 - t2)

            factor = self.config['mid_attenuation'] + \
                     (1.0 - self.config['mid_attenuation']) * progress

        return float(factor)  # 确保返回float类型

    def update_frame_counters(self):
        """更新内部帧计数器"""
        if self.state['is_impact']:
            self.state['frames_since_impact'] += 1

        self.state['frames_since_last_impact'] += 1
        self.stats['total_frames'] += 1

    def process(self, acc_x: float) -> Tuple[float, bool]:
        """
        处理单帧数据，返回衰减系数
        这是主要的对外接口

        Args:
            acc_x: 当前ACC_X传感器值

        Returns:
            (衰减系数, 是否检测到新冲击)
        """
        # 1. 检测冲击
        new_impact = self.detect_impact(acc_x)

        # 2. 计算衰减系数
        attenuation_factor = self.calculate_attenuation()
        if attenuation_factor != 1.0:
            print(f"当前衰减系数Attenuation={attenuation_factor:.2f}, ")

        # 3. 更新帧计数器
        self.update_frame_counters()

        # 4. 调试输出
        if self.debug and attenuation_factor < 1.0:
            print(f"[Frame {self.stats['total_frames']}] "
                  f"Attenuation={attenuation_factor:.2f}, "
                  f"FramesSinceImpact={self.state['frames_since_impact']}")

        return attenuation_factor, new_impact

    def get_stats(self) -> Dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return {
            'impact_count': self.stats['impact_count'],
            'total_frames': self.stats['total_frames'],
            'is_impact_active': self.state['is_impact'],
            'frames_since_impact': self.state['frames_since_impact'],
            'last_impact_frame': self.stats['last_impact_frame'],
            'last_impact_acc': self.stats['last_impact_acc'],
            'max_impact_acc': self.stats['max_impact_acc'],
        }

    def reset(self):
        """重置所有状态"""
        self.state = {
            'is_impact': False,
            'frames_since_impact': 0,
            'frames_since_last_impact': 999,
            'last_acc_x': 0.0,
            'acc_x_buffer': deque(maxlen=5),
        }
        self.stats = {
            'impact_count': 0,
            'total_frames': 0,
            'last_impact_frame': -1,
            'last_impact_acc': 0.0,
            'max_impact_acc': 0.0,
        }
        if self.debug:
            print("[ImpactAttenuator] 状态已重置")

    def update_config(self, **kwargs):
        """
        动态更新配置参数

        Args:
            可选参数包括: acc_threshold, min_attenuation, 等
        """
        for key, value in kwargs.items():
            if key in ['acc_threshold', 'acc_rate_threshold', 'min_attenuation',
                       'mid_attenuation', 'enable_rate_detection']:
                self.config[key] = value
            elif key.endswith('_ms'):
                # 将毫秒参数转换为帧数
                frame_key = key.replace('_ms', '_frames')
                if frame_key in self.config:
                    self.config[frame_key] = int(value * self.frame_rate / 1000)

        if self.debug:
            print(f"[ImpactAttenuator] 配置已更新: {kwargs}")