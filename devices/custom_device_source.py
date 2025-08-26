"""
扩展的设备数据源，支持自定义CSV数据和ROS接口
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_inference.data_stream import DataSource
from devices.custom_data_loader import CustomDataLoader, DataPreprocessor
import threading
import time
from typing import Dict, Optional, Tuple, List
from queue import Queue, Empty
import numpy as np


class CustomDeviceSource(DataSource):
    """支持自定义数据格式的设备数据源"""

    def __init__(self, config):
        self.config = config
        self.is_connected = False
        self.data_mode = 'offline'  # 'offline' or 'ros' or 'device'

        # 数据加载器和预处理器
        self.custom_loader = None
        self.preprocessor = None
        self.side = 'r'  # 默认右腿

        # 数据缓冲
        self.data_buffer = Queue(maxsize=100)
        self.streaming_thread = None
        self.is_streaming = False

        # ROS相关（可选）
        self.ros_enabled = False
        self.ros_bridge = None

        # 性能监控
        self.last_frame_time = 0
        self.frame_count = 0

    def load_csv_file(self, csv_path: str, label_filter: Optional[int] = None, side: str = 'r'):
        """
        加载CSV文件
        Args:
            csv_path: CSV文件路径
            label_filter: 标签过滤
            side: 腿部侧面 ('r' 或 'l')
        """
        try:
            # 创建数据加载器
            self.custom_loader = CustomDataLoader(csv_path, label_filter)

            # 创建预处理器
            self.side = side
            self.preprocessor = DataPreprocessor(self.config, side)

            # 设置为离线模式
            self.data_mode = 'offline'
            self.is_connected = True

            print(f"Successfully loaded CSV: {csv_path}")
            print(f"Total frames: {self.custom_loader.get_total_frames()}")
            print(f"Label filter: {label_filter}, Side: {side}")

            return True

        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.is_connected = False
            return False

    def start_streaming(self):
        """启动数据流"""
        if not self.is_connected or self.custom_loader is None:
            print("Device not connected or no data loaded")
            return False

        if self.is_streaming:
            return True

        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_worker)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()

        print("Started data streaming")
        return True

    def stop_streaming(self):
        """停止数据流"""
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2.0)

        # 清空缓冲区
        while not self.data_buffer.empty():
            try:
                self.data_buffer.get_nowait()
            except:
                break

        print("Stopped data streaming")

    def _streaming_worker(self):
        """数据流工作线程"""
        frame_interval = 1.0 / 200.0  # 200Hz

        while self.is_streaming:
            start_time = time.time()

            # 从加载器获取下一帧
            if self.custom_loader and not self.custom_loader.is_finished():
                raw_data = self.custom_loader.get_next_frame()

                if raw_data:
                    # 预处理数据
                    processed_data = self.preprocessor.process(raw_data)

                    # 放入缓冲区（非阻塞）
                    try:
                        self.data_buffer.put_nowait({
                            'raw': raw_data,
                            'processed': processed_data,
                            'timestamp': time.time()
                        })
                        self.frame_count += 1
                    except:
                        # 缓冲区满，丢弃最旧的数据
                        try:
                            self.data_buffer.get_nowait()
                            self.data_buffer.put_nowait({
                                'raw': raw_data,
                                'processed': processed_data,
                                'timestamp': time.time()
                            })
                        except:
                            pass

            # 控制帧率
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def get_next_frame(self) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """
        获取下一帧数据
        Returns:
            (sensor_data, ground_truth) - ground_truth在实时模式下为None
        """
        if not self.is_connected:
            return None, None

        try:
            # 从缓冲区获取数据（非阻塞）
            data = self.data_buffer.get_nowait()
            self.last_frame_time = data['timestamp']

            # 返回处理后的数据
            return data['processed'], None

        except Empty:
            # 缓冲区为空
            return None, None

    def get_next_frame_with_debug(self) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        """
        获取下一帧数据（带调试信息）
        Returns:
            (processed_data, raw_data, debug_info)
        """
        if not self.is_connected:
            return None, None, None

        try:
            data = self.data_buffer.get_nowait()

            # 生成调试信息
            debug_info = {
                'timestamp': data['timestamp'],
                'frame_count': self.frame_count,
                'buffer_size': self.data_buffer.qsize(),
                'preprocessing_info': self.preprocessor.get_debug_info(
                    data['raw'],
                    data['processed']
                )
            }

            return data['processed'], data['raw'], debug_info

        except Empty:
            return None, None, None

    def reset(self):
        """重置数据源"""
        if self.custom_loader:
            self.custom_loader.reset()

        # 清空缓冲区
        while not self.data_buffer.empty():
            try:
                self.data_buffer.get_nowait()
            except:
                break

        self.frame_count = 0

    def is_finished(self) -> bool:
        """检查数据是否已结束"""
        if self.custom_loader:
            return self.custom_loader.is_finished() and self.data_buffer.empty()
        return True

    def get_progress(self) -> float:
        """获取进度"""
        if self.custom_loader:
            return self.custom_loader.get_progress()
        return 0

    def connect_ros(self, topic_name: str = "/exo_sensor_data") -> bool:
        """
        连接ROS话题（预留接口）
        Args:
            topic_name: ROS话题名称
        """
        try:
            # 这里可以添加实际的ROS连接代码
            # import rospy
            # from std_msgs.msg import Float32MultiArray
            # self.ros_subscriber = rospy.Subscriber(topic_name, Float32MultiArray, self.ros_callback)

            print(f"ROS interface not implemented yet. Topic: {topic_name}")
            return False

        except Exception as e:
            print(f"Error connecting to ROS: {e}")
            return False

    def set_preprocessing_params(self, params: Dict):
        """
        设置预处理参数
        Args:
            params: 预处理参数字典
        """
        if self.preprocessor:
            if 'coordinate_transform' in params:
                self.preprocessor.set_coordinate_transform(params['coordinate_transform'])
            if 'unit_conversion' in params:
                self.preprocessor.set_unit_conversion(params['unit_conversion'])

    def get_status(self) -> Dict:
        """获取状态信息"""
        return {
            'connected': self.is_connected,
            'mode': self.data_mode,
            'side': self.side,
            'streaming': self.is_streaming,
            'buffer_size': self.data_buffer.qsize() if self.data_buffer else 0,
            'frame_count': self.frame_count,
            'progress': self.get_progress()
        }


class ROSBridge:
    """
    ROS桥接器（预留接口）
    用于将来集成ROS系统
    """

    def __init__(self):
        self.is_initialized = False
        self.publishers = {}
        self.subscribers = {}

    def initialize(self):
        """初始化ROS节点"""
        try:
            # import rospy
            # rospy.init_node('exo_inference_bridge', anonymous=True)
            # self.is_initialized = True
            print("ROS bridge initialization not implemented")
            return False
        except Exception as e:
            print(f"Failed to initialize ROS: {e}")
            return False

    def publish_data(self, topic: str, data: Dict):
        """发布数据到ROS话题"""
        if not self.is_initialized:
            return False

        # 实际的ROS发布代码
        # msg = Float32MultiArray()
        # msg.data = [data[key] for key in sorted(data.keys())]
        # publisher.publish(msg)

        return True

    def subscribe_data(self, topic: str, callback):
        """订阅ROS话题"""
        if not self.is_initialized:
            return False

        # 实际的ROS订阅代码
        # subscriber = rospy.Subscriber(topic, Float32MultiArray, callback)
        # self.subscribers[topic] = subscriber

        return True