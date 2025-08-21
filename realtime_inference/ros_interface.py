"""
ROS接口 - 为后续ROS集成预留
"""

from typing import Dict, List, Callable, Optional
import time
import threading
from queue import Queue


class ROSInterface:
    """ROS接口封装类"""

    def __init__(self, inference_engine):
        """
        初始化ROS接口
        Args:
            inference_engine: 推理引擎实例
        """
        self.engine = inference_engine
        self.callbacks = []
        self.is_running = False
        self.data_queue = Queue(maxsize=100)
        self.worker_thread = None

    def feed_sensor_data(self, timestamp: float, sensor_dict: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        接收传感器数据并推理
        Args:
            timestamp: 时间戳
            sensor_dict: 传感器数据字典 {sensor_name: value}
        Returns:
            关节力矩字典 {joint_name: moment_value} 或 None（如果队列满）
        """
        try:
            # 非阻塞方式放入队列
            self.data_queue.put_nowait((timestamp, sensor_dict))

            # 执行推理
            moments = self.engine.process_frame(sensor_dict)

            # 触发所有回调
            for callback in self.callbacks:
                try:
                    callback(timestamp, moments)
                except Exception as e:
                    print(f"Callback error: {e}")

            return moments

        except:
            # 队列满，丢弃数据
            return None

    def register_callback(self, callback_func: Callable[[float, Dict[str, float]], None]):
        """
        注册推理完成回调函数
        Args:
            callback_func: 回调函数，签名为 (timestamp, moments) -> None
        """
        self.callbacks.append(callback_func)
        print(f"Registered callback: {callback_func.__name__}")

    def unregister_callback(self, callback_func: Callable):
        """
        注销回调函数
        Args:
            callback_func: 要注销的回调函数
        """
        if callback_func in self.callbacks:
            self.callbacks.remove(callback_func)
            print(f"Unregistered callback: {callback_func.__name__}")

    def get_latest_moments(self) -> Optional[Dict[str, float]]:
        """
        获取最新的关节力矩估计
        Returns:
            最新的关节力矩字典或None
        """
        # 这里可以维护一个最新结果的缓存
        # 暂时返回None，需要时可以扩展
        return None

    def start_async_processing(self):
        """启动异步处理线程"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._process_queue)
            self.worker_thread.start()
            print("Started async processing thread")

    def stop_async_processing(self):
        """停止异步处理线程"""
        if self.is_running:
            self.is_running = False
            if self.worker_thread:
                self.worker_thread.join()
            print("Stopped async processing thread")

    def _process_queue(self):
        """处理数据队列的工作线程"""
        while self.is_running:
            try:
                # 带超时的获取，避免阻塞
                timestamp, sensor_data = self.data_queue.get(timeout=0.1)

                # 执行推理
                moments = self.engine.process_frame(sensor_data)

                # 触发回调
                for callback in self.callbacks:
                    try:
                        callback(timestamp, moments)
                    except Exception as e:
                        print(f"Async callback error: {e}")

            except:
                # 队列为空或超时，继续循环
                continue

    def get_status(self) -> Dict:
        """
        获取接口状态
        Returns:
            状态字典
        """
        return {
            'is_running': self.is_running,
            'queue_size': self.data_queue.qsize(),
            'num_callbacks': len(self.callbacks),
            'performance': self.engine.get_performance_stats()
        }


# 示例使用代码
def example_usage():
    """示例：如何使用ROS接口"""
    from inference_engine import InferenceEngine

    # 初始化
    engine = InferenceEngine('configs.hiponly_config')
    ros_interface = ROSInterface(engine)

    # 定义回调函数
    def on_moment_computed(timestamp, moments):
        print(f"[{timestamp:.3f}] Moments: {moments}")

    # 注册回调
    ros_interface.register_callback(on_moment_computed)

    # 模拟发送传感器数据
    import numpy as np
    for i in range(100):
        timestamp = i * 0.005  # 200Hz

        # 模拟传感器数据
        sensor_data = {}
        for name in engine.input_names:
            sensor_data[name] = np.random.randn()

        # 发送数据并获取结果
        moments = ros_interface.feed_sensor_data(timestamp, sensor_data)

        if i % 20 == 0:
            print(f"Processed frame {i}")

    print("Example completed")


if __name__ == "__main__":
    example_usage()