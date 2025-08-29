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
from typing import Dict, Optional, Tuple, List, Callable
from queue import Queue, Empty
import numpy as np

# ROS导入（条件导入，避免没有ROS时报错）
try:
    import rospy
    from std_msgs.msg import Float64MultiArray, Header
    from sensor_msgs.msg import Imu, JointState
    from geometry_msgs.msg import Vector3
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available. ROS features will be disabled.")


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

        # ROS相关
        self.ros_enabled = False
        self.ros_bridge = None
        self.ros_subscriber = None
        self.ros_publisher = None
        self.ros_data_count = 0

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

            # 创建预处理器（始终启用坐标转换）
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

    def connect_ros(self, topic_name: str = "/exo_sensor_data",
                   output_topic: str = "/exo_moments",
                   side: str = 'r') -> bool:
        """
        连接ROS话题
        Args:
            topic_name: 输入传感器数据的ROS话题名称
            output_topic: 输出力矩估计的ROS话题名称
            side: 腿部侧面 ('r' 或 'l')
        """
        if not ROS_AVAILABLE:
            print("ROS is not installed. Please install ROS and rospy.")
            return False

        try:
            # 初始化ROS节点（如果还没有初始化）
            if not rospy.core.is_initialized():
                rospy.init_node('exo_inference_node', anonymous=True)
                print("ROS node initialized: exo_inference_node")

            # 创建预处理器
            self.side = side
            self.preprocessor = DataPreprocessor(self.config, side)

            # 创建ROS桥接器
            self.ros_bridge = ROSBridge(self.config, self.preprocessor)


            self.ros_subscriber = rospy.Subscriber(
                    topic_name,
                    Float64MultiArray,
                    self._ros_array_callback,
                    queue_size=10
            )
            print(f"Subscribed to Float64MultiArray topic: {topic_name}")

            # 创建力矩发布器
            self.ros_publisher = rospy.Publisher(
                output_topic,
                Float64MultiArray,
                queue_size=10
            )
            print(f"Publishing moments to: {output_topic}")

            # 设置为ROS模式
            self.data_mode = 'ros'
            self.ros_enabled = True
            self.is_connected = True
            self.is_streaming = True  # ROS模式下自动开始流

            # 启动ROS spin线程
            self.ros_thread = threading.Thread(target=self._ros_spin_thread)
            self.ros_thread.daemon = True
            self.ros_thread.start()

            print(f"ROS interface connected successfully")
            return True

        except Exception as e:
            print(f"Error connecting to ROS: {e}")
            import traceback
            traceback.print_exc()
            self.ros_enabled = False
            self.is_connected = False
            return False

    def _ros_spin_thread(self):
        """ROS spin线程，保持节点活跃"""
        while self.ros_enabled and not rospy.is_shutdown():
            time.sleep(0.01)

    def _ros_array_callback(self, msg: 'Float32MultiArray'):
        """
        ROS Float32MultiArray消息回调
        期望数据格式: [motorPos, motorVel, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, label]
        """
        try:
            if len(msg.data) >= 8:  # 至少需要8个数据（不包括label）
                # 构造原始数据字典
                raw_data = {
                    'motorPos': msg.data[0] if len(msg.data) > 0 else 0.0,
                    'motorVel': msg.data[1] if len(msg.data) > 1 else 0.0,
                    'acc_x': msg.data[2] if len(msg.data) > 2 else 0.0,
                    'acc_y': msg.data[3] if len(msg.data) > 3 else 0.0,
                    'acc_z': msg.data[4] if len(msg.data) > 4 else 0.0,
                    'gyro_x': msg.data[5] if len(msg.data) > 5 else 0.0,
                    'gyro_y': msg.data[6] if len(msg.data) > 6 else 0.0,
                    'gyro_z': msg.data[7] if len(msg.data) > 7 else 0.0,
                    'label': int(msg.data[8]) if len(msg.data) > 8 else 0
                }

                # 预处理数据
                processed_data = self.preprocessor.process(raw_data)

                # 放入缓冲区
                try:
                    self.data_buffer.put_nowait({
                        'raw': raw_data,
                        'processed': processed_data,
                        'timestamp': rospy.Time.now().to_sec()
                    })
                    self.frame_count += 1
                    self.ros_data_count += 1

                    if self.ros_data_count % 100 == 0:
                        print(f"Received {self.ros_data_count} ROS messages")

                except:
                    # 缓冲区满，丢弃最旧的
                    try:
                        self.data_buffer.get_nowait()
                        self.data_buffer.put_nowait({
                            'raw': raw_data,
                            'processed': processed_data,
                            'timestamp': rospy.Time.now().to_sec()
                        })
                    except:
                        pass

        except Exception as e:
            print(f"Error in ROS callback: {e}")

    def _process_and_buffer(self, raw_data: dict):
        """处理并缓冲数据的通用方法"""
        processed_data = self.preprocessor.process(raw_data)

        try:
            self.data_buffer.put_nowait({
                'raw': raw_data,
                'processed': processed_data,
                'timestamp': rospy.Time.now().to_sec() if ROS_AVAILABLE else time.time()
            })
            self.frame_count += 1
            self.ros_data_count += 1

            if self.ros_data_count % 100 == 0:
                print(f"Processed {self.ros_data_count} ROS messages")

        except:
            # 缓冲区满，丢弃最旧的
            try:
                self.data_buffer.get_nowait()
                self.data_buffer.put_nowait({
                    'raw': raw_data,
                    'processed': processed_data,
                    'timestamp': rospy.Time.now().to_sec() if ROS_AVAILABLE else time.time()
                })
            except:
                pass

    def publish_moments(self, moments: Dict[str, float]):
        """
        发布力矩估计到ROS话题
        Args:
            moments: 力矩字典 {joint_name: moment_value}
        """
        if self.ros_publisher and self.ros_enabled:
            try:
                msg = Float64MultiArray()
                # 按照固定顺序发布力矩值
                msg.data = [moments.get(name, 0.0) for name in self.config.label_names]
                self.ros_publisher.publish(msg)
            except Exception as e:
                print(f"Error publishing moments: {e}")

    def start_streaming(self):
        """启动数据流"""
        if self.data_mode == 'ros':
            # ROS模式下自动开始，不需要额外线程
            print("ROS streaming active")
            return True

        # 离线模式的原始逻辑
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

        if self.data_mode == 'ros':
            self.ros_enabled = False
            if self.ros_subscriber:
                self.ros_subscriber.unregister()
            if self.ros_publisher:
                self.ros_publisher.unregister()
            print("ROS streaming stopped")

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
        """数据流工作线程（离线模式）"""
        frame_interval = 1.0 / 200.0  # 200Hz

        while self.is_streaming:
            start_time = time.time()

            # 从加载器获取下一帧
            if self.custom_loader and not self.custom_loader.is_finished():
                raw_data = self.custom_loader.get_next_frame()

                if raw_data:
                    # 预处理数据（始终进行坐标转换）
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

    def get_next_frame_with_raw(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        获取下一帧数据（带原始数据）
        Returns:
            (processed_data, raw_data)
        """
        if not self.is_connected:
            return None, None

        try:
            data = self.data_buffer.get_nowait()
            return data['processed'], data['raw']

        except Empty:
            return None, None

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
        self.ros_data_count = 0

    def is_finished(self) -> bool:
        """检查数据是否已结束"""
        if self.data_mode == 'ros':
            # ROS模式永不结束（除非手动停止）
            return False

        if self.custom_loader:
            return self.custom_loader.is_finished() and self.data_buffer.empty()
        return True

    def get_progress(self) -> float:
        """获取进度"""
        if self.data_mode == 'ros':
            # ROS模式返回已处理帧数（没有总进度概念）
            return self.ros_data_count

        if self.custom_loader:
            return self.custom_loader.get_progress()
        return 0

    def get_status(self) -> Dict:
        """获取状态信息"""
        status = {
            'connected': self.is_connected,
            'mode': self.data_mode,
            'side': self.side,
            'streaming': self.is_streaming,
            'buffer_size': self.data_buffer.qsize() if self.data_buffer else 0,
            'frame_count': self.frame_count,
            'progress': self.get_progress()
        }

        if self.data_mode == 'ros':
            status['ros_messages_received'] = self.ros_data_count
            status['ros_enabled'] = self.ros_enabled

        return status


class ROSBridge:
    """
    ROS桥接器
    用于处理ROS消息和数据转换
    """

    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.is_initialized = ROS_AVAILABLE
        self.publishers = {}
        self.subscribers = {}
        self.message_count = 0

    def create_combined_subscriber(self, imu_topic: str, joint_topic: str, callback: Callable):
        """
        创建组合订阅器，同时订阅IMU和关节数据
        Args:
            imu_topic: IMU话题名
            joint_topic: 关节话题名
            callback: 处理组合数据的回调函数
        """
        if not self.is_initialized:
            return False

        try:
            # 缓存最新的IMU和关节数据
            self.latest_imu = None
            self.latest_joint = None
            self.combined_callback = callback

            def imu_callback(msg):
                self.latest_imu = msg
                self._check_and_combine()

            def joint_callback(msg):
                self.latest_joint = msg
                self._check_and_combine()

            # 订阅话题
            self.subscribers['imu'] = rospy.Subscriber(imu_topic, Imu, imu_callback, queue_size=10)
            self.subscribers['joint'] = rospy.Subscriber(joint_topic, JointState, joint_callback, queue_size=10)

            print(f"Subscribed to combined topics: {imu_topic}, {joint_topic}")
            return True

        except Exception as e:
            print(f"Error creating combined subscriber: {e}")
            return False

    def _check_and_combine(self):
        """检查并组合IMU和关节数据"""
        if self.latest_imu and self.latest_joint:
            # 组合数据
            combined_data = {
                'motorPos': self.latest_joint.position[0] * 180.0 / np.pi if len(self.latest_joint.position) > 0 else 0.0,
                'motorVel': self.latest_joint.velocity[0] * 180.0 / np.pi if len(self.latest_joint.velocity) > 0 else 0.0,
                'acc_x': self.latest_imu.linear_acceleration.x,
                'acc_y': self.latest_imu.linear_acceleration.y,
                'acc_z': self.latest_imu.linear_acceleration.z,
                'gyro_x': self.latest_imu.angular_velocity.x,
                'gyro_y': self.latest_imu.angular_velocity.y,
                'gyro_z': self.latest_imu.angular_velocity.z,
                'label': 0
            }

            # 调用回调函数
            if self.combined_callback:
                self.combined_callback(combined_data)

            self.message_count += 1
            if self.message_count % 100 == 0:
                print(f"Combined {self.message_count} messages")

    def publish_diagnostic(self, topic: str, info: Dict):
        """
        发布诊断信息
        Args:
            topic: 诊断话题名
            info: 诊断信息字典
        """
        if not self.is_initialized:
            return False

        try:
            if topic not in self.publishers:
                from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
                self.publishers[topic] = rospy.Publisher(topic, DiagnosticArray, queue_size=10)

            # 创建诊断消息
            diag_array = DiagnosticArray()
            diag_array.header.stamp = rospy.Time.now()

            status = DiagnosticStatus()
            status.name = "exo_inference"
            status.level = DiagnosticStatus.OK
            status.message = "Running"

            for key, value in info.items():
                status.values.append(KeyValue(key=str(key), value=str(value)))

            diag_array.status.append(status)
            self.publishers[topic].publish(diag_array)

            return True

        except Exception as e:
            print(f"Error publishing diagnostic: {e}")
            return False