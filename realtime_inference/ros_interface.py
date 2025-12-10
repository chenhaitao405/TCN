#!/usr/bin/env python
"""
ROS推理节点可视化界面
提供实时力矩曲线显示、参数配置和控制功能
"""

import sys
import os
import time
import argparse
from collections import deque
import threading

import numpy as np
# ROS imports
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray

# PyQt5 imports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyqtgraph as pg
from utils.config_utils import ConfigManager

import warnings

warnings.filterwarnings('ignore', message='dropout2d: Received a 3D input')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime_inference.inference_engine import InferenceEngine
from devices.custom_data_loader import DataPreprocessor


class PublishWorker(QThread):
    """力矩发布工作线程 - 以100Hz频率固定发送"""
    status_update = pyqtSignal(float)  # 发布频率
    error_occurred = pyqtSignal(str)

    def __init__(self, topic: str = '/moment', label_hip:bool = False):
        super().__init__()
        self.topic = topic
        self.is_running = False
        self.publisher = None
        self.label_hip = label_hip

        # 共享数据和线程锁
        self.moments_lock = threading.Lock()
        self.current_moments = {}
        self.timestamp_sensor = 0 #传感器时间戳
        self.body_weight = 70.0
        self.num_joints = 0  # 关节数量

        # 性能统计
        self.publish_count = 0
        self.last_stats_time = time.time()

    def set_topic(self, topic: str):
        """更新发布话题"""
        with self.moments_lock:
            self.topic = topic
            if self.publisher:
                self.publisher.unregister()
                self.publisher = rospy.Publisher(
                    self.topic,
                    Float64MultiArray,
                    queue_size=10
                )

    def update_moments_with_timestamp(self, moments: dict, body_weight: float, timestamp_sensor: float):
        """更新力矩数据和时间戳（线程安全）"""
        with self.moments_lock:
            self.current_moments = moments.copy()
            self.body_weight = body_weight
            self.timestamp_sensor = timestamp_sensor
            if moments:
                self.num_joints = len(moments)

    def set_num_joints(self, num: int):
        """设置关节数量"""
        with self.moments_lock:
            self.num_joints = num
#TODO: 髋关节的发布要修改
    def run_knee(self):
        self.is_running = True

        # 创建发布器
        self.publisher = rospy.Publisher(
            self.topic,
            Float64MultiArray,
            queue_size=10
        )

        # 100Hz -> 10ms周期
        rate = rospy.Rate(100)  # 使用设置的频率

        while self.is_running and not rospy.is_shutdown():

            try:
                msg = Float64MultiArray()

                with self.moments_lock:
                    if self.current_moments:
                        # 发送值 = 推理值 × 体重 × 0.2
                        moment_values = [value * self.body_weight * 0.2
                                         for value in self.current_moments.values()]

                        # 将力矩值和时间戳打包在一起发送
                        # 格式：[moment1, moment2, ..., momentN, timestamp_sensor, timestamp_publish]
                        msg.data = moment_values + [
                            self.timestamp_sensor  # 原始传感器时间戳
                        ]
                    else:
                        # 没有数据时发送零值 + 时间戳
                        msg.data = [0.0] * max(self.num_joints, 1) + [0.0]

                # 发布消息
                if self.publisher:
                    self.publisher.publish(msg)
                    self.publish_count += 1

                # 更新发布频率统计
                current_time = time.time()
                if current_time - self.last_stats_time >= 1.0:
                    publish_rate = self.publish_count / (current_time - self.last_stats_time)
                    self.status_update.emit(publish_rate)
                    self.publish_count = 0
                    self.last_stats_time = current_time

                rate.sleep()

            except Exception as e:
                self.error_occurred.emit(f"发布错误: {str(e)}")


    def run(self):

        if self.label_hip:
            self.run_hip()
        else:
            self.run_knee()



    def run_hip(self):
        """线程主循环 - 100Hz固定频率发布"""
        self.is_running = True

        # 创建发布器
        self.publisher = rospy.Publisher(
            self.topic,
            Float32MultiArray,
            queue_size=10
        )

        # 100Hz -> 10ms周期
        rate = rospy.Rate(100)  # 使用设置的频率

        while self.is_running and not rospy.is_shutdown():

            try:
                msg = Float32MultiArray()

                with self.moments_lock:
                    if self.current_moments:
                        # 发送值 = 推理值 × 体重 × 0.2
                        moment_values = [value * self.body_weight * 0.2
                                    for value in self.current_moments.values()]

                        msg.data = moment_values

                    else:
                        # 没有数据时发送零值 + 时间戳
                        msg.data = [0.0] * max(self.num_joints, 1)

                # 发布消息
                if self.publisher:
                    self.publisher.publish(msg)
                    self.publish_count += 1

                # 更新发布频率统计
                current_time = time.time()
                if current_time - self.last_stats_time >= 1.0:
                    publish_rate = self.publish_count / (current_time - self.last_stats_time)
                    self.status_update.emit(publish_rate)
                    self.publish_count = 0
                    self.last_stats_time = current_time

                rate.sleep()

            except Exception as e:
                self.error_occurred.emit(f"发布错误: {str(e)}")

    def stop(self):
        """停止线程"""
        self.is_running = False
        if self.publisher:
            self.publisher.unregister()
            self.publisher = None


class InferenceWorker(QThread):
    """推理工作线程"""
    # 修改信号定义，添加 timestamp_sensor 和 timestamp_back 参数
    data_ready = pyqtSignal(dict, float, dict, float, float)
    # sensor_data, moments, timestamp, return_moment, timestamp_sensor, timestamp_back
    status_update = pyqtSignal(dict)  # performance stats
    error_occurred = pyqtSignal(str)  # error message


    def __init__(self, config_path: str, side: str = 'r'):
        super().__init__()
        self.config_path = config_path
        self.side = side
        self.is_running = False
        self.is_paused = False

        # 初始化推理引擎
        self.engine = InferenceEngine(config_path)
        self.preprocessor = DataPreprocessor(self.engine.config, side)

        # 性能统计
        self.frame_count = 0
        self.last_time = time.time()
        self.receive_rate = 0

        # 时间基准（用于计算相对时间）
        self.start_timestamp = None

    def run(self):
        """线程主循环"""
        self.is_running = True

        # 注意：时间基准（start_timestamp）会在第一次接收到数据时自动设置

        def sensor_callback_hip(msg):
            """处理传感器数据回调"""
            if self.is_paused or not self.is_running:
                return

            try:
                # 解析传感器数据
                if len(msg.data) < 1:
                    self.error_occurred.emit(f"数据不足: {len(msg.data)} 值 (需要至少1个)")
                    return
                ## 将角度转弧度
                # 直接转换前4个弧度值转为角度
                radians = np.degrees(np.array(msg.data[:4])) * -1
                # 构建字典
                raw_data = {
                    'hip_angle_l': radians[0],
                    'hip_angle_r': radians[1],
                    'hip_vel_l': radians[2],
                    'hip_vel_r': radians[3],
                }

                processed_data = self.preprocessor.process_hip(raw_data)
                return_moment = {
                    'hip_flexion_l_moment': msg.data[-2]/100,
                    'hip_flexion_r_moment': msg.data[-1]/100,
                }
                # 执行推理 - 添加计时
                inference_start = time.time()
                moments = self.engine.process_frame_hip(processed_data)

                inference_end = time.time()
                inference_time = (inference_end - inference_start) * 1000  # 转换为毫秒

                # 打印耗时统计
                # print(f"预处理耗时: {preprocess_time:.3f} ms")
                # print(f"推理耗时: {inference_time:.3f} ms")
                # print(f"总耗时: {(preprocess_time + inference_time):.3f} ms")
                # print("-" * 40)  # 分隔线，便于查看

                # 获取时间戳（使用相对时间）
                current_timestamp = rospy.Time.now().to_sec()

                # 如果是第一次，记录起始时间
                if self.start_timestamp is None:
                    self.start_timestamp = current_timestamp

                # 计算相对时间（从开始推理到现在的秒数）
                relative_time = current_timestamp - self.start_timestamp

                # 发送数据信号，使用相对时间（含对ros发布线程更新力矩）
                self.data_ready.emit(
                    moments,
                    relative_time,return_moment,0.0,0.0
                )

                # 更新统计
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.receive_rate = self.frame_count / (current_time - self.last_time)
                    stats = self.engine.get_performance_stats()
                    stats['receive_rate'] = self.receive_rate
                    stats['frame_count'] = self.frame_count
                    self.status_update.emit(stats)
                    self.frame_count = 0
                    self.last_time = current_time

            except Exception as e:
                self.error_occurred.emit(str(e))

        def sensor_callback(msg):
            """处理传感器数据回调"""
            if self.is_paused or not self.is_running:
                return

            try:
                # 解析传感器数据
                if len(msg.data) < 9:
                    self.error_occurred.emit(f"数据不足: {len(msg.data)} 值 (需要至少9个)")
                    return

                # 构建原始数据字典（索引0-7是传感器数据，索引8是返回的力矩值）
                raw_data = {
                    'motorPos': msg.data[0],
                    'motorVel': msg.data[1],
                    'acc_x': msg.data[2],
                    'acc_y': msg.data[3],
                    'acc_z': msg.data[4],
                    'gyro_x': msg.data[5],
                    'gyro_y': msg.data[6],
                    'gyro_z': msg.data[7],
                    'moment': msg.data[8],  # 返回值（用于对比）
                    'timestamp_sensor': msg.data[9] if len(msg.data) > 9 else 0,
                    'timestamp_back': msg.data[10] if len(msg.data) > 10 else 0,
                }

                # 保存返回值
                timestamp_sensor = raw_data['timestamp_sensor']
                timestamp_back = raw_data['timestamp_back']

                # 预处理数据 - 添加计时
                preprocess_start = time.time()
                processed_data = self.preprocessor.process(raw_data)
                preprocess_end = time.time()
                preprocess_time = (preprocess_end - preprocess_start) * 1000  # 转换为毫秒

                # 执行推理 - 添加计时
                inference_start = time.time()
                moments = self.engine.process_frame(processed_data)

                inference_end = time.time()
                inference_time = (inference_end - inference_start) * 1000  # 转换为毫秒

                # 打印耗时统计
                # print(f"预处理耗时: {preprocess_time:.3f} ms")
                # print(f"推理耗时: {inference_time:.3f} ms")
                # print(f"总耗时: {(preprocess_time + inference_time):.3f} ms")
                # print("-" * 40)  # 分隔线，便于查看

                # 获取时间戳（使用相对时间）
                current_timestamp = rospy.Time.now().to_sec()

                # 如果是第一次，记录起始时间
                if self.start_timestamp is None:
                    self.start_timestamp = current_timestamp

                # 计算相对时间（从开始推理到现在的秒数）
                relative_time = current_timestamp - self.start_timestamp

                first_key = list(moments.keys())[0]
                return_moments = {first_key: raw_data['moment']}

                # 发送数据信号，使用相对时间（含对ros发布线程更新力矩）
                self.data_ready.emit(
                    moments,
                    relative_time,
                    return_moments,
                    timestamp_sensor,  # 传感器时间戳
                    timestamp_back     # 返回时间戳
                )

                # 更新统计
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.receive_rate = self.frame_count / (current_time - self.last_time)
                    stats = self.engine.get_performance_stats()
                    stats['receive_rate'] = self.receive_rate
                    stats['frame_count'] = self.frame_count
                    self.status_update.emit(stats)
                    self.frame_count = 0
                    self.last_time = current_time

            except Exception as e:
                self.error_occurred.emit(str(e))

        # 订阅传感器数据
        if any("hip" in label_name for label_name in self.engine.label_names):
            self.subscriber = rospy.Subscriber(
                '/wgg_msg',
                Float32MultiArray,
                sensor_callback_hip,
                queue_size=10
            )
        else:
            self.subscriber = rospy.Subscriber(
                '/motor12_left',
                Float64MultiArray,
                sensor_callback,
                queue_size=10
            )

        # 保持线程运行
        while self.is_running and not rospy.is_shutdown():
            QThread.msleep(10)

    def pause(self):
        """暂停推理"""
        self.is_paused = True

    def resume(self):
        """恢复推理"""
        self.is_paused = False

    def reset_time(self):
        """重置时间基准"""
        self.start_timestamp = None

    def stop(self):
        """停止线程"""
        self.is_running = False
        self.start_timestamp = None
        if hasattr(self, 'subscriber'):
            self.subscriber.unregister()


class ROSInferenceUI(QMainWindow):
    """ROS推理节点可视化界面"""

    def __init__(self, config_path: str, side: str = 'r'):
        super().__init__()
        self.config_path = config_path

        # 复用现有的配置加载器
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)
        self.side = side
        self.label_hip = False
        if any("hip" in label_name for label_name in self.config.label_names):
            self.label_hip = True


        # 初始化ROS节点
        rospy.init_node('exo_inference_ui', anonymous=True)

        # 状态标志
        self.is_inferencing = False
        self.is_publishing = False
        self.body_weight = 70.0  # 默认体重

        # ROS相关
        self.sub_topic = '/motor12_left'
        self.pub_topic = '/moment'

        # 工作线程
        self.inference_worker = None
        self.publish_worker = None


        # 数据缓存
        self.plot_buffer_size = 1000
        self.time_buffer = deque(maxlen=self.plot_buffer_size)
        self.moment_buffers = {}
        self.return_moment_buffers = {}  # 新增：返回值缓存
        self.current_moments = {}
        self.current_return_moment = 0  # 新增：当前返回值
        self.current_joint = None

        # 性能数据
        self.inference_speed = 0
        self.receive_rate = 0
        self.publish_rate = 0
        self.frame_count = 0

        # 时延统计
        self.current_latency = 0.0  # 当前时延
        self.latency_sum = 0.0  # 时延总和
        self.latency_count = 0  # 时延计数
        self.avg_latency = 0.0  # 平均时延
        self.max_latency = 0.0  # 最大时延
        self.min_latency = float('inf')  # 最小时延
        self.first_return_received = False  # 标记是否收到第一个返回值

        # UI元素（会在init_ui中初始化）
        self.runtime_label = None

        # 绘图节流相关
        self.plot_fps = 30  # 目标刷新率，20~50 之间都可以；示例用 30Hz
        self._plot_dirty = False  # 是否有新数据需要画
        self._last_axis_update = 0.0  # 上次坐标轴更新的时间点
        self._axis_update_interval = 0.25  # 坐标轴 250ms 更新一次，避免每帧重算

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("ROS外骨骼关节力矩实时推理系统")
        self.setGeometry(100, 100, 1200, 800)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:enabled:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)

        # 主widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建各个面板
        main_layout.addWidget(self.create_status_panel())
        main_layout.addWidget(self.create_config_panel())
        main_layout.addWidget(self.create_control_panel())
        main_layout.addWidget(self.create_performance_panel())
        main_layout.addWidget(self.create_plot_panel(), stretch=1)
        main_layout.addWidget(self.create_log_panel())

        # 启动绘图节流定时器
        self.plot_timer = QTimer(self)
        self.plot_timer.setTimerType(Qt.PreciseTimer)  # 更精确的定时
        self.plot_timer.setInterval(int(1000 / self.plot_fps))  # 例如 33ms ≈ 30Hz
        self.plot_timer.timeout.connect(self._on_plot_timer)
        self.plot_timer.start()

        # 创建状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪")

    def create_status_panel(self) -> QWidget:
        """创建系统状态面板"""
        panel = QGroupBox("系统状态")
        layout = QHBoxLayout()

        # 模型信息
        model_name = os.path.basename(self.config_path)
        layout.addWidget(QLabel(f"模型: {model_name}"))

        # 设备信息
        layout.addWidget(QLabel("设备: CUDA"))

        # 腿侧信息
        side_text = "右腿" if self.side == 'r' else "左腿"
        layout.addWidget(QLabel(f"腿侧: {side_text}"))

        layout.addSpacing(20)

        # ROS连接状态
        self.ros_status_label = QLabel("● ROS状态")
        self.ros_status_label.setStyleSheet("color: green;")
        layout.addWidget(self.ros_status_label)

        # 推理状态
        self.inference_status_label = QLabel("● 推理状态: 停止")
        self.inference_status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.inference_status_label)

        # 发布状态
        self.publish_status_label = QLabel("● 发布状态: 停止")
        self.publish_status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.publish_status_label)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_config_panel(self) -> QWidget:
        """创建参数配置面板"""
        panel = QGroupBox("参数配置")
        layout = QHBoxLayout()

        # 体重输入
        layout.addWidget(QLabel("体重(kg):"))
        self.weight_input = QSpinBox()
        self.weight_input.setRange(10, 200)
        self.weight_input.setValue(70)
        self.weight_input.setSuffix(" kg")
        self.weight_input.valueChanged.connect(self.on_weight_changed)
        layout.addWidget(self.weight_input)

        layout.addSpacing(30)

        # 订阅话题
        layout.addWidget(QLabel("订阅话题:"))
        self.sub_topic_input = QLineEdit(self.sub_topic)
        self.sub_topic_input.setMinimumWidth(150)
        layout.addWidget(self.sub_topic_input)

        self.sub_topic_btn = QPushButton("应用")
        self.sub_topic_btn.clicked.connect(self.on_sub_topic_changed)
        layout.addWidget(self.sub_topic_btn)

        layout.addSpacing(30)

        # 发布话题
        layout.addWidget(QLabel("发布话题:"))
        self.pub_topic_input = QLineEdit(self.pub_topic)
        self.pub_topic_input.setMinimumWidth(150)
        layout.addWidget(self.pub_topic_input)

        self.pub_topic_btn = QPushButton("应用")
        self.pub_topic_btn.clicked.connect(self.on_pub_topic_changed)
        layout.addWidget(self.pub_topic_btn)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QGroupBox("控制面板")
        layout = QHBoxLayout()

        # 推理控制
        self.start_inference_btn = QPushButton("▶ 开始推理")
        self.start_inference_btn.clicked.connect(self.on_start_inference)
        layout.addWidget(self.start_inference_btn)

        self.pause_inference_btn = QPushButton("⏸ 暂停推理")
        self.pause_inference_btn.clicked.connect(self.on_pause_inference)
        self.pause_inference_btn.setEnabled(False)
        layout.addWidget(self.pause_inference_btn)

        self.stop_inference_btn = QPushButton("⏹ 停止推理")
        self.stop_inference_btn.clicked.connect(self.on_stop_inference)
        self.stop_inference_btn.setEnabled(False)
        layout.addWidget(self.stop_inference_btn)

        layout.addSpacing(50)

        # 发送控制
        self.start_publish_btn = QPushButton("📤 发送力矩")
        self.start_publish_btn.clicked.connect(self.on_start_publish)
        layout.addWidget(self.start_publish_btn)

        self.stop_publish_btn = QPushButton("⏹ 停止发送")
        self.stop_publish_btn.clicked.connect(self.on_stop_publish)
        self.stop_publish_btn.setEnabled(False)
        layout.addWidget(self.stop_publish_btn)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_performance_panel(self) -> QWidget:
        """创建性能监控面板"""
        panel = QGroupBox("性能监控")
        layout = QHBoxLayout()

        # 推理速度
        self.inference_speed_label = QLabel("推理速度: -- ms/帧")
        layout.addWidget(self.inference_speed_label)

        # 接收帧率
        self.receive_rate_label = QLabel("接收帧率: -- Hz")
        layout.addWidget(self.receive_rate_label)

        # 发送帧率（固定100Hz）
        self.publish_rate_label = QLabel("发送帧率: -- Hz")
        layout.addWidget(self.publish_rate_label)

        # 已处理帧数
        self.frame_count_label = QLabel("已处理帧数: 0")
        layout.addWidget(self.frame_count_label)

        # 运行时间
        self.runtime_label = QLabel("运行时间: 0.0 s")
        layout.addWidget(self.runtime_label)

        layout.addSpacing(30)

        # 时延统计 - 新增部分
        self.latency_label = QLabel("当前时延: -- ms")
        self.latency_label.setStyleSheet("color: blue; font-weight: bold;")
        layout.addWidget(self.latency_label)

        self.avg_latency_label = QLabel("平均时延: -- ms")
        self.avg_latency_label.setStyleSheet("color: green;")
        layout.addWidget(self.avg_latency_label)

        self.latency_range_label = QLabel("时延范围: -- ~ -- ms")
        self.latency_range_label.setStyleSheet("color: gray;")
        layout.addWidget(self.latency_range_label)

        layout.addSpacing(30)

        # 缓冲区进度条
        layout.addWidget(QLabel("缓冲区:"))
        self.buffer_progress = QProgressBar()
        self.buffer_progress.setMaximum(100)
        self.buffer_progress.setMinimumWidth(150)
        layout.addWidget(self.buffer_progress)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_plot_panel(self) -> QWidget:
        """创建实时曲线显示面板"""
        panel = QGroupBox("实时力矩曲线 (发送值 vs 返回值)")
        layout = QVBoxLayout()

        # 关节选择
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("选择关节:"))
        self.joint_combo = QComboBox()
        self.joint_combo.currentTextChanged.connect(self.on_joint_changed)
        select_layout.addWidget(self.joint_combo)

        # 当前值显示
        self.current_value_label = QLabel("发送值: -- Nm")
        self.current_value_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        select_layout.addWidget(self.current_value_label)

        self.return_value_label = QLabel("返回值: -- Nm")
        self.return_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        select_layout.addWidget(self.return_value_label)

        self.peak_value_label = QLabel("峰值: -- Nm")
        self.peak_value_label.setStyleSheet("font-size: 14px;")
        select_layout.addWidget(self.peak_value_label)

        select_layout.addStretch()
        layout.addLayout(select_layout)

        # 设置pyqtgraph
        pg.setConfigOptions(antialias=False)

        # 创建绘图控件
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', '力矩', units='Nm')
        self.plot_widget.setLabel('bottom', '时间（从开始推理）', units='s')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(-10, 10)  # 初始Y轴范围，会自动调整
        self.plot_widget.addLegend()

        # 创建曲线 - 修改名称
        self.moment_curve = self.plot_widget.plot(
            pen=pg.mkPen('g', width=2),
            name="发送值"
        )
        self.return_moment_curve = self.plot_widget.plot(
            pen=pg.mkPen('b', width=2, style=Qt.DashLine),
            name="返回值"
        )

        layout.addWidget(self.plot_widget)
        panel.setLayout(layout)
        return panel

    def create_log_panel(self) -> QWidget:
        """创建日志面板"""
        panel = QGroupBox("系统日志")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)

        panel.setLayout(layout)
        return panel

    def create_log_panel(self) -> QWidget:
        """创建日志面板"""
        panel = QGroupBox("系统日志")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)

        panel.setLayout(layout)
        return panel

    def add_log(self, level: str, message: str):
        """添加日志信息"""
        timestamp = time.strftime("%H:%M:%S")
        color = {
            "INFO": "black",
            "WARN": "orange",
            "ERROR": "red"
        }.get(level, "black")

        log_html = f'<span style="color: {color}">[{timestamp}] [{level}] {message}</span>'
        self.log_text.append(log_html)

        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_weight_changed(self, value: int):
        """体重改变回调"""
        self.body_weight = value
        # 更新发布线程的体重值
        if self.publish_worker and self.current_moments:
            self.publish_worker.update_moments_with_timestamp(self.current_moments, self.body_weight)
        self.add_log("INFO", f"体重更新为: {value} kg (影响发送值计算)")
        # 立即更新当前显示的发送值
        self.update_current_values()
        self.update_plot()

    def on_sub_topic_changed(self):
        """订阅话题改变"""
        new_topic = self.sub_topic_input.text()
        if new_topic and new_topic != self.sub_topic:
            # 如果正在推理，需要重启
            if self.is_inferencing:
                self.add_log("WARN", "请先停止推理再修改订阅话题")
                return

            self.sub_topic = new_topic
            self.add_log("INFO", f"订阅话题更新为: {new_topic}")

    def on_pub_topic_changed(self):
        """发布话题改变"""
        new_topic = self.pub_topic_input.text()
        if new_topic and new_topic != self.pub_topic:
            self.pub_topic = new_topic

            # 如果正在发布，更新发布器
            if self.is_publishing and self.publish_worker:
                self.publish_worker.set_topic(new_topic)

            self.add_log("INFO", f"发布话题更新为: {new_topic}")

    def on_start_inference(self):
        """开始推理"""
        # 清空之前的数据缓存
        self.time_buffer.clear()
        for buffer in self.moment_buffers.values():
            buffer.clear()
        for buffer in self.return_moment_buffers.values():
            buffer.clear()

        if self.inference_worker is None:
            self.inference_worker = InferenceWorker(self.config_path, self.side)
            # 连接信号时参数数量要匹配
            #TODO: 修改髋关节数据发送回调
            if any("hip" in label_name for label_name in self.inference_worker.engine.label_names):
                self.inference_worker.data_ready.connect(self.on_data_received_hip)
            else:
                self.inference_worker.data_ready.connect(self.on_data_received)

            self.inference_worker.status_update.connect(self.on_status_update)
            self.inference_worker.error_occurred.connect(self.on_error)

        # 初始化关节列表
        if not self.joint_combo.count():
            for name in self.inference_worker.engine.label_dir_names:
                self.joint_combo.addItem(name)
                self.moment_buffers[name] = deque(maxlen=self.plot_buffer_size)
                self.return_moment_buffers[name] = deque(maxlen=self.plot_buffer_size)  # 初始化返回值缓存

            if self.joint_combo.count() > 0:
                self.current_joint = self.joint_combo.itemText(0)

                # 通知发布线程关节数量
                if self.publish_worker:
                    self.publish_worker.set_num_joints(self.joint_combo.count())

        # 重置时间基准
        self.inference_worker.reset_time()

        self.inference_worker.start()
        self.is_inferencing = True

        # 更新UI状态
        self.start_inference_btn.setEnabled(False)
        self.pause_inference_btn.setEnabled(True)
        self.stop_inference_btn.setEnabled(True)
        self.inference_status_label.setText("● 推理状态: 运行中")
        self.inference_status_label.setStyleSheet("color: green;")

        self.add_log("INFO", "推理已启动")

    def on_pause_inference(self):
        """暂停/恢复推理"""
        if self.inference_worker:
            if self.pause_inference_btn.text() == "⏸ 暂停推理":
                self.inference_worker.pause()
                self.pause_inference_btn.setText("▶ 恢复推理")
                self.inference_status_label.setText("● 推理状态: 暂停")
                self.inference_status_label.setStyleSheet("color: orange;")
                self.add_log("INFO", "推理已暂停")
            else:
                self.inference_worker.resume()
                self.pause_inference_btn.setText("⏸ 暂停推理")
                self.inference_status_label.setText("● 推理状态: 运行中")
                self.inference_status_label.setStyleSheet("color: green;")
                self.add_log("INFO", "推理已恢复")

    def on_stop_inference(self):
        """停止推理"""
        if self.inference_worker:
            self.inference_worker.stop()
            self.inference_worker.wait()
            self.inference_worker = None

        self.is_inferencing = False

        # 清空图表
        self.moment_curve.setData([], [])
        self.return_moment_curve.setData([], [])

        # 重置显示值
        self.current_value_label.setText("发送值: -- Nm")
        self.return_value_label.setText("返回值: -- Nm")
        self.peak_value_label.setText("峰值: -- Nm")
        if self.runtime_label:
            self.runtime_label.setText("运行时间: 0.0 s")

        # 重置时延统计
        self.current_latency = 0.0
        self.latency_sum = 0.0
        self.latency_count = 0
        self.avg_latency = 0.0
        self.max_latency = 0.0
        self.min_latency = float('inf')
        self.first_return_received = False  # 重置首次返回标志
        self.latency_label.setText("当前时延: -- ms")
        self.avg_latency_label.setText("平均时延: -- ms")
        self.latency_range_label.setText("时延范围: -- ~ -- ms")

        # 更新UI状态
        self.start_inference_btn.setEnabled(True)
        self.pause_inference_btn.setEnabled(False)
        self.pause_inference_btn.setText("⏸ 暂停推理")
        self.stop_inference_btn.setEnabled(False)
        self.inference_status_label.setText("● 推理状态: 停止")
        self.inference_status_label.setStyleSheet("color: gray;")

        self.add_log("INFO", "推理已停止")

    def on_start_publish(self):
        """开始发布力矩"""
        # 创建发布线程
        if self.publish_worker is None:
            self.publish_worker = PublishWorker(self.pub_topic,self.label_hip)
            self.publish_worker.status_update.connect(self.on_publish_rate_update)
            self.publish_worker.error_occurred.connect(self.on_error)

            # 设置关节数量
            if self.joint_combo.count() > 0:
                self.publish_worker.set_num_joints(self.joint_combo.count())

            # 如果有当前数据，立即更新
            # if self.current_moments:
            #     self.publish_worker.update_moments_with_timestamp(self.current_moments, self.body_weight)

        self.publish_worker.start()
        self.is_publishing = True

        # 更新UI状态
        self.start_publish_btn.setEnabled(False)
        self.stop_publish_btn.setEnabled(True)
        self.publish_status_label.setText("● 发布状态: 运行中(100Hz)")
        self.publish_status_label.setStyleSheet("color: green;")

        self.add_log("INFO", f"开始以100Hz频率发布力矩到 {self.pub_topic}")

    def on_stop_publish(self):
        """停止发布力矩"""
        if self.publish_worker:
            self.publish_worker.stop()
            self.publish_worker.wait()

        self.is_publishing = False

        # 更新UI状态
        self.start_publish_btn.setEnabled(True)
        self.stop_publish_btn.setEnabled(False)
        self.publish_status_label.setText("● 发布状态: 停止")
        self.publish_status_label.setStyleSheet("color: gray;")

        self.add_log("INFO", "停止发布力矩")

    def on_joint_changed(self, joint_name: str):
        """关节选择改变"""
        self.current_joint = joint_name
        self._plot_dirty = True

#TODO: 髋部数据回调重写
    def on_data_received_hip(self, moments: dict,relative_time: float,
                         return_moment: dict, timestamp_sensor: float, timestamp_back: float):
        """接收到推理数据 - 包含时间戳
        timestamp_sensor 当前传感器对应的时间戳
        timestamp_back: 当前力矩对应推理时刻的时间戳
        （时延 = timestamp_sensor-timestamp_back）
        """
        # 计算时延（转换为毫秒）
        # 只有当 timestamp_back 不为0时才开始统计（刚启动时没有返回值）
        # 更新时间缓存（relative_time已经是相对时间，单位：秒）
        self.time_buffer.append(relative_time)

        # 更新力矩缓存和返回值缓存
        for joint_name, value in moments.items():
            if joint_name in self.moment_buffers:
                self.moment_buffers[joint_name].append(value)
                # 为每个关节保存相同的返回值
                self.return_moment_buffers[joint_name].append(return_moment[joint_name])

        # 保存当前力矩值和返回值
        self.current_moments = moments
        self.current_return_moment = return_moment

        if self.publish_worker and self.is_publishing:
            self.publish_worker.update_moments_with_timestamp(
                moments,
                self.body_weight,
                relative_time  # 传递传感器时间戳
            )

        if timestamp_sensor > 0 and timestamp_back > 0:
            # 如果是第一次收到返回值，记录日志
            if not self.first_return_received:
                self.first_return_received = True
                self.add_log("INFO", "开始接收返回值，时延统计已启动")

            self.current_latency = (timestamp_sensor - timestamp_back) * 10  # 转换为毫秒

            # 更新统计数据
            self.latency_sum += self.current_latency
            self.latency_count += 1
            self.avg_latency = self.latency_sum / self.latency_count

            # 更新最大最小值
            self.max_latency = max(self.max_latency, self.current_latency)
            self.min_latency = min(self.min_latency, self.current_latency)

        # 更新运行时间显示
        if self.runtime_label:
            self.runtime_label.setText(f"运行时间: {relative_time:.1f} s")

        # 更新界面
        self._plot_dirty = True

    def on_data_received(self,  moments: dict, relative_time: float,
                         return_moment: dict, timestamp_sensor: float, timestamp_back: float):
        """接收到推理数据 - 包含时间戳
        timestamp_sensor 当前传感器对应的时间戳
        timestamp_back: 当前力矩对应推理时刻的时间戳
        （时延 = timestamp_sensor-timestamp_back）
        """
        # 计算时延（转换为毫秒）
        # 只有当 timestamp_back 不为0时才开始统计（刚启动时没有返回值）
        # 更新时间缓存（relative_time已经是相对时间，单位：秒）
        self.time_buffer.append(relative_time)

        # 更新力矩缓存和返回值缓存
        for joint_name, value in moments.items():
            if joint_name in self.moment_buffers:
                self.moment_buffers[joint_name].append(value)
                if joint_name in self.return_moment_buffers and joint_name in return_moment:
                    self.return_moment_buffers[joint_name].append(return_moment[joint_name])

        # 保存当前力矩值和返回值
        self.current_moments = moments
        self.current_return_moment = return_moment

        if self.publish_worker and self.is_publishing:
            self.publish_worker.update_moments_with_timestamp(
                moments,
                self.body_weight,
                timestamp_sensor  # 传递传感器时间戳
            )

        if timestamp_sensor > 0 and timestamp_back > 0:
            # 如果是第一次收到返回值，记录日志
            if not self.first_return_received:
                self.first_return_received = True
                self.add_log("INFO", "开始接收返回值，时延统计已启动")

            self.current_latency = (timestamp_sensor - timestamp_back) * 10  # 转换为毫秒

            # 更新统计数据
            self.latency_sum += self.current_latency
            self.latency_count += 1
            self.avg_latency = self.latency_sum / self.latency_count

            # 更新最大最小值
            self.max_latency = max(self.max_latency, self.current_latency)
            self.min_latency = min(self.min_latency, self.current_latency)

        # 更新运行时间显示
        if self.runtime_label:
            self.runtime_label.setText(f"运行时间: {relative_time:.1f} s")

        # 更新界面
        self._plot_dirty = True


    def on_publish_rate_update(self, rate: float):
        """更新发布频率显示"""
        self.publish_rate_label.setText(f"发送帧率: {rate:.1f} Hz")

    def _on_plot_timer(self):
        # 仅在有新数据时才刷新；并且仅推理进行中才画，避免空转
        if self._plot_dirty and self.is_inferencing:
            self.update_plot()
            self._plot_dirty = False

    def update_plot(self):
        """更新绘图 - 显示发送值和返回值"""
        if not self.current_joint or not self.time_buffer:
            return
        if self.current_joint not in self.moment_buffers:
            return

        # 取出原始缓冲
        times = list(self.time_buffer)
        values = list(self.moment_buffers[self.current_joint])
        return_values = list(self.return_moment_buffers[self.current_joint])

        # ——关键修改：按共同长度对齐末尾，避免长度不等直接 return——
        n = min(len(times), len(values), len(return_values))
        if n < 2:
            return
        times = times[-n:]
        values = values[-n:]
        return_values = return_values[-n:]

        # 发送值（推理值 × 体重 × 0.2）
        send_values = [v * self.body_weight * 0.2 for v in values]
        self.moment_curve.setData(times, send_values)
        self.return_moment_curve.setData(times, return_values)

        # X 轴（显示最近 10 秒）
        current_time = times[-1]
        window_size = 10.0
        self.plot_widget.setXRange(max(0, current_time - window_size), current_time + 0.5)

        # Y 轴（带简单异常值过滤）
        all_values = send_values + return_values
        valid_values = [v for v in all_values if abs(v) < 1000]
        if valid_values:
            y_min = min(valid_values);
            y_max = max(valid_values)
            pad = 0.1 * max(abs(y_min), abs(y_max))
            if (y_max - y_min) < 1.0:
                mid = 0.5 * (y_max + y_min)
                y_min, y_max = mid - 0.5, mid + 0.5
            else:
                y_min, y_max = y_min - pad, y_max + pad
            self.plot_widget.setYRange(y_min, y_max)

        #更新参数显示
        self.update_current_values()
        # 更新时延显示
        self.update_latency_display()
        # 更新缓冲区进度
        buffer_usage = len(self.time_buffer) * 100 // self.plot_buffer_size
        self.buffer_progress.setValue(buffer_usage)



    def update_current_values(self):
        """更新当前值显示 - 显示发送值和返回值"""
        if self.current_joint and self.current_joint in self.current_moments:
            value = self.current_moments[self.current_joint]
            send_value = value * self.body_weight * 0.2
            self.current_value_label.setText(f"发送值: {send_value:.3f} Nm")

            # 显示返回值
            self.return_value_label.setText(f"返回值: {self.current_return_moment[self.current_joint]:.3f} Nm")

            # 更新峰值（发送值的峰值）
            if self.current_joint in self.moment_buffers:
                values = list(self.moment_buffers[self.current_joint])
                if values:
                    send_values = [v * self.body_weight * 0.2 for v in values]
                    peak = max(abs(min(send_values)), abs(max(send_values)))
                    self.peak_value_label.setText(f"峰值: {peak:.3f} Nm")

    def on_status_update(self, stats: dict):
        """更新性能状态"""
        self.inference_speed_label.setText(f"推理速度: {stats.get('avg_inference_time', 0):.2f} ms/帧")
        self.receive_rate_label.setText(f"接收帧率: {stats.get('receive_rate', 0):.1f} Hz")
        self.frame_count_label.setText(f"已处理帧数: {stats.get('frame_count', 0)}")

    def update_latency_display(self):
        """更新时延显示"""
        # 当前时延
        self.latency_label.setText(f"当前时延: {self.current_latency:.2f} ms")

        # 平均时延
        self.avg_latency_label.setText(f"平均时延: {self.avg_latency:.2f} ms")

        # 时延范围
        if self.min_latency != float('inf'):
            self.latency_range_label.setText(f"时延范围: {self.min_latency:.2f} ~ {self.max_latency:.2f} ms")

        # 根据时延大小改变颜色提示
        if self.current_latency < 10:
            self.latency_label.setStyleSheet("color: green; font-weight: bold;")
        elif self.current_latency < 50:
            self.latency_label.setStyleSheet("color: blue; font-weight: bold;")
        elif self.current_latency < 100:
            self.latency_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.latency_label.setStyleSheet("color: red; font-weight: bold;")

    def on_error(self, error_msg: str):
        """处理错误"""
        self.add_log("ERROR", error_msg)

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止推理线程
        if self.inference_worker:
            self.inference_worker.stop()
            self.inference_worker.wait()

        # 停止发布线程
        if self.publish_worker:
            self.publish_worker.stop()
            self.publish_worker.wait()

        rospy.signal_shutdown("UI closed")
        event.accept()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ROS Inference UI')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/val/kneeonly_config.py',
        help='Path to model config file'
    )
    parser.add_argument(
        '--side',
        type=str,
        choices=['r', 'l'],
        default='l',
        help='Leg side: r (right) or l (left)'
    )

    args = parser.parse_args(rospy.myargv()[1:])

    # 创建Qt应用
    app = QApplication(sys.argv)

    # 创建并显示主窗口
    window = ROSInferenceUI(args.config, args.side)
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
