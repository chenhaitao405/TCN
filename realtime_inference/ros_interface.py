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
from typing import Dict, Optional
import threading

# ROS imports
import rospy
from std_msgs.msg import Float64MultiArray

# PyQt5 imports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import numpy as np

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

    def __init__(self, topic: str = '/moment'):
        super().__init__()
        self.topic = topic
        self.is_running = False
        self.publisher = None

        # 共享数据和线程锁
        self.moments_lock = threading.Lock()
        self.current_moments = {}
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

    def update_moments(self, moments: dict, body_weight: float):
        """更新力矩数据（线程安全）"""
        with self.moments_lock:
            self.current_moments = moments.copy()
            self.body_weight = body_weight
            if moments:
                self.num_joints = len(moments)

    def set_num_joints(self, num: int):
        """设置关节数量"""
        with self.moments_lock:
            self.num_joints = num

    def run(self):
        """线程主循环 - 100Hz固定频率发布"""
        self.is_running = True

        # 创建发布器
        self.publisher = rospy.Publisher(
            self.topic,
            Float64MultiArray,
            queue_size=10
        )

        # 100Hz -> 10ms周期
        publish_period = 0.01  # 10ms

        while self.is_running and not rospy.is_shutdown():
            start_time = time.time()

            try:
                msg = Float64MultiArray()

                with self.moments_lock:
                    if self.current_moments:
                        # 发送值 = 推理值 × 体重 × 0.2
                        msg.data = [value * self.body_weight * 0.2
                                    for value in self.current_moments.values()]
                    else:
                        # 没有数据时发送零值
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

            except Exception as e:
                self.error_occurred.emit(f"发布错误: {str(e)}")

            # 精确控制频率
            elapsed = time.time() - start_time
            sleep_time = publish_period - elapsed
            if sleep_time > 0:
                QThread.msleep(int(sleep_time * 1000))

    def stop(self):
        """停止线程"""
        self.is_running = False
        if self.publisher:
            self.publisher.unregister()
            self.publisher = None


class InferenceWorker(QThread):
    """推理工作线程"""
    data_ready = pyqtSignal(dict, dict, float, float)  # sensor_data, moments, timestamp, return_moment
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

                    'label': int(msg.data[9]) if len(msg.data) > 9 else 0
                }

                # 保存返回值
                return_moment = msg.data[8]

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
                print(f"预处理耗时: {preprocess_time:.3f} ms")
                print(f"推理耗时: {inference_time:.3f} ms")
                print(f"总耗时: {(preprocess_time + inference_time):.3f} ms")
                print("-" * 40)  # 分隔线，便于查看

                # 获取时间戳（使用相对时间）
                current_timestamp = rospy.Time.now().to_sec()

                # 如果是第一次，记录起始时间
                if self.start_timestamp is None:
                    self.start_timestamp = current_timestamp

                # 计算相对时间（从开始推理到现在的秒数）
                relative_time = current_timestamp - self.start_timestamp

                # 发送数据信号，使用相对时间
                self.data_ready.emit(processed_data, moments, relative_time, return_moment)

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
        self.side = side

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

        # UI元素（会在init_ui中初始化）
        self.runtime_label = None

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
        pg.setConfigOptions(antialias=True)

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
            self.publish_worker.update_moments(self.current_moments, self.body_weight)
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
            self.inference_worker.data_ready.connect(self.on_data_received)
            self.inference_worker.status_update.connect(self.on_status_update)
            self.inference_worker.error_occurred.connect(self.on_error)

        # 初始化关节列表
        if not self.joint_combo.count():
            for name in self.inference_worker.engine.label_names:
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
            self.publish_worker = PublishWorker(self.pub_topic)
            self.publish_worker.status_update.connect(self.on_publish_rate_update)
            self.publish_worker.error_occurred.connect(self.on_error)

            # 设置关节数量
            if self.joint_combo.count() > 0:
                self.publish_worker.set_num_joints(self.joint_combo.count())

            # 如果有当前数据，立即更新
            if self.current_moments:
                self.publish_worker.update_moments(self.current_moments, self.body_weight)

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
        self.update_plot()

    def on_data_received(self, sensor_data: dict, moments: dict, relative_time: float, return_moment: float):
        """接收到推理数据 - 包含推理值和返回值（使用相对时间）"""
        # 更新时间缓存（relative_time已经是相对时间，单位：秒）
        self.time_buffer.append(relative_time)

        # 更新力矩缓存和返回值缓存
        for joint_name, value in moments.items():
            if joint_name in self.moment_buffers:
                self.moment_buffers[joint_name].append(value)
                # 为每个关节保存相同的返回值
                self.return_moment_buffers[joint_name].append(return_moment)

        # 保存当前力矩值和返回值
        self.current_moments = moments
        self.current_return_moment = return_moment

        # 更新发布线程的力矩数据
        if self.publish_worker and self.is_publishing:
            self.publish_worker.update_moments(moments, self.body_weight)

        # 更新界面
        self.update_plot()
        self.update_current_values()

        # 更新运行时间显示
        if self.runtime_label:
            self.runtime_label.setText(f"运行时间: {relative_time:.1f} s")

        # 更新缓冲区进度
        buffer_usage = len(self.time_buffer) * 100 // self.plot_buffer_size
        self.buffer_progress.setValue(buffer_usage)

    def on_publish_rate_update(self, rate: float):
        """更新发布频率显示"""
        self.publish_rate_label.setText(f"发送帧率: {rate:.1f} Hz")

    def update_plot(self):
        """更新绘图 - 显示发送值和返回值"""
        if not self.current_joint or not self.time_buffer:
            return

        if self.current_joint not in self.moment_buffers:
            return

        times = list(self.time_buffer)
        values = list(self.moment_buffers[self.current_joint])
        return_values = list(self.return_moment_buffers[self.current_joint])

        if len(times) != len(values) or len(times) != len(return_values):
            return

        # 发送值（推理值 × 体重 × 0.2）
        send_values = [v * self.body_weight * 0.2 for v in values]
        self.moment_curve.setData(times, send_values)

        # 返回值
        self.return_moment_curve.setData(times, return_values)

        # 自动调整X轴范围（显示最近10秒的数据）
        if times:
            current_time = times[-1]
            window_size = 10.0  # 显示窗口大小（秒）
            self.plot_widget.setXRange(max(0, current_time - window_size), current_time + 0.5)

        # 自动调整Y轴范围
        if send_values and return_values:
            all_values = send_values + return_values
            # 过滤掉可能的异常值
            valid_values = [v for v in all_values if abs(v) < 1000]
            if valid_values:
                y_min = min(valid_values) - abs(min(valid_values)) * 0.1
                y_max = max(valid_values) + abs(max(valid_values)) * 0.1
                # 确保Y轴范围不会太小
                y_range = y_max - y_min
                if y_range < 1.0:
                    y_center = (y_max + y_min) / 2
                    y_min = y_center - 0.5
                    y_max = y_center + 0.5
                self.plot_widget.setYRange(y_min, y_max)

    def update_current_values(self):
        """更新当前值显示 - 显示发送值和返回值"""
        if self.current_joint and self.current_joint in self.current_moments:
            value = self.current_moments[self.current_joint]
            send_value = value * self.body_weight * 0.2
            self.current_value_label.setText(f"发送值: {send_value:.3f} Nm")

            # 显示返回值
            self.return_value_label.setText(f"返回值: {self.current_return_moment:.3f} Nm")

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