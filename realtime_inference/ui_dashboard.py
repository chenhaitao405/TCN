from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import numpy as np
from collections import deque
import sys
import os
from typing import Dict, Optional


class InferenceWorker(QThread):
    """推理工作线程"""
    data_ready = pyqtSignal(dict, dict, float)  # sensor_data, moments, timestamp
    status_update = pyqtSignal(str, dict)  # status_text, performance_stats
    finished = pyqtSignal()

    def __init__(self, inference_engine, data_stream):
        super().__init__()
        self.inference_engine = inference_engine
        self.data_stream = data_stream
        self.is_running = False

    def run(self):
        """线程主循环"""
        self.is_running = True

        for sensor_data, ground_truth, timestamp in self.data_stream.stream_frames():
            if not self.is_running:
                break

            # 执行推理
            moments = self.inference_engine.process_frame(sensor_data)

            # 合并ground truth到moments中（用于显示）
            if ground_truth:
                for key in ground_truth:
                    moments[f"{key}_truth"] = ground_truth[key]

            # 发送数据信号
            self.data_ready.emit(sensor_data, moments, timestamp)

            # 定期发送状态更新
            if int(timestamp * 10) % 10 == 0:  # 每秒更新一次
                stats = self.inference_engine.get_performance_stats()
                status_text = f"推理时间: {stats['avg_inference_time']:.2f}ms"
                self.status_update.emit(status_text, stats)

        self.finished.emit()

    def stop(self):
        """停止线程"""
        self.is_running = False
        self.data_stream.stop()


class RealtimeInferenceDashboard(QMainWindow):
    """实时推理可视化界面"""

    def __init__(self, inference_engine, data_stream):
        super().__init__()
        self.inference_engine = inference_engine
        self.data_stream = data_stream
        self.worker = None

        # 数据缓存用于绘图
        self.plot_buffer_size = 1000
        self.time_buffer = deque(maxlen=self.plot_buffer_size)
        self.sensor_buffers = {name: deque(maxlen=self.plot_buffer_size)
                               for name in self.inference_engine.input_names}
        self.moment_buffers = {}
        for name in self.inference_engine.label_names:
            self.moment_buffers[name] = {
                'predicted': deque(maxlen=self.plot_buffer_size),
                'ground_truth': deque(maxlen=self.plot_buffer_size)
            }

        # 当前选择的传感器和关节
        self.current_sensor = self.inference_engine.input_names[0] if self.inference_engine.input_names else None
        self.current_joints = self.inference_engine.label_names[:2]  # 默认显示前两个

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("外骨骼关节力矩实时推理系统")
        self.setGeometry(100, 100, 1600, 900)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 5px 15px;
                border-radius: 3px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

        # 主widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 状态栏
        self.create_status_bar()

        # 信息面板
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel)

        # 控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # 图表区域
        charts_widget = self.create_charts()
        main_layout.addWidget(charts_widget)

    def create_info_panel(self):
        """创建信息面板"""
        panel = QGroupBox("系统信息")
        layout = QHBoxLayout()

        # 模型信息
        model_info = QLabel(f"模型: {os.path.basename(self.inference_engine.config.model_path)}")
        layout.addWidget(model_info)

        # 输入维度
        input_info = QLabel(f"输入维度: {len(self.inference_engine.input_names)}")
        layout.addWidget(input_info)

        # 输出维度
        output_info = QLabel(f"输出维度: {len(self.inference_engine.label_names)}")
        layout.addWidget(output_info)

        # 设备信息
        device_info = QLabel(f"设备: {self.inference_engine.device}")
        layout.addWidget(device_info)

        # 性能指标
        self.perf_label = QLabel("推理时间: -- ms")
        layout.addWidget(self.perf_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

        panel.setLayout(layout)
        return panel

    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("控制面板")
        layout = QHBoxLayout()

        # 数据源选择
        layout.addWidget(QLabel("数据源:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["数据集模拟", "实时设备"])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        layout.addWidget(self.source_combo)

        # 试验选择
        layout.addWidget(QLabel("试验:"))
        self.trial_combo = QComboBox()
        trial_list = self.data_stream.dataset_source.get_trial_list()
        self.trial_combo.addItems(trial_list)
        self.trial_combo.currentIndexChanged.connect(self.on_trial_changed)
        layout.addWidget(self.trial_combo)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 控制按钮
        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.clicked.connect(self.on_start_clicked)
        layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.clicked.connect(self.on_pause_clicked)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn)

        self.reset_btn = QPushButton("↺ 重置")
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        layout.addWidget(self.reset_btn)

        # 播放速度
        layout.addWidget(QLabel("速度:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "2.0x"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        layout.addWidget(self.speed_combo)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_charts(self):
        """创建图表区域"""
        widget = QWidget()
        layout = QGridLayout()

        # 设置pyqtgraph全局选项
        pg.setConfigOptions(antialias=True)

        # 1. 传感器数据图表
        sensor_group = QGroupBox("传感器数据")
        sensor_layout = QVBoxLayout()

        # 传感器选择下拉框
        sensor_select_layout = QHBoxLayout()
        sensor_select_layout.addWidget(QLabel("选择传感器:"))
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(self.inference_engine.input_names)
        self.sensor_combo.currentTextChanged.connect(self.on_sensor_changed)
        sensor_select_layout.addWidget(self.sensor_combo)
        sensor_select_layout.addStretch()
        sensor_layout.addLayout(sensor_select_layout)

        # 传感器数据图
        self.sensor_plot = pg.PlotWidget()
        self.sensor_plot.setLabel('left', '数值')
        self.sensor_plot.setLabel('bottom', '时间', units='s')
        self.sensor_plot.showGrid(x=True, y=True, alpha=0.3)
        self.sensor_plot.setYRange(-10, 10)
        self.sensor_curve = self.sensor_plot.plot(pen=pg.mkPen('b', width=2))
        sensor_layout.addWidget(self.sensor_plot)

        sensor_group.setLayout(sensor_layout)
        layout.addWidget(sensor_group, 0, 0)

        # 2. 第一个关节力矩图表
        self.joint1_group = QGroupBox("关节力矩 1")
        joint1_layout = QVBoxLayout()

        # 关节选择下拉框1
        joint1_select_layout = QHBoxLayout()
        joint1_select_layout.addWidget(QLabel("选择关节:"))
        self.joint1_combo = QComboBox()
        self.joint1_combo.addItems(self.inference_engine.label_names)
        if len(self.inference_engine.label_names) > 0:
            self.joint1_combo.setCurrentIndex(0)
        self.joint1_combo.currentTextChanged.connect(lambda: self.on_joint_changed(1))
        joint1_select_layout.addWidget(self.joint1_combo)
        joint1_select_layout.addStretch()
        joint1_layout.addLayout(joint1_select_layout)

        # 力矩图1
        self.joint1_plot = pg.PlotWidget()
        self.joint1_plot.setLabel('left', '力矩', units='Nm/kg')
        self.joint1_plot.setLabel('bottom', '时间', units='s')
        self.joint1_plot.showGrid(x=True, y=True, alpha=0.3)
        self.joint1_plot.addLegend()
        self.joint1_plot.setYRange(-2, 2)
        self.joint1_pred_curve = self.joint1_plot.plot(pen=pg.mkPen('g', width=2), name="推理值")
        self.joint1_truth_curve = self.joint1_plot.plot(pen=pg.mkPen('r', width=1, style=Qt.DashLine), name="真实值")
        joint1_layout.addWidget(self.joint1_plot)

        self.joint1_group.setLayout(joint1_layout)
        layout.addWidget(self.joint1_group, 0, 1)

        # 3. 第二个关节力矩图表
        self.joint2_group = QGroupBox("关节力矩 2")
        joint2_layout = QVBoxLayout()

        # 关节选择下拉框2
        joint2_select_layout = QHBoxLayout()
        joint2_select_layout.addWidget(QLabel("选择关节:"))
        self.joint2_combo = QComboBox()
        self.joint2_combo.addItems(self.inference_engine.label_names)
        if len(self.inference_engine.label_names) > 1:
            self.joint2_combo.setCurrentIndex(1)
        self.joint2_combo.currentTextChanged.connect(lambda: self.on_joint_changed(2))
        joint2_select_layout.addWidget(self.joint2_combo)
        joint2_select_layout.addStretch()
        joint2_layout.addLayout(joint2_select_layout)

        # 力矩图2
        self.joint2_plot = pg.PlotWidget()
        self.joint2_plot.setLabel('left', '力矩', units='Nm/kg')
        self.joint2_plot.setLabel('bottom', '时间', units='s')
        self.joint2_plot.showGrid(x=True, y=True, alpha=0.3)
        self.joint2_plot.addLegend()
        self.joint2_plot.setYRange(-2, 2)
        self.joint2_pred_curve = self.joint2_plot.plot(pen=pg.mkPen('g', width=2), name="推理值")
        self.joint2_truth_curve = self.joint2_plot.plot(pen=pg.mkPen('r', width=1, style=Qt.DashLine), name="真实值")
        joint2_layout.addWidget(self.joint2_plot)

        self.joint2_group.setLayout(joint2_layout)
        layout.addWidget(self.joint2_group, 0, 2)

        widget.setLayout(layout)
        return widget

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

    def setup_connections(self):
        """设置信号连接"""
        # 初始选择第一个试验
        if self.trial_combo.count() > 0:
            self.on_trial_changed(0)

    def on_source_changed(self, index):
        """数据源切换"""
        use_device = (index == 1)
        self.data_stream.set_source(use_device)
        self.trial_combo.setEnabled(not use_device)

        if use_device:
            self.status_bar.showMessage("已切换到实时设备模式（功能待实现）")
        else:
            self.status_bar.showMessage("已切换到数据集模式")

    def on_trial_changed(self, index):
        """试验选择改变"""
        if index >= 0:
            self.data_stream.dataset_source.select_trial(index)
            trial_name = self.trial_combo.currentText()
            self.status_bar.showMessage(f"已选择试验: {trial_name}")

    def on_sensor_changed(self, sensor_name):
        """传感器选择改变"""
        self.current_sensor = sensor_name

    def on_joint_changed(self, joint_num):
        """关节选择改变"""
        if joint_num == 1:
            self.current_joints[0] = self.joint1_combo.currentText()
        elif joint_num == 2 and len(self.current_joints) > 1:
            self.current_joints[1] = self.joint2_combo.currentText()

    def on_speed_changed(self, speed_text):
        """播放速度改变"""
        speed = float(speed_text[:-1])  # 移除'x'
        self.data_stream.set_playback_speed(speed)

    def on_start_clicked(self):
        """开始按钮点击"""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()

        # 清空缓存
        self.clear_buffers()

        # 创建并启动工作线程
        self.worker = InferenceWorker(self.inference_engine, self.data_stream)
        self.worker.data_ready.connect(self.update_plots)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.start()

        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 暂停")
        self.status_bar.showMessage("推理进行中...")

    def on_pause_clicked(self):
        """暂停按钮点击"""
        if self.data_stream.is_paused:
            self.data_stream.resume()
            self.pause_btn.setText("⏸ 暂停")
            self.status_bar.showMessage("已恢复")
        else:
            self.data_stream.pause()
            self.pause_btn.setText("▶ 继续")
            self.status_bar.showMessage("已暂停")

    def on_reset_clicked(self):
        """重置按钮点击"""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        self.data_stream.dataset_source.reset()
        self.inference_engine.reset_buffer()
        self.clear_buffers()

        # 重置按钮状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("已重置")

    def clear_buffers(self):
        """清空数据缓存"""
        self.time_buffer.clear()
        for buffer in self.sensor_buffers.values():
            buffer.clear()
        for joint_buffer in self.moment_buffers.values():
            joint_buffer['predicted'].clear()
            joint_buffer['ground_truth'].clear()

    def update_plots(self, sensor_data: Dict, moments: Dict, timestamp: float):
        """更新图表"""
        # 更新时间缓存
        self.time_buffer.append(timestamp)

        # 更新传感器数据缓存
        for name, value in sensor_data.items():
            if name in self.sensor_buffers:
                self.sensor_buffers[name].append(value)

        # 更新力矩数据缓存
        for name in self.inference_engine.label_names:
            if name in moments:
                self.moment_buffers[name]['predicted'].append(moments[name])
            if f"{name}_truth" in moments:
                self.moment_buffers[name]['ground_truth'].append(moments[f"{name}_truth"])

        # 更新传感器图表
        if self.current_sensor and self.current_sensor in self.sensor_buffers:
            sensor_data = list(self.sensor_buffers[self.current_sensor])
            if sensor_data:
                self.sensor_curve.setData(list(self.time_buffer), sensor_data)

        # 更新关节力矩图表1
        if len(self.current_joints) > 0 and self.current_joints[0] in self.moment_buffers:
            joint_data = self.moment_buffers[self.current_joints[0]]
            if joint_data['predicted']:
                self.joint1_pred_curve.setData(list(self.time_buffer), list(joint_data['predicted']))
            if joint_data['ground_truth']:
                self.joint1_truth_curve.setData(list(self.time_buffer), list(joint_data['ground_truth']))

        # 更新关节力矩图表2
        if len(self.current_joints) > 1 and self.current_joints[1] in self.moment_buffers:
            joint_data = self.moment_buffers[self.current_joints[1]]
            if joint_data['predicted']:
                self.joint2_pred_curve.setData(list(self.time_buffer), list(joint_data['predicted']))
            if joint_data['ground_truth']:
                self.joint2_truth_curve.setData(list(self.time_buffer), list(joint_data['ground_truth']))

        # 更新进度条
        progress = self.data_stream.dataset_source.get_progress()
        self.progress_bar.setValue(int(progress))

    def update_status(self, status_text: str, performance_stats: Dict):
        """更新状态信息"""
        self.perf_label.setText(f"推理时间: {performance_stats['avg_inference_time']:.2f}ms")

    def on_inference_finished(self):
        """推理完成"""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.status_bar.showMessage("推理完成")

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
        event.accept()