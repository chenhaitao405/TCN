from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import numpy as np
from scipy import interpolate
from collections import deque, defaultdict
import sys
import os
from typing import Dict, Optional, List, Tuple
from devices.device_config_dialog import DeviceConfigDialog, DebugPanel


class InferenceWorker(QThread):
    """推理工作线程（修改版）"""
    data_ready = pyqtSignal(dict, dict, float)  # sensor_data, moments, timestamp
    raw_data_ready = pyqtSignal(dict, dict, float)  # processed_data, raw_data, timestamp
    status_update = pyqtSignal(str, dict)
    finished = pyqtSignal()

    def __init__(self, inference_engine, data_stream, use_raw_display=False):
        super().__init__()
        self.inference_engine = inference_engine
        self.data_stream = data_stream
        self.is_running = False
        self.use_raw_display = use_raw_display  # 是否使用原始数据显示

    def run(self):
        """线程主循环（修改版）"""
        self.is_running = True

        if self.use_raw_display and hasattr(self.data_stream, 'stream_frames_with_debug'):
            # 原始数据显示模式
            for data in self.data_stream.stream_frames_with_debug():
                if not self.is_running:
                    break

                if len(data) == 5:  # 带原始数据信息
                    sensor_data, ground_truth, timestamp, raw_data, _ = data

                    if sensor_data is None:
                        continue

                    # 执行推理
                    moments = self.inference_engine.process_frame(sensor_data)
                    #TODO:调用ROS，发布moments

                    # 发送原始数据信号
                    self.raw_data_ready.emit(
                        sensor_data,  # 处理后的数据
                        raw_data if raw_data else {},  # 原始数据
                        timestamp
                    )

                    # 简单地合并数据
                    combined_data = moments.copy() if moments else {}
                    if ground_truth:
                        for key, value in ground_truth.items():
                            combined_data[f"{key}_truth"] = value

                    # 发送常规数据信号
                    self.data_ready.emit(sensor_data, combined_data, timestamp)

        else:
            # 常规模式（原始代码）
            for sensor_data, ground_truth, timestamp in self.data_stream.stream_frames():
                if not self.is_running:
                    break

                # 执行推理
                moments = self.inference_engine.process_frame(sensor_data)

                # 简单地合并数据
                combined_data = moments.copy() if moments else {}
                if ground_truth:
                    for key, value in ground_truth.items():
                        combined_data[f"{key}_truth"] = value

                # 发送数据信号
                self.data_ready.emit(sensor_data, combined_data, timestamp)

                # 定期发送状态更新
                if int(timestamp * 10) % 10 == 0:
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

        # 创建力矩缓冲区和对应的时间缓冲区
        self.moment_buffers = {}
        self.moment_time_buffers = {}

        for name in self.inference_engine.label_names:
            self.moment_buffers[name] = {
                'predicted': deque(maxlen=self.plot_buffer_size),
                'ground_truth': deque(maxlen=self.plot_buffer_size)
            }
            self.moment_time_buffers[name] = {
                'predicted': deque(maxlen=self.plot_buffer_size),
                'ground_truth': deque(maxlen=self.plot_buffer_size)
            }

        # 当前选择的传感器和关节
        self.current_sensor = self.inference_engine.input_names[0] if self.inference_engine.input_names else None
        self.current_joints = self.inference_engine.label_names[:2] if self.inference_engine.label_names else []

        # 播放状态
        self.is_playing = False

        # 添加：R²分数标签字典
        self.r2_labels = {}

        # 添加自定义设备相关属性
        self.custom_device_config = None
        self.debug_panel = None
        self.use_custom_device = False
        self.show_raw_panel = False

        # 组织试验数据
        self.organize_trial_data()

        self.init_ui()
        self.setup_connections()

    def organize_trial_data(self):
        """组织试验数据为受试者-动作结构"""
        self.participants_actions = defaultdict(list)
        self.trial_indices = {}  # 存储 (participant, action) -> trial_index 的映射

        trial_list = self.data_stream.dataset_source.get_trial_list()
        for idx, trial_name in enumerate(trial_list):
            # 解析试验名称
            if '/' in trial_name:
                participant = trial_name.split('/')[0]
                action = trial_name.split('/')[-1]
            elif '\\' in trial_name:
                participant = trial_name.split('\\')[0]
                action = trial_name.split('\\')[-1]
            else:
                participant = "Unknown"
                action = trial_name

            self.participants_actions[participant].append(action)
            self.trial_indices[(participant, action)] = idx

        # 排序
        self.participants = sorted(self.participants_actions.keys())

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
        """创建控制面板（修改版）"""
        panel = QGroupBox("控制面板")
        layout = QHBoxLayout()

        # 数据源选择
        layout.addWidget(QLabel("数据源:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["数据集模拟", "实时设备"])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        layout.addWidget(self.source_combo)

        # 添加配置按钮
        self.config_device_btn = QPushButton("⚙ 配置设备")
        self.config_device_btn.setVisible(False)
        self.config_device_btn.clicked.connect(self.on_config_device_clicked)
        layout.addWidget(self.config_device_btn)

        # 受试者选择（在实时设备模式下隐藏）
        layout.addWidget(QLabel("受试者:"))
        self.participant_combo = QComboBox()
        self.participant_combo.addItems(self.participants)
        self.participant_combo.currentTextChanged.connect(self.on_participant_changed)
        layout.addWidget(self.participant_combo)

        # 动作选择（在实时设备模式下隐藏）
        layout.addWidget(QLabel("动作:"))
        self.action_combo = QComboBox()
        if self.participants:
            first_participant = self.participants[0]
            self.action_combo.addItems(sorted(self.participants_actions[first_participant]))
        self.action_combo.currentTextChanged.connect(self.on_action_changed)
        layout.addWidget(self.action_combo)

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

        # 禁用右键菜单
        self.sensor_plot.setMenuEnabled(False)
        # 设置初始交互模式
        self.sensor_plot.setMouseEnabled(x=False, y=False)

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

        # 禁用右键菜单
        self.joint1_plot.setMenuEnabled(False)
        # 设置初始交互模式
        self.joint1_plot.setMouseEnabled(x=False, y=False)

        self.joint1_pred_curve = self.joint1_plot.plot(pen=pg.mkPen('g', width=2), name="推理值")
        self.joint1_truth_curve = self.joint1_plot.plot(pen=pg.mkPen('r', width=1, style=Qt.DashLine), name="真实值")
        joint1_layout.addWidget(self.joint1_plot)

        # 添加：R²分数显示标签
        self.joint1_r2_label = QLabel("R² Score: --")
        self.joint1_r2_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        joint1_layout.addWidget(self.joint1_r2_label)
        self.r2_labels['joint1'] = self.joint1_r2_label

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
        elif len(self.inference_engine.label_names) > 0:
            self.joint2_combo.setCurrentIndex(0)
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

        # 禁用右键菜单
        self.joint2_plot.setMenuEnabled(False)
        # 设置初始交互模式
        self.joint2_plot.setMouseEnabled(x=False, y=False)

        self.joint2_pred_curve = self.joint2_plot.plot(pen=pg.mkPen('g', width=2), name="推理值")
        self.joint2_truth_curve = self.joint2_plot.plot(pen=pg.mkPen('r', width=1, style=Qt.DashLine), name="真实值")
        joint2_layout.addWidget(self.joint2_plot)

        # 添加：R²分数显示标签
        self.joint2_r2_label = QLabel("R² Score: --")
        self.joint2_r2_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        joint2_layout.addWidget(self.joint2_r2_label)
        self.r2_labels['joint2'] = self.joint2_r2_label

        self.joint2_group.setLayout(joint2_layout)
        layout.addWidget(self.joint2_group, 0, 2)

        widget.setLayout(layout)
        return widget

    def calculate_r2_score_aligned(self, pred_times, pred_values, truth_times, truth_values):
        """计算考虑时间对齐的R²分数"""
        if len(pred_values) < 2 or len(truth_values) < 2:
            return None

        # 转换为numpy数组
        pred_times = np.array(list(pred_times))
        pred_values = np.array(list(pred_values))
        truth_times = np.array(list(truth_times))
        truth_values = np.array(list(truth_values))

        # 找出重叠的时间范围
        min_time = max(pred_times[0], truth_times[0])
        max_time = min(pred_times[-1], truth_times[-1])

        if min_time >= max_time:
            return None

        # 在重叠时间范围内进行插值对齐
        # 使用真实值的时间点作为参考
        overlap_truth_mask = (truth_times >= min_time) & (truth_times <= max_time)
        overlap_truth_times = truth_times[overlap_truth_mask]
        overlap_truth_values = truth_values[overlap_truth_mask]

        if len(overlap_truth_times) < 2:
            return None

        # 对预测值进行线性插值到真实值的时间点
        from scipy import interpolate
        f = interpolate.interp1d(pred_times, pred_values, kind='linear', fill_value='extrapolate')
        aligned_pred_values = f(overlap_truth_times)

        # 计算R²
        ss_res = np.sum((overlap_truth_values - aligned_pred_values) ** 2)
        ss_tot = np.sum((overlap_truth_values - np.mean(overlap_truth_values)) ** 2)

        if ss_tot == 0:
            return None

        r2 = 1 - (ss_res / ss_tot)
        return r2

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

    def setup_connections(self):
        """设置信号连接"""
        # 初始选择第一个试验
        if self.participants and self.action_combo.count() > 0:
            self.on_action_changed(self.action_combo.currentText())

    def set_plot_interaction(self, enable_pan: bool):
        """设置图表交互模式
        Args:
            enable_pan: True可用拖动（暂停时），False禁用拖动（播放时）
        """
        # 播放时：禁用x轴拖动，但保留滚轮缩放
        # 暂停时：可用x轴拖动和滚轮缩放
        for plot in [self.sensor_plot, self.joint1_plot, self.joint2_plot]:
            plot.setMouseEnabled(x=enable_pan, y=False)
            # 设置鼠标模式：PanMode表示左键拖拽，RectMode表示框选
            viewbox = plot.getViewBox()
            if enable_pan:
                viewbox.setMouseMode(pg.ViewBox.PanMode)  # 左键拖拽模式
            else:
                viewbox.setMouseMode(pg.ViewBox.PanMode)  # 播放时也保持拖拽模式，但被禁用

    def auto_range_plots(self):
        """自动调整图表范围，使最新数据在右侧"""
        if self.time_buffer:
            current_time = self.time_buffer[-1]
            # 显示最近5秒的数据
            x_min = max(0, current_time - 5.0)
            x_max = current_time + 0.1  # 稍微留点余地

            # 设置所有图表的X轴范围
            self.sensor_plot.setXRange(x_min, x_max, padding=0)
            self.joint1_plot.setXRange(x_min, x_max, padding=0)
            self.joint2_plot.setXRange(x_min, x_max, padding=0)

    def on_source_changed(self, index):
        """数据源切换（修改版）"""
        if index == 1:  # 实时设备
            # 显示配置按钮
            self.config_device_btn.setVisible(True)

            # 如果还没有配置，提示用户配置
            if not self.custom_device_config:
                reply = QMessageBox.question(
                    self,
                    "配置设备",
                    "需要配置设备数据源。是否现在配置？",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.on_config_device_clicked()
                else:
                    # 切换回数据集模式
                    self.source_combo.setCurrentIndex(0)

        else:  # 数据集模拟
            use_device = False
            self.data_stream.set_source(use_device)
            self.participant_combo.setEnabled(True)
            self.action_combo.setEnabled(True)
            self.participant_combo.setVisible(True)
            self.action_combo.setVisible(True)
            self.config_device_btn.setVisible(False)
            self.use_custom_device = False

            # 隐藏原始数据面板
            if self.debug_panel:
                for dock in self.findChildren(QDockWidget):
                    if dock.widget() == self.debug_panel:
                        dock.setVisible(False)

            self.status_bar.showMessage("已切换到数据集模式")

    def on_participant_changed(self, participant: str):
        """受试者选择改变"""
        if participant in self.participants_actions:
            # 更新动作列表
            self.action_combo.blockSignals(True)  # 临时阻止信号
            self.action_combo.clear()
            self.action_combo.addItems(sorted(self.participants_actions[participant]))
            self.action_combo.blockSignals(False)

            # 自动选择第一个动作
            if self.action_combo.count() > 0:
                self.on_action_changed(self.action_combo.currentText())

    def on_action_changed(self, action: str):
        """动作选择改变"""
        participant = self.participant_combo.currentText()
        if participant and action:
            # 如果正在播放，先停止
            if self.worker is not None and self.is_playing:
                self.worker.stop()
                self.worker.wait()
                self.worker = None
                self.is_playing = False
                self.start_btn.setEnabled(True)
                self.pause_btn.setEnabled(False)
                self.pause_btn.setText("⏸ 暂停")

            # 获取对应的试验索引
            key = (participant, action)
            if key in self.trial_indices:
                trial_idx = self.trial_indices[key]
                self.data_stream.dataset_source.select_trial(trial_idx)
                self.status_bar.showMessage(f"已选择: {participant}/{action}")

    def on_sensor_changed(self, sensor_name):
        """传感器选择改变"""
        self.current_sensor = sensor_name

    def on_joint_changed(self, joint_num):
        """关节选择改变"""
        if joint_num == 1 and self.joint1_combo.currentText():
            if len(self.current_joints) > 0:
                self.current_joints[0] = self.joint1_combo.currentText()
            else:
                self.current_joints.append(self.joint1_combo.currentText())
        elif joint_num == 2 and self.joint2_combo.currentText():
            if len(self.current_joints) > 1:
                self.current_joints[1] = self.joint2_combo.currentText()
            elif len(self.current_joints) == 1:
                self.current_joints.append(self.joint2_combo.currentText())

    def on_speed_changed(self, speed_text):
        """播放速度改变"""
        speed = float(speed_text[:-1])  # 移除'x'
        self.data_stream.set_playback_speed(speed)

    def on_start_clicked(self):
        """开始按钮点击（修改版）"""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()

        # 清空缓存
        self.clear_buffers()

        # 确保数据流不在暂停状态
        self.data_stream.is_paused = False
        self.data_stream.is_playing = True

        # 设置播放状态
        self.is_playing = True
        self.set_plot_interaction(False)

        # 创建并启动工作线程
        use_raw_display = self.use_custom_device and self.show_raw_panel
        self.worker = InferenceWorker(self.inference_engine, self.data_stream, use_raw_display)
        self.worker.data_ready.connect(self.update_plots)

        # 连接原始数据信号
        if use_raw_display:
            self.worker.raw_data_ready.connect(self.update_raw_info)

        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.start()

        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 暂停")
        self.status_bar.showMessage("推理进行中...")

    def update_raw_info(self, processed_data: dict, raw_data: dict, timestamp: float):
        """更新原始数据信息"""
        if self.debug_panel:
            self.debug_panel.update_data(raw_data, processed_data)

    def on_pause_clicked(self):
        """暂停按钮点击"""
        if self.data_stream.is_paused:
            # 恢复播放
            self.data_stream.resume()
            self.pause_btn.setText("⏸ 暂停")
            self.status_bar.showMessage("已恢复")
            self.is_playing = True
            self.set_plot_interaction(False)  # 恢复播放时禁用拖动
            self.auto_range_plots()  # 自动调整到最新位置
        else:
            # 暂停
            self.data_stream.pause()
            self.pause_btn.setText("▶ 继续")
            self.status_bar.showMessage("已暂停")
            self.is_playing = False
            self.set_plot_interaction(True)  # 暂停时可用拖动

    def reset_plot_ranges(self):
        """重置所有图表的范围和缩放"""
        # 重置X轴范围到默认值
        for plot in [self.sensor_plot, self.joint1_plot, self.joint2_plot]:
            plot.setXRange(0, 5, padding=0)  # 显示0-5秒
            plot.enableAutoRange(axis='y')  # Y轴自动范围

        # 清除所有曲线数据
        self.sensor_curve.setData([], [])
        self.joint1_pred_curve.setData([], [])
        self.joint1_truth_curve.setData([], [])
        self.joint2_pred_curve.setData([], [])
        self.joint2_truth_curve.setData([], [])

        # 重置R²标签
        self.joint1_r2_label.setText("R² Score: --")
        self.joint2_r2_label.setText("R² Score: --")
        self.joint1_r2_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        self.joint2_r2_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)

    def on_reset_clicked(self):
        """重置按钮点击"""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        # 重置数据流状态（包括暂停状态）
        self.data_stream.dataset_source.reset()
        self.data_stream.is_paused = False  # 添加：重置暂停状态
        self.data_stream.is_playing = False  # 添加：重置播放状态

        self.inference_engine.reset_buffer()
        self.clear_buffers()

        # 重置图表范围和缩放
        self.reset_plot_ranges()

        # 重置播放状态
        self.is_playing = False
        self.set_plot_interaction(True)  # 重置后可用拖动

        # 重置按钮状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")  # 确保文本正确
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("已重置")

    def clear_buffers(self):
        """清空所有数据缓存"""
        # 清空主时间缓冲区
        self.time_buffer.clear()

        # 清空传感器缓冲区
        for buffer in self.sensor_buffers.values():
            buffer.clear()

        # 清空力矩数据缓冲区
        for joint_buffer in self.moment_buffers.values():
            joint_buffer['predicted'].clear()
            joint_buffer['ground_truth'].clear()

        # 清空力矩时间缓冲区
        for time_buffer in self.moment_time_buffers.values():
            time_buffer['predicted'].clear()
            time_buffer['ground_truth'].clear()

    def update_plots(self, sensor_data: Dict, moments: Dict, timestamp: float):
        """更新图表"""
        # 更新时间缓存
        self.time_buffer.append(timestamp)

        # 更新传感器数据缓存
        for name, value in sensor_data.items():
            if name in self.sensor_buffers:
                self.sensor_buffers[name].append(value)

        # 更新力矩数据缓存（考虑延迟）
        for i, name in enumerate(self.inference_engine.label_names):
            delay = self.inference_engine.model_delays[i] if i < len(self.inference_engine.model_delays) else 0
            delay_time = delay / 200.0  # 转换为秒（假设200Hz采样率）

            # 预测值：模型在当前时刻预测的是(当前-延迟)时刻的值
            if name in moments:
                self.moment_buffers[name]['predicted'].append(moments[name])
                self.moment_time_buffers[name]['predicted'].append(timestamp - delay_time)

            # 真实值：当前时刻的真实值
            if f"{name}_truth" in moments:
                self.moment_buffers[name]['ground_truth'].append(moments[f"{name}_truth"])
                self.moment_time_buffers[name]['ground_truth'].append(timestamp)

        # 更新传感器图表
        if self.current_sensor and self.current_sensor in self.sensor_buffers:
            sensor_data_list = list(self.sensor_buffers[self.current_sensor])
            time_list = list(self.time_buffer)
            if sensor_data_list and len(time_list) == len(sensor_data_list):
                self.sensor_curve.setData(time_list, sensor_data_list)

            # 更新关节力矩图表1
        if len(self.current_joints) > 0 and self.current_joints[0] in self.moment_buffers:
            joint_name = self.current_joints[0]
            joint_data = self.moment_buffers[joint_name]
            time_data = self.moment_time_buffers[joint_name]

            if joint_data['predicted'] and time_data['predicted']:
                self.joint1_pred_curve.setData(
                    list(time_data['predicted']),
                    list(joint_data['predicted'])
                )

            if joint_data['ground_truth'] and time_data['ground_truth']:
                self.joint1_truth_curve.setData(
                    list(time_data['ground_truth']),
                    list(joint_data['ground_truth'])
                )

            # 计算并更新R²分数（使用时间对齐的版本）
            if len(joint_data['predicted']) > 10 and len(joint_data['ground_truth']) > 10:  # 确保有足够数据
                r2 = self.calculate_r2_score_aligned(
                    time_data['predicted'], joint_data['predicted'],
                    time_data['ground_truth'], joint_data['ground_truth']
                )
                if r2 is not None:
                    self.joint1_r2_label.setText(f"R² Score: {r2:.4f}")
                    # 根据R²值设置颜色
                    if r2 >= 0.9:
                        color = "#4CAF50"  # 绿色
                    elif r2 >= 0.7:
                        color = "#FFA726"  # 橙色
                    else:
                        color = "#EF5350"  # 红色
                    self.joint1_r2_label.setStyleSheet(f"""
                         QLabel {{
                             font-size: 14px;
                             font-weight: bold;
                             padding: 5px;
                             background-color: {color};
                             color: white;
                             border-radius: 3px;
                         }}
                     """)

            # 更新关节力矩图表2
        if len(self.current_joints) > 1 and self.current_joints[1] in self.moment_buffers:
            joint_name = self.current_joints[1]
            joint_data = self.moment_buffers[joint_name]
            time_data = self.moment_time_buffers[joint_name]

            if joint_data['predicted'] and time_data['predicted']:
                self.joint2_pred_curve.setData(
                    list(time_data['predicted']),
                    list(joint_data['predicted'])
                )

            if joint_data['ground_truth'] and time_data['ground_truth']:
                self.joint2_truth_curve.setData(
                    list(time_data['ground_truth']),
                    list(joint_data['ground_truth'])
                )

            # 计算并更新R²分数（使用时间对齐的版本）
            if len(joint_data['predicted']) > 10 and len(joint_data['ground_truth']) > 10:
                r2 = self.calculate_r2_score_aligned(
                    time_data['predicted'], joint_data['predicted'],
                    time_data['ground_truth'], joint_data['ground_truth']
                )
                if r2 is not None:
                    self.joint2_r2_label.setText(f"R² Score: {r2:.4f}")
                    # 根据R²值设置颜色
                    if r2 >= 0.9:
                        color = "#4CAF50"  # 绿色
                    elif r2 >= 0.7:
                        color = "#FFA726"  # 橙色
                    else:
                        color = "#EF5350"  # 红色
                    self.joint2_r2_label.setStyleSheet(f"""
                         QLabel {{
                             font-size: 14px;
                             font-weight: bold;
                             padding: 5px;
                             background-color: {color};
                             color: white;
                             border-radius: 3px;
                         }}
                     """)

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
        self.pause_btn.setText("⏸ 暂停")  # 添加：重置按钮文本
        self.is_playing = False
        self.data_stream.is_paused = False  # 添加：重置暂停状态
        self.data_stream.is_playing = False  # 添加：重置播放状态
        self.set_plot_interaction(True)  # 完成后可用拖动
        self.status_bar.showMessage("推理完成")

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
        event.accept()

    def on_config_device_clicked(self):
        """配置设备按钮点击"""
        dialog = DeviceConfigDialog(self)
        dialog.config_confirmed.connect(self.on_device_config_confirmed)
        dialog.exec_()

    def on_device_config_confirmed(self, config: dict):
        """设备配置确认"""
        self.custom_device_config = config

        # 配置数据流管理器
        success = self.data_stream.configure_custom_device(config)

        if success:
            self.use_custom_device = True
            self.show_raw_panel = config.get('show_raw_data', False)

            # 设置数据源
            self.data_stream.set_source(True, True)  # use_device=True, use_custom=True

            # 显示或隐藏原始数据面板
            if self.show_raw_panel and self.debug_panel is None:
                self.show_raw_panel_ui()

            self.status_bar.showMessage(f"已加载自定义数据: {config.get('csv_path', 'Unknown')}")

            # 更新UI状态
            self.update_ui_for_custom_device()

            # 可用开始按钮
            self.start_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "错误", "配置设备失败")
            self.use_custom_device = False

    def show_raw_panel_ui(self):
        """显示原始数据面板"""
        if self.debug_panel is None:
            self.debug_panel = DebugPanel()

            # 创建一个可停靠的窗口
            dock = QDockWidget("原始数据面板", self)
            dock.setWidget(self.debug_panel)
            self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def update_ui_for_custom_device(self):
        """更新UI以适应自定义设备模式"""
        # 隐藏数据集相关控件
        self.participant_combo.setVisible(False)
        self.action_combo.setVisible(False)

        # 显示设备状态
        status = self.data_stream.get_custom_device_status()
        if status:
            info_text = f"模式: {status.get('mode', 'Unknown')}, "
            info_text += f"侧面: {status.get('side', 'Unknown')}, "
            info_text += f"帧数: {status.get('frame_count', 0)}"
            self.status_bar.showMessage(info_text)