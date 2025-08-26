"""
设备配置对话框，用于选择和配置自定义数据源
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os


class DeviceConfigDialog(QDialog):
    """设备配置对话框"""

    # 信号定义
    config_confirmed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("设备数据源配置")
        self.setModal(True)
        self.setMinimumWidth(600)

        layout = QVBoxLayout()

        # 数据源类型选择
        source_group = QGroupBox("数据源类型")
        source_layout = QVBoxLayout()

        self.offline_radio = QRadioButton("离线CSV文件")
        self.offline_radio.setChecked(True)
        self.offline_radio.toggled.connect(self.on_source_type_changed)
        source_layout.addWidget(self.offline_radio)

        self.ros_radio = QRadioButton("ROS话题 (实时)")
        self.ros_radio.toggled.connect(self.on_source_type_changed)
        source_layout.addWidget(self.ros_radio)

        self.device_radio = QRadioButton("实际设备 (预留)")
        self.device_radio.setEnabled(False)
        source_layout.addWidget(self.device_radio)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # CSV文件配置
        self.csv_group = QGroupBox("CSV文件配置")
        csv_layout = QGridLayout()

        # 文件选择
        csv_layout.addWidget(QLabel("CSV文件:"), 0, 0)
        self.file_path_edit = QLineEdit()
        csv_layout.addWidget(self.file_path_edit, 0, 1)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_file)
        csv_layout.addWidget(self.browse_btn, 0, 2)

        # 标签过滤
        csv_layout.addWidget(QLabel("标签过滤:"), 1, 0)
        self.label_combo = QComboBox()
        self.label_combo.addItem("所有标签", None)
        for i in range(10):  # 假设标签范围0-9
            self.label_combo.addItem(f"标签 {i}", i)
        csv_layout.addWidget(self.label_combo, 1, 1, 1, 2)

        # 腿部选择
        csv_layout.addWidget(QLabel("腿部侧面:"), 2, 0)
        self.side_combo = QComboBox()
        self.side_combo.addItem("右腿", 'r')
        self.side_combo.addItem("左腿", 'l')
        csv_layout.addWidget(self.side_combo, 2, 1, 1, 2)

        self.csv_group.setLayout(csv_layout)
        layout.addWidget(self.csv_group)

        # ROS配置
        self.ros_group = QGroupBox("ROS配置")
        ros_layout = QGridLayout()

        ros_layout.addWidget(QLabel("话题名称:"), 0, 0)
        self.topic_edit = QLineEdit("/exo_sensor_data")
        ros_layout.addWidget(self.topic_edit, 0, 1)

        ros_layout.addWidget(QLabel("注意: ROS功能尚未实现"), 1, 0, 1, 2)

        self.ros_group.setLayout(ros_layout)
        self.ros_group.setVisible(False)
        layout.addWidget(self.ros_group)

        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()

        self.show_raw_data_check = QCheckBox("显示原始数据")
        self.show_raw_data_check.setChecked(False)
        display_layout.addWidget(self.show_raw_data_check)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept_config)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def on_source_type_changed(self):
        """数据源类型改变"""
        if self.offline_radio.isChecked():
            self.csv_group.setVisible(True)
            self.ros_group.setVisible(False)
        elif self.ros_radio.isChecked():
            self.csv_group.setVisible(False)
            self.ros_group.setVisible(True)

    def browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择CSV文件",
            "",
            "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def accept_config(self):
        """确认配置"""
        # 收集配置信息
        self.config = {
            'source_type': 'offline' if self.offline_radio.isChecked() else 'ros',
            'show_raw_data': self.show_raw_data_check.isChecked(),
            'preprocessing': {
                'coordinate_transform': True  # 始终开启坐标转换
            }
        }

        if self.offline_radio.isChecked():
            # CSV配置
            csv_path = self.file_path_edit.text()
            if not csv_path:
                QMessageBox.warning(self, "警告", "请选择CSV文件")
                return

            if not os.path.exists(csv_path):
                QMessageBox.warning(self, "警告", "文件不存在")
                return

            self.config['csv_path'] = csv_path
            self.config['label_filter'] = self.label_combo.currentData()
            self.config['side'] = self.side_combo.currentData()

        else:
            # ROS配置
            self.config['topic'] = self.topic_edit.text()

        # 发送配置信号
        self.config_confirmed.emit(self.config)
        self.accept()


class DebugPanel(QWidget):
    """调试面板，显示原始数据和预处理信息"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)

        # 原始数据显示
        raw_group = QGroupBox("原始数据")
        raw_layout = QVBoxLayout()
        self.raw_data_table = QTableWidget()
        self.raw_data_table.setColumnCount(2)
        self.raw_data_table.setHorizontalHeaderLabels(["传感器", "值"])
        raw_layout.addWidget(self.raw_data_table)
        raw_group.setLayout(raw_layout)
        splitter.addWidget(raw_group)

        # 处理后数据显示
        processed_group = QGroupBox("处理后数据")
        processed_layout = QVBoxLayout()
        self.processed_data_table = QTableWidget()
        self.processed_data_table.setColumnCount(2)
        self.processed_data_table.setHorizontalHeaderLabels(["传感器", "值"])
        processed_layout.addWidget(self.processed_data_table)
        processed_group.setLayout(processed_layout)
        splitter.addWidget(processed_group)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def update_data(self, raw_data: dict, processed_data: dict):
        """更新显示数据"""
        # 更新原始数据表
        if raw_data:
            self.raw_data_table.setRowCount(len(raw_data))
            for i, (key, value) in enumerate(raw_data.items()):
                self.raw_data_table.setItem(i, 0, QTableWidgetItem(key))
                self.raw_data_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

        # 更新处理后数据表（只显示实际使用的传感器）
        if processed_data:
            # 筛选出非零或重要的传感器数据
            filtered_data = {k: v for k, v in processed_data.items()
                           if abs(v) > 0.0001 or 'angle' in k or 'velocity' in k}

            self.processed_data_table.setRowCount(len(filtered_data))
            for i, (key, value) in enumerate(filtered_data.items()):
                self.processed_data_table.setItem(i, 0, QTableWidgetItem(key))
                self.processed_data_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))