#!/usr/bin/env python
"""
ROS通信时延测试UI
基于PyQt5的图形界面，用于控制ROS话题的接收和发送，并实时显示时延统计
"""

import sys
import rospy
from std_msgs.msg import Float64MultiArray
import time
import numpy as np
from collections import deque
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class LatencyStats(QObject):
    """时延统计信号"""
    updated = pyqtSignal(float, float, float, float, int)  # avg, min, max, std, count


class MessageStats(QObject):
    """消息统计信号"""
    updated = pyqtSignal(int, int, float, float)  # recv_count, send_count, recv_rate, send_rate


class ROSLatencyTester(QObject):
    """ROS时延测试核心类"""

    stats_updated = pyqtSignal(dict)

    def __init__(self, sub_topic='/motor12_left', pub_topic='/moment', num_joints=1):
        super().__init__()

        # 配置参数
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.num_joints = num_joints
        self.publish_rate = 100  # 100Hz

        # 发布器（延迟创建）
        self.publisher = None

        # 订阅器（延迟创建）
        self.subscriber = None

        # 共享数据
        self.data_lock = threading.Lock()
        self.current_timestamp_sensor = 0.0

        # 统计数据
        self.msg_receive_count = 0
        self.msg_publish_count = 0
        self.start_time = time.time()
        self.latencies = deque(maxlen=1000)

        # 状态标志
        self.is_receiving = False
        self.is_publishing = False

        # 发布线程
        self.publish_thread = None

        # 统计更新定时器
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(100)  # 每100ms更新一次统计

        # 频率计算
        self.last_recv_count = 0
        self.last_send_count = 0
        self.last_stats_time = time.time()

    def start_receiving(self):
        """开始接收话题"""
        if not self.is_receiving:
            self.subscriber = rospy.Subscriber(
                self.sub_topic,
                Float64MultiArray,
                self.sensor_callback,
                queue_size=1
            )
            self.is_receiving = True
            rospy.loginfo(f"开始订阅话题: {self.sub_topic}")

    def stop_receiving(self):
        """停止接收话题"""
        if self.is_receiving and self.subscriber:
            self.subscriber.unregister()
            self.subscriber = None
            self.is_receiving = False
            rospy.loginfo(f"停止订阅话题: {self.sub_topic}")

    def start_publishing(self):
        """开始发布话题"""
        if not self.is_publishing:
            # 创建发布器
            if not self.publisher:
                self.publisher = rospy.Publisher(
                    self.pub_topic,
                    Float64MultiArray,
                    queue_size=1
                )

            # 启动发布线程
            self.is_publishing = True
            self.publish_thread = threading.Thread(target=self.publish_loop)
            self.publish_thread.daemon = True
            self.publish_thread.start()
            rospy.loginfo(f"开始发布话题: {self.pub_topic} @ {self.publish_rate}Hz")

    def stop_publishing(self):
        """停止发布话题"""
        if self.is_publishing:
            self.is_publishing = False
            if self.publish_thread:
                self.publish_thread.join(timeout=1.0)
            rospy.loginfo(f"停止发布话题: {self.pub_topic}")

    def sensor_callback(self, msg):
        """处理传感器数据"""
        try:
            if len(msg.data) < 10:
                return

            # 提取timestamp_sensor
            timestamp_sensor = msg.data[9] if len(msg.data) > 9 else 0

            # 更新共享数据
            with self.data_lock:
                self.current_timestamp_sensor = timestamp_sensor

            # 计算时延
            if len(msg.data) > 10 and msg.data[10] > 0:
                latency = (timestamp_sensor - msg.data[10]) * 10  # ms  #msg.data[10] stamp返回值。stamp单位为10ms
                self.latencies.append(latency)

            self.msg_receive_count += 1

        except Exception as e:
            rospy.logerr(f"处理消息错误: {e}")

    def publish_loop(self):
        """发布线程"""
        rate = rospy.Rate(self.publish_rate)

        while self.is_publishing and not rospy.is_shutdown():
            try:
                msg = Float64MultiArray()

                with self.data_lock:
                    timestamp_to_send = self.current_timestamp_sensor

                # 格式: [moment×N, timestamp_back]
                zeros = [1.0] * self.num_joints
                msg.data = zeros + [timestamp_to_send]

                if self.publisher:
                    self.publisher.publish(msg)
                    self.msg_publish_count += 1

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"发布错误: {e}")

    def update_stats(self):
        """更新统计信息"""
        current_time = time.time()
        dt = current_time - self.last_stats_time

        if dt > 0:
            recv_rate = (self.msg_receive_count - self.last_recv_count) / dt
            send_rate = (self.msg_publish_count - self.last_send_count) / dt

            self.last_recv_count = self.msg_receive_count
            self.last_send_count = self.msg_publish_count
            self.last_stats_time = current_time

            # 计算时延统计
            latency_stats = {}
            if self.latencies:
                lat_array = np.array(self.latencies)
                latency_stats = {
                    'avg': np.mean(lat_array),
                    'min': np.min(lat_array),
                    'max': np.max(lat_array),
                    'std': np.std(lat_array),
                    'count': len(self.latencies)
                }
            else:
                latency_stats = {
                    'avg': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'std': 0.0,
                    'count': 0
                }

            # 发送统计信号
            stats = {
                'recv_count': self.msg_receive_count,
                'send_count': self.msg_publish_count,
                'recv_rate': recv_rate,
                'send_rate': send_rate,
                'latency': latency_stats,
                'is_receiving': self.is_receiving,
                'is_publishing': self.is_publishing
            }

            self.stats_updated.emit(stats)

    def reset_stats(self):
        """重置统计数据"""
        with self.data_lock:
            self.msg_receive_count = 0
            self.msg_publish_count = 0
            self.latencies.clear()
            self.start_time = time.time()
            self.last_recv_count = 0
            self.last_send_count = 0
            self.last_stats_time = time.time()


class LatencyTestUI(QMainWindow):
    """主界面"""

    def __init__(self):
        super().__init__()

        # 初始化ROS节点
        rospy.init_node('latency_tester_ui', anonymous=True)

        # 创建测试器
        self.tester = ROSLatencyTester()
        self.tester.stats_updated.connect(self.update_display)

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('ROS通信时延测试')
        self.setFixedSize(500, 450)

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
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                min-height: 35px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #357a38;
            }
            QPushButton#stopBtn {
                background-color: #f44336;
            }
            QPushButton#stopBtn:hover {
                background-color: #da190b;
            }
            QLabel {
                font-size: 13px;
            }
            QLabel#titleLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
            QLabel#valueLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2196F3;
            }
            QLabel#statusLabel {
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
        """)

        # 主Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)

        # 标题
        title = QLabel('ROS 通信时延测试系统')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 18px; font-weight: bold; color: #333; padding: 10px;')
        layout.addWidget(title)

        # 控制区域
        control_group = QGroupBox('控制面板')
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        # 接收控制
        self.recv_btn = QPushButton('开始接收')
        self.recv_btn.clicked.connect(self.toggle_receiving)
        self.recv_status = QLabel('停止')
        self.recv_status.setObjectName('statusLabel')
        self.recv_status.setAlignment(Qt.AlignCenter)
        self.update_status_label(self.recv_status, False)

        control_layout.addWidget(QLabel('接收话题:'), 0, 0)
        control_layout.addWidget(self.recv_btn, 0, 1)
        control_layout.addWidget(self.recv_status, 0, 2)

        # 发送控制
        self.send_btn = QPushButton('开始发送')
        self.send_btn.clicked.connect(self.toggle_publishing)
        self.send_status = QLabel('停止')
        self.send_status.setObjectName('statusLabel')
        self.send_status.setAlignment(Qt.AlignCenter)
        self.update_status_label(self.send_status, False)

        control_layout.addWidget(QLabel('发送话题:'), 1, 0)
        control_layout.addWidget(self.send_btn, 1, 1)
        control_layout.addWidget(self.send_status, 1, 2)

        layout.addWidget(control_group)

        # 时延统计区域
        stats_group = QGroupBox('时延统计 (ms)')
        stats_layout = QGridLayout()
        stats_group.setLayout(stats_layout)

        # 创建统计标签
        self.avg_label = self.create_stat_label('平均值:', '0.00')
        self.min_label = self.create_stat_label('最小值:', '0.00')
        self.max_label = self.create_stat_label('最大值:', '0.00')
        self.std_label = self.create_stat_label('标准差:', '0.00')

        stats_layout.addWidget(self.avg_label[0], 0, 0)
        stats_layout.addWidget(self.avg_label[1], 0, 1)
        stats_layout.addWidget(self.min_label[0], 0, 2)
        stats_layout.addWidget(self.min_label[1], 0, 3)

        stats_layout.addWidget(self.max_label[0], 1, 0)
        stats_layout.addWidget(self.max_label[1], 1, 1)
        stats_layout.addWidget(self.std_label[0], 1, 2)
        stats_layout.addWidget(self.std_label[1], 1, 3)

        layout.addWidget(stats_group)

        # 消息统计区域
        msg_group = QGroupBox('消息统计')
        msg_layout = QGridLayout()
        msg_group.setLayout(msg_layout)

        self.recv_count_label = self.create_stat_label('接收数量:', '0')
        self.send_count_label = self.create_stat_label('发送数量:', '0')
        self.recv_rate_label = self.create_stat_label('接收频率:', '0.0 Hz')
        self.send_rate_label = self.create_stat_label('发送频率:', '0.0 Hz')

        msg_layout.addWidget(self.recv_count_label[0], 0, 0)
        msg_layout.addWidget(self.recv_count_label[1], 0, 1)
        msg_layout.addWidget(self.recv_rate_label[0], 0, 2)
        msg_layout.addWidget(self.recv_rate_label[1], 0, 3)

        msg_layout.addWidget(self.send_count_label[0], 1, 0)
        msg_layout.addWidget(self.send_count_label[1], 1, 1)
        msg_layout.addWidget(self.send_rate_label[0], 1, 2)
        msg_layout.addWidget(self.send_rate_label[1], 1, 3)

        layout.addWidget(msg_group)

        # 重置按钮
        reset_btn = QPushButton('重置统计')
        reset_btn.clicked.connect(self.reset_stats)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                max-width: 150px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 底部信息
        info_label = QLabel(f'订阅: {self.tester.sub_topic} | 发布: {self.tester.pub_topic} @ 100Hz')
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet('color: #666; font-size: 11px; padding: 5px;')
        layout.addWidget(info_label)

    def create_stat_label(self, title, value):
        """创建统计标签对"""
        title_label = QLabel(title)
        title_label.setObjectName('titleLabel')
        value_label = QLabel(value)
        value_label.setObjectName('valueLabel')
        value_label.setAlignment(Qt.AlignRight)
        return (title_label, value_label)

    def update_status_label(self, label, is_active):
        """更新状态标签样式"""
        if is_active:
            label.setText('运行中')
            label.setStyleSheet('background-color: #4CAF50; color: white;')
        else:
            label.setText('停止')
            label.setStyleSheet('background-color: #f44336; color: white;')

    def toggle_receiving(self):
        """切换接收状态"""
        if self.tester.is_receiving:
            self.tester.stop_receiving()
            self.recv_btn.setText('开始接收')
            self.update_status_label(self.recv_status, False)
        else:
            self.tester.start_receiving()
            self.recv_btn.setText('停止接收')
            self.recv_btn.setObjectName('stopBtn')
            self.update_status_label(self.recv_status, True)
        self.recv_btn.style().polish(self.recv_btn)

    def toggle_publishing(self):
        """切换发送状态"""
        if self.tester.is_publishing:
            self.tester.stop_publishing()
            self.send_btn.setText('开始发送')
            self.update_status_label(self.send_status, False)
        else:
            self.tester.start_publishing()
            self.send_btn.setText('停止发送')
            self.send_btn.setObjectName('stopBtn')
            self.update_status_label(self.send_status, True)
        self.send_btn.style().polish(self.send_btn)

    def reset_stats(self):
        """重置统计"""
        reply = QMessageBox.question(
            self, '确认重置',
            '确定要重置所有统计数据吗？',
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.tester.reset_stats()
            QMessageBox.information(self, '提示', '统计数据已重置')

    def update_display(self, stats):
        """更新显示"""
        # 更新时延统计
        latency = stats['latency']
        if latency['count'] > 0:
            self.avg_label[1].setText(f"{latency['avg']:.2f}")
            self.min_label[1].setText(f"{latency['min']:.2f}")
            self.max_label[1].setText(f"{latency['max']:.2f}")
            self.std_label[1].setText(f"{latency['std']:.2f}")
        else:
            self.avg_label[1].setText("--")
            self.min_label[1].setText("--")
            self.max_label[1].setText("--")
            self.std_label[1].setText("--")

        # 更新消息统计
        self.recv_count_label[1].setText(str(stats['recv_count']))
        self.send_count_label[1].setText(str(stats['send_count']))
        self.recv_rate_label[1].setText(f"{stats['recv_rate']:.1f} Hz")
        self.send_rate_label[1].setText(f"{stats['send_rate']:.1f} Hz")

    def closeEvent(self, event):
        """关闭事件"""
        # 停止所有操作
        if self.tester.is_receiving:
            self.tester.stop_receiving()
        if self.tester.is_publishing:
            self.tester.stop_publishing()

        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用图标和样式
    app.setStyle('Fusion')

    # 创建主窗口
    window = LatencyTestUI()
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()