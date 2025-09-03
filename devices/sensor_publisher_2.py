#!/usr/bin/env python
"""
从CSV文件读取数据并发布到ROS话题
用于测试外骨骼推理系统
增强版：包含moment订阅和时间戳模拟
"""

import rospy
from std_msgs.msg import Float64MultiArray
import pandas as pd
import numpy as np
import argparse
import os
import time
import threading


class CSVDataPublisher:
    def __init__(self, csv_path, label_filter=None, topic='/motor12_left',
                 moment_topic='/moment', rate=200):
        """
        初始化CSV数据发布器
        Args:
            csv_path: CSV文件路径
            label_filter: 标签过滤器，只发送特定label的数据
            topic: ROS话题名称
            moment_topic: moment数据订阅话题
            rate: 发布频率(Hz)
        """
        self.csv_path = csv_path
        self.label_filter = label_filter
        self.topic = topic
        self.moment_topic = moment_topic
        self.rate = rate

        # 列名定义（与CustomDataLoader保持一致）
        self.column_names = [
            'motorPos', 'motorVel',
            'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'label'
        ]

        # 数据相关
        self.data = None
        self.filtered_data = None
        self.current_index = 0

        # ROS发布器和订阅器
        self.publisher = None
        self.moment_subscriber = None

        # moment数据缓存（线程安全）
        self.moment_lock = threading.Lock()
        self.latest_moment = 0.0
        self.latest_timestamp_back = 0.0

        # 模拟的sensor时间戳（单位：ms）
        self.timestamp_sensor = 0.0
        self.timestamp_increment = self.rate / 100  # 每次增加10ms

        # 加载数据
        self.load_data()

    def moment_callback(self, msg):
        """
        moment话题回调函数
        假设moment话题发布的是Float64MultiArray，格式为[moment_value, timestamp]
        """
        with self.moment_lock:
            if len(msg.data) > 0:
                self.latest_moment = msg.data[0]
                # 如果有第二个值，作为timestamp_back
                if len(msg.data) > 1:
                    self.latest_timestamp_back = msg.data[1]
                else:
                    # 如果没有时间戳，使用当前ROS时间（转换为ms）
                    self.latest_timestamp_back = 0.0

                rospy.logdebug(f"Received moment: {self.latest_moment}, timestamp: {self.latest_timestamp_back}")

    def load_data(self):
        """加载CSV数据"""
        if not os.path.exists(self.csv_path):
            rospy.logerr(f"CSV file not found: {self.csv_path}")
            return False

        try:
            # 尝试读取CSV文件，自动检测是否有表头
            try:
                # 先尝试读取第一行
                first_line = pd.read_csv(self.csv_path, nrows=1, header=None)
                # 检查第一行是否包含字符串（表头）
                if first_line.iloc[0].dtype == object and 'motorPos' in str(first_line.iloc[0, 0]):
                    # 有表头
                    self.data = pd.read_csv(self.csv_path)
                    self.data.columns = self.column_names
                    rospy.loginfo(f"Loaded {len(self.data)} rows from {self.csv_path} (with header)")
                else:
                    # 没有表头
                    self.data = pd.read_csv(self.csv_path, header=None, names=self.column_names)
                    rospy.loginfo(f"Loaded {len(self.data)} rows from {self.csv_path} (no header)")
            except:
                # 默认假设没有表头
                self.data = pd.read_csv(self.csv_path, header=None, names=self.column_names)
                rospy.loginfo(f"Loaded {len(self.data)} rows from {self.csv_path}")

            # 验证数据类型
            for col in self.column_names[:-1]:  # 除了label列
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # label列转换为整数
            self.data['label'] = pd.to_numeric(self.data['label'], errors='coerce').astype('Int64')

            # 删除包含NaN的行
            self.data = self.data.dropna()

            # 应用标签过滤
            if self.label_filter is not None:
                self.filtered_data = self.data[self.data['label'] == self.label_filter].reset_index(drop=True)
                rospy.loginfo(f"Filtered to {len(self.filtered_data)} rows with label={self.label_filter}")

                if len(self.filtered_data) == 0:
                    rospy.logwarn(f"No data found with label={self.label_filter}")
                    return False
            else:
                self.filtered_data = self.data
                rospy.loginfo(f"Using all {len(self.filtered_data)} rows (no label filter)")

            self.current_index = 0
            return True

        except Exception as e:
            rospy.logerr(f"Error loading CSV file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_next_frame(self):
        """获取下一帧数据"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            return None

        # 如果到达末尾，循环回到开始
        if self.current_index >= len(self.filtered_data):
            self.current_index = 0
            rospy.loginfo(f"Looping back to start of data (frame {len(self.filtered_data)})")

        row = self.filtered_data.iloc[self.current_index]
        self.current_index += 1

        # 获取moment数据（线程安全）
        with self.moment_lock:
            current_moment = self.latest_moment
            current_timestamp_back = self.latest_timestamp_back

        # 增加sensor时间戳
        self.timestamp_sensor += self.timestamp_increment

        # 构建扩展的数据格式
        # 注意：label不再包含在发布的数据中，因为新格式中没有label字段
        frame_data = [
            float(row['motorPos']),  # [0]
            float(row['motorVel']),  # [1]
            float(row['acc_x']),  # [2]
            float(row['acc_y']),  # [3]
            float(row['acc_z']),  # [4]
            float(row['gyro_x']),  # [5]
            float(row['gyro_y']),  # [6]
            float(row['gyro_z']),  # [7]
            float(current_moment),  # [8] moment (从/moment话题接收)
            float(self.timestamp_sensor),  # [9] timestamp_sensor (模拟设备时钟)
            float(current_timestamp_back)  # [10] timestamp_back (从/moment话题接收)
        ]

        return frame_data, int(row['label'])  # 返回数据和label（用于日志）

    def publish_data(self):
        """主发布循环"""
        # 初始化ROS节点
        rospy.init_node('csv_sensor_publisher', anonymous=True)

        # 创建发布器
        self.publisher = rospy.Publisher(self.topic, Float64MultiArray, queue_size=10)

        # 创建moment订阅器
        self.moment_subscriber = rospy.Subscriber(
            self.moment_topic,
            Float64MultiArray,
            self.moment_callback,
            queue_size=10
        )

        # 设置发布频率
        rate = rospy.Rate(self.rate)

        rospy.loginfo(f"Starting to publish data to {self.topic} at {self.rate}Hz")
        rospy.loginfo(f"Subscribing to moment data from {self.moment_topic}")

        frame_count = 0
        start_time = rospy.Time.now()

        # 等待一小段时间让订阅器初始化
        rospy.sleep(0.5)

        while not rospy.is_shutdown():
            # 获取下一帧数据
            result = self.get_next_frame()

            if result is None:
                rospy.logwarn("No data available to publish")
                break

            frame_data, label = result

            # 创建并发布消息
            msg = Float64MultiArray()
            msg.data = frame_data
            self.publisher.publish(msg)

            # 进度显示（每秒显示一次）
            frame_count += 1
            if frame_count % self.rate == 0:
                elapsed = (rospy.Time.now() - start_time).to_sec()
                actual_rate = frame_count / elapsed if elapsed > 0 else 0
                progress = (self.current_index / len(self.filtered_data)) * 100

                rospy.loginfo(
                    f"Published {frame_count} frames | "
                    f"Progress: {progress:.1f}% | "
                    f"Rate: {actual_rate:.1f}Hz | "
                    f"Frame: {self.current_index}/{len(self.filtered_data)} | "
                    f"Label: {label} | "
                    f"Moment: {frame_data[8]:.3f} | "
                    f"Sensor_TS: {frame_data[9]:.1f}ms | "
                    f"Back_TS: {frame_data[10]:.1f}ms"
                )

            # 按照设定频率休眠
            rate.sleep()

        rospy.loginfo("Publishing stopped")

    def show_info(self):
        """显示数据信息"""
        if self.data is not None:
            print("\n=== CSV Data Info ===")
            print(f"Total rows: {len(self.data)}")
            print(f"Filtered rows: {len(self.filtered_data)}")
            print(f"Unique labels: {sorted(self.data['label'].unique())}")
            print("\nLabel distribution:")
            print(self.data['label'].value_counts().sort_index())
            print("\nFirst 5 rows of filtered data:")
            print(self.filtered_data.head())
            print("\n=== Publishing Format ===")
            print("Data array indices:")
            print("  [0]: motorPos")
            print("  [1]: motorVel")
            print("  [2]: acc_x")
            print("  [3]: acc_y")
            print("  [4]: acc_z")
            print("  [5]: gyro_x")
            print("  [6]: gyro_y")
            print("  [7]: gyro_z")
            print("  [8]: moment (from /moment topic)")
            print("  [9]: timestamp_sensor (simulated, 10ms steps)")
            print("  [10]: timestamp_back (from /moment topic)")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Publish CSV sensor data to ROS topic with moment subscription')
    parser.add_argument(
        '--csv',
        type=str,
        default='kneeData_left.csv',
        help='Path to CSV file (default: kneeData_left.csv)'
    )
    parser.add_argument(
        '--label',
        type=int,
        default=2,
        help='Label filter: only publish data with this label (default: 2)'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='/motor12_left',
        help='ROS topic name for publishing data (default: /motor12_left)'
    )
    parser.add_argument(
        '--moment-topic',
        type=str,
        default='/moment',
        help='ROS topic name for subscribing moment data (default: /moment)'
    )
    parser.add_argument(
        '--rate',
        type=int,
        default=100,
        help='Publishing rate in Hz (default: 200)'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show data info and exit without publishing'
    )

    args = parser.parse_args()

    # 创建发布器
    publisher = CSVDataPublisher(
        csv_path=args.csv,
        label_filter=args.label,
        topic=args.topic,
        moment_topic=args.moment_topic,
        rate=args.rate
    )

    # 如果只是查看信息
    if args.info:
        publisher.show_info()
        return

    # 检查数据是否加载成功
    if publisher.filtered_data is None or len(publisher.filtered_data) == 0:
        rospy.logerr("Failed to load data or no data after filtering")
        return

    try:
        # 开始发布数据
        publisher.publish_data()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received, shutting down")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received, shutting down")


if __name__ == '__main__':
    main()