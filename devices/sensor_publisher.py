#!/usr/bin/env python
"""
从CSV文件读取数据并发布到ROS话题
用于测试外骨骼推理系统
"""

import rospy
from std_msgs.msg import Float64MultiArray
import pandas as pd
import numpy as np
import argparse
import os
import time


class CSVDataPublisher:
    def __init__(self, csv_path, label_filter=None, topic='/exo_sensor_data', rate=200):
        """
        初始化CSV数据发布器
        Args:
            csv_path: CSV文件路径
            label_filter: 标签过滤器，只发送特定label的数据
            topic: ROS话题名称
            rate: 发布频率(Hz)
        """
        self.csv_path = csv_path
        self.label_filter = label_filter
        self.topic = topic
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

        # ROS发布器
        self.publisher = None

        # 加载数据
        self.load_data()

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

        # 转换为列表格式（用于ROS消息）
        frame_data = [
            float(row['motorPos']),
            float(row['motorVel']),
            float(row['acc_x']),
            float(row['acc_y']),
            float(row['acc_z']),
            float(row['gyro_x']),
            float(row['gyro_y']),
            float(row['gyro_z']),
            int(row['label'])
        ]

        return frame_data

    def publish_data(self):
        """主发布循环"""
        # 初始化ROS节点
        rospy.init_node('csv_sensor_publisher', anonymous=True)

        # 创建发布器
        self.publisher = rospy.Publisher(self.topic, Float64MultiArray, queue_size=10)

        # 设置发布频率
        rate = rospy.Rate(self.rate)

        rospy.loginfo(f"Starting to publish data to {self.topic} at {self.rate}Hz")

        frame_count = 0
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            # 获取下一帧数据
            frame_data = self.get_next_frame()

            if frame_data is None:
                rospy.logwarn("No data available to publish")
                break

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
                    f"Actual rate: {actual_rate:.1f}Hz | "
                    f"Current frame: {self.current_index}/{len(self.filtered_data)}"
                )

            # 按照设定频率休眠
            rate.sleep()

        rospy.loginfo("Publishing stopped")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Publish CSV sensor data to ROS topic')
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
        help='Label filter: only publish data with this label (default: all labels)'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='/exo_sensor_data',
        help='ROS topic name (default: /exo_sensor_data)'
    )
    parser.add_argument(
        '--rate',
        type=int,
        default=200,
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
        rate=args.rate
    )

    # 如果只是查看信息
    if args.info:
        if publisher.data is not None:
            print("\n=== CSV Data Info ===")
            print(f"Total rows: {len(publisher.data)}")
            print(f"Filtered rows: {len(publisher.filtered_data)}")
            print(f"Unique labels: {sorted(publisher.data['label'].unique())}")
            print("\nLabel distribution:")
            print(publisher.data['label'].value_counts().sort_index())
            print("\nFirst 5 rows of filtered data:")
            print(publisher.filtered_data.head())
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