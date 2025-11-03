#!/usr/bin/env python
"""
从CSV文件读取电机数据并发布到ROS话题
CSV格式: timestamp, motor_angle_L_float, motor_angle_R_float, motor_vel_L_float, motor_vel_R_float
发布格式: [hip_angle_l, hip_angle_r, hip_angle_l_velocity, hip_angle_r_velocity, torque_L, torque_R]
"""

import rospy
from std_msgs.msg import Float32MultiArray
import pandas as pd
import numpy as np
import argparse
import os


class MotorDataPublisher:
    def __init__(self, csv_path, topic='/motor_data', rate=100, torque_L=0.0, torque_R=0.0):
        """
        初始化电机数据发布器
        Args:
            csv_path: CSV文件路径
            topic: ROS话题名称
            rate: 发布频率(Hz)
            torque_L: 左侧扭矩默认值
            torque_R: 右侧扭矩默认值
        """
        self.csv_path = csv_path
        self.topic = topic
        self.rate = rate
        self.torque_L = torque_L
        self.torque_R = torque_R

        # CSV列名定义
        self.column_names = [
            'timestamp',
            'motor_angle_L_float',
            'motor_angle_R_float',
            'motor_vel_L_float',
            'motor_vel_R_float'
        ]

        # 数据相关
        self.data = None
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
            # 读取CSV文件
            self.data = pd.read_csv(self.csv_path)

            # 检查是否包含所需的列
            required_cols = ['motor_angle_L_float', 'motor_angle_R_float',
                             'motor_vel_L_float', 'motor_vel_R_float']

            if not all(col in self.data.columns for col in required_cols):
                rospy.logerr(f"CSV file missing required columns. Expected: {required_cols}")
                rospy.logerr(f"Found columns: {list(self.data.columns)}")
                return False

            rospy.loginfo(f"Loaded {len(self.data)} rows from {self.csv_path}")

            # 验证数据类型
            for col in required_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # 删除包含NaN的行
            original_len = len(self.data)
            self.data = self.data.dropna(subset=required_cols)
            if len(self.data) < original_len:
                rospy.logwarn(f"Removed {original_len - len(self.data)} rows with invalid data")

            if len(self.data) == 0:
                rospy.logerr("No valid data after cleaning")
                return False

            self.current_index = 0
            return True

        except Exception as e:
            rospy.logerr(f"Error loading CSV file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_next_frame(self):
        """获取下一帧数据"""
        if self.data is None or len(self.data) == 0:
            return None

        # 如果到达末尾，循环回到开始
        if self.current_index >= len(self.data):
            self.current_index = 0
            rospy.loginfo("Looping back to start of data")

        row = self.data.iloc[self.current_index]
        self.current_index += 1

        # 构建发布数据格式
        # 接收端格式: sensor_data[:4] = [hip_angle_l, hip_angle_r, hip_angle_l_velocity, hip_angle_r_velocity]
        #            torque[-2:] = [torque_L, torque_R]
        frame_data = [
            float(row['motor_angle_L_float']),  # [0] hip_angle_l
            float(row['motor_angle_R_float']),  # [1] hip_angle_r
            float(row['motor_vel_L_float']),  # [2] hip_angle_l_velocity
            float(row['motor_vel_R_float']),  # [3] hip_angle_r_velocity
            float(self.torque_L),  # [4] torque_L
            float(self.torque_R)  # [5] torque_R
        ]

        return frame_data

    def publish_data(self):
        """主发布循环"""
        # 初始化ROS节点
        rospy.init_node('motor_data_publisher', anonymous=True)

        # 创建发布器
        self.publisher = rospy.Publisher(self.topic, Float32MultiArray, queue_size=10)

        # 设置发布频率
        rate = rospy.Rate(self.rate)

        rospy.loginfo(f"Starting to publish data to {self.topic} at {self.rate}Hz")
        rospy.loginfo(f"Data format: [hip_angle_l, hip_angle_r, hip_vel_l, hip_vel_r, torque_L, torque_R]")

        frame_count = 0
        start_time = rospy.Time.now()

        # 等待一小段时间让发布器初始化
        rospy.sleep(0.5)

        while not rospy.is_shutdown():
            # 获取下一帧数据
            frame_data = self.get_next_frame()

            if frame_data is None:
                rospy.logwarn("No data available to publish")
                break

            # 创建并发布消息
            msg = Float32MultiArray()
            msg.data = frame_data
            self.publisher.publish(msg)

            # 进度显示（每秒显示一次）
            frame_count += 1
            if frame_count % self.rate == 0:
                elapsed = (rospy.Time.now() - start_time).to_sec()
                actual_rate = frame_count / elapsed if elapsed > 0 else 0
                progress = (self.current_index / len(self.data)) * 100

                rospy.loginfo(
                    f"Published {frame_count} frames | "
                    f"Progress: {progress:.1f}% | "
                    f"Rate: {actual_rate:.1f}Hz | "
                    f"Frame: {self.current_index}/{len(self.data)} | "
                    f"L_angle: {frame_data[0]:.3f}, R_angle: {frame_data[1]:.3f}"
                )

            # 按照设定频率休眠
            rate.sleep()

        rospy.loginfo("Publishing stopped")

    def show_info(self):
        """显示数据信息"""
        if self.data is not None:
            print("\n=== CSV Data Info ===")
            print(f"Total rows: {len(self.data)}")
            print(f"Columns: {list(self.data.columns)}")
            print("\nData statistics:")
            print(self.data[['motor_angle_L_float', 'motor_angle_R_float',
                             'motor_vel_L_float', 'motor_vel_R_float']].describe())
            print("\nFirst 5 rows:")
            print(self.data.head())
            print("\n=== Publishing Format ===")
            print("Data array structure (6 elements):")
            print("  [0]: hip_angle_l (motor_angle_L_float)")
            print("  [1]: hip_angle_r (motor_angle_R_float)")
            print("  [2]: hip_angle_l_velocity (motor_vel_L_float)")
            print("  [3]: hip_angle_r_velocity (motor_vel_R_float)")
            print(f"  [4]: torque_L (fixed value: {self.torque_L})")
            print(f"  [5]: torque_R (fixed value: {self.torque_R})")
            print("\nReceiver code extracts:")
            print("  sensor_data = msg.data[:4]  # angles and velocities")
            print("  torque = msg.data[-2:]      # torques")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Publish motor data from CSV to ROS topic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python motor_data_publisher.py --csv data.csv

  # Specify topic and rate
  python motor_data_publisher.py --csv data.csv --topic /hip_motors --rate 200

  # Set torque values
  python motor_data_publisher.py --csv data.csv --torque-l 5.0 --torque-r 5.0

  # Show data info without publishing
  python motor_data_publisher.py --csv data.csv --info
        """
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with motor data'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='/motor_data',
        help='ROS topic name for publishing (default: /motor_data)'
    )
    parser.add_argument(
        '--rate',
        type=int,
        default=100,
        help='Publishing rate in Hz (default: 100)'
    )
    parser.add_argument(
        '--torque-l',
        type=float,
        default=0.0,
        help='Left torque value (default: 0.0)'
    )
    parser.add_argument(
        '--torque-r',
        type=float,
        default=0.0,
        help='Right torque value (default: 0.0)'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show data info and exit without publishing'
    )

    args = parser.parse_args()

    # 创建发布器
    publisher = MotorDataPublisher(
        csv_path=args.csv,
        topic=args.topic,
        rate=args.rate,
        torque_L=args.torque_l,
        torque_R=args.torque_r
    )

    # 如果只是查看信息
    if args.info:
        publisher.show_info()
        return

    # 检查数据是否加载成功
    if publisher.data is None or len(publisher.data) == 0:
        rospy.logerr("Failed to load data")
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