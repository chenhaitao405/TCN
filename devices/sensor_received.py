#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64MultiArray


def motor_callback(data):
    """
    回调函数，处理接收到的Float64MultiArray消息

    Args:
        data: Float64MultiArray消息
    """
    # 打印接收到的数据
    rospy.loginfo("Received motor data:")
    rospy.loginfo("  Data: %s", data.data)

    # 如果有layout信息，也可以打印
    if data.layout.dim:
        rospy.loginfo("  Layout dimensions: %s", data.layout.dim)

    # 处理数据（根据你的需求修改）
    for i, value in enumerate(data.data):
        rospy.loginfo("  Motor[%d] = %f", i, value)


def motor_subscriber():
    """
    初始化ROS节点并订阅motor话题
    """
    # 初始化ROS节点
    rospy.init_node('motor_subscriber', anonymous=True)

    # 订阅话题
    motor_topic = "/motor12_left"
    rospy.Subscriber(motor_topic, Float64MultiArray, motor_callback)

    rospy.loginfo("Motor subscriber started. Listening to topic: %s", motor_topic)

    # 保持节点运行
    rospy.spin()


if __name__ == '__main__':
    try:
        motor_subscriber()
    except rospy.ROSInterruptException:
        rospy.loginfo("Motor subscriber node terminated.")