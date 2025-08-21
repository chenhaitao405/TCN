#!/usr/bin/env python3
"""
实时推理系统主程序
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore', message='dropout2d: Received a 3D input')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必须在导入PyQt5之前设置
from PyQt5.QtCore import Qt, QCoreApplication

# 在创建QApplication之前设置高DPI支持
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from PyQt5.QtWidgets import QApplication
from inference_engine import InferenceEngine
from data_stream import DataStreamManager
from ui_dashboard import RealtimeInferenceDashboard


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='实时关节力矩推理系统')
    parser.add_argument('--config', type=str, default='configs.hiponly_config',
                        help='配置文件路径 (默认: configs.hiponly_config)')
    parser.add_argument('--no-gui', action='store_true',
                        help='无GUI模式运行')
    args = parser.parse_args()

    print("=" * 60)
    print("外骨骼关节力矩实时推理系统")
    print("=" * 60)

    try:
        # 初始化推理引擎
        print("\n初始化推理引擎...")
        engine = InferenceEngine(args.config)

        # 初始化数据流管理器
        print("\n初始化数据流管理器...")
        data_stream = DataStreamManager(engine.config)

        if args.no_gui:
            # 无GUI模式（用于测试或集成）
            print("\n运行无GUI模式...")
            run_headless(engine, data_stream)
        else:
            # 启动GUI
            print("\n启动图形界面...")
            app = QApplication(sys.argv)

            # 创建主窗口
            dashboard = RealtimeInferenceDashboard(engine, data_stream)
            dashboard.show()

            print("系统已就绪！")
            print("-" * 60)

            sys.exit(app.exec_())

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_headless(engine, data_stream):
    """无GUI模式运行"""
    print("选择第一个试验进行测试...")
    data_stream.dataset_source.select_trial(0)

    print("开始推理...")
    frame_count = 0
    for sensor_data, ground_truth, timestamp in data_stream.stream_frames():
        # 执行推理
        moments = engine.process_frame(sensor_data)

        # 打印结果
        if frame_count % 100 == 0:  # 每100帧打印一次
            print(f"\n帧 {frame_count} (时间: {timestamp:.2f}s):")
            print(f"  推理结果: {moments}")
            if ground_truth:
                print(f"  真实值: {ground_truth}")

            stats = engine.get_performance_stats()
            print(f"  推理时间: {stats['avg_inference_time']:.2f}ms")

        frame_count += 1

        # 测试1000帧后停止
        if frame_count >= 1000:
            break

    print(f"\n测试完成，共处理 {frame_count} 帧")


if __name__ == "__main__":
    main()