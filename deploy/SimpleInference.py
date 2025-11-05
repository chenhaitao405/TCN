"""
简单的推理测试脚本
用于膝关节力矩预测的单元测试
"""
import torch
import numpy as np
import pandas as pd
from collections import deque
from scipy import signal
import time
import sys
import os
import inspect

# 添加模型路径（根据实际部署调整）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleTCNInference:
    """简化的TCN模型推理类"""

    def __init__(self, model_path: str, csv_path: str):
        """
        初始化推理器

        Args:
            model_path: TCN模型文件路径 (.tar文件)
            csv_path: 输入数据CSV文件路径
        """
        # ========== 固定参数配置 ==========
        # 模型参数
        self.history_window = 280  # 默认历史窗口大小
        self.input_rate = 100  # 输入采样率
        self.target_rate = 200  # 目标采样率（用于重采样）

        # ========== 加载模型 ==========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.model = self._load_model(model_path)
        print(f"模型加载成功: {model_path}")

        # ========== 加载数据 ==========
        self.data = self._load_csv_data(csv_path)
        print(f"数据加载成功: {len(self.data)} 帧")

        # ========== 初始化缓冲区 ==========
        self.buffer_size = self.history_window + 100
        self.data_buffer = deque(maxlen=self.buffer_size)

        # 计算需要的输入帧数
        self.input_frames_needed = int(self.history_window * self.input_rate / 200)
        print(f"每次推理需要 {self.input_frames_needed} 帧输入数据")

    def _load_model(self, model_path: str):
        """
        加载TCN模型

        Args:
            model_path: 模型文件路径

        Returns:
            加载好的模型
        """
        # 加载模型信息
        try:
            from utils.tcn import TCN
        except ImportError:
            print("警告: 无法导入TCN")

        model_info = torch.load(model_path, map_location=self.device)
        state_dict = model_info["state_dict"]
        del model_info["state_dict"]
        tcn_signature = inspect.signature(TCN.__init__)
        tcn_param_names = [param.name for param in tcn_signature.parameters.values()
                           if param.name != 'self']

        # Only pass parameters that TCN needs
        tcn_params = {k: v for k, v in model_info.items()
                      if k in tcn_param_names}
        tcn = TCN(**tcn_params).to(self.device)
        tcn.load_state_dict(state_dict)

        self.history_window = tcn.get_effective_history() #获取模型感受野
        self.buffer_size = self.history_window + 100
        self.data_buffer = deque(maxlen=self.buffer_size)

        tcn.eval()  # 设置为评估模式
        return tcn

    def _load_csv_data(self, csv_path: str) -> np.ndarray:
        """
        从CSV文件加载数据

        期望的CSV列格式:
        motorPos, motorVel, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

        Args:
            csv_path: CSV文件路径

        Returns:
            numpy数组，shape: (n_frames, 8)
        """
        # 读取CSV
        df = pd.read_csv(csv_path)

        # 提取需要的列并按照模型输入顺序排列
        # 模型期望的顺序: thigh_gyro_xyz, thigh_accel_xyz, knee_angle, knee_velocity
        required_columns = ['gyro_x', 'gyro_y', 'gyro_z',
                            'acc_x', 'acc_y', 'acc_z',
                            'motorPos', 'motorVel']

        # 检查列是否存在
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必需的列: {col}")

        # 按照正确顺序提取数据
        data = df[required_columns].values

        return data


    def process_frame(self, frame_data: np.ndarray) -> float:
        """
        处理单帧数据并返回预测的膝关节力矩

        Args:
            frame_data: 单帧传感器数据，shape: (8,)

        Returns:
            预测的膝关节力矩值，如果缓冲区数据不足则返回None
        """
        # 添加到缓冲区
        self.data_buffer.append(frame_data)

        # 检查是否有足够的历史数据
        if len(self.data_buffer) < self.input_frames_needed:
            return None

        # 准备输入窗口
        input_window = np.array(list(self.data_buffer))[-self.input_frames_needed:]

        # 重采样（如果需要）
        if self.input_rate != 200:
            # 从100Hz重采样到200Hz
            input_window = signal.resample(input_window, self.history_window, axis=0)

        # 转换为张量: shape [1, features, time]
        input_tensor = torch.tensor(
            input_window.T,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # 模型推理
        with torch.no_grad():
            output = self.model(input_tensor)

        # 提取预测值（最后一个时间步）
        # output shape: [batch, outputs, time]
        torque = output[0, 0, -1].item()


        return torque

    def run_inference(self):
        """
        运行完整的推理测试
        """
        print("\n" + "=" * 50)
        print("开始推理测试")
        print("=" * 50)

        predictions = []
        inference_times = []

        # 逐帧处理数据
        for i, frame in enumerate(self.data):
            start_time = time.time()

            # 推理
            torque = self.process_frame(frame)

            # 记录推理时间
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒

            if torque is not None:
                predictions.append(torque)
                inference_times.append(inference_time)

                # 每100帧打印一次状态
                if len(predictions) % 100 == 0:
                    avg_torque = np.mean(predictions[-100:])
                    avg_time = np.mean(inference_times[-100:])
                    print(f"处理帧 {i + 1}/{len(self.data)}: "
                          f"平均力矩={avg_torque:.4f} Nm, "
                          f"平均推理时间={avg_time:.2f} ms")

        # 打印统计信息
        print("\n" + "=" * 50)
        print("推理完成 - 统计信息")
        print("=" * 50)
        print(f"总帧数: {len(self.data)}")
        print(f"有效预测数: {len(predictions)}")
        print(f"跳过帧数（缓冲区填充）: {len(self.data) - len(predictions)}")
        print(f"\n力矩统计:")
        print(f"  - 平均值: {np.mean(predictions):.4f} Nm")
        print(f"  - 标准差: {np.std(predictions):.4f} Nm")
        print(f"  - 最小值: {np.min(predictions):.4f} Nm")
        print(f"  - 最大值: {np.max(predictions):.4f} Nm")
        print(f"\n推理时间统计:")
        print(f"  - 平均值: {np.mean(inference_times):.2f} ms")
        print(f"  - 最大值: {np.max(inference_times):.2f} ms")
        print(f"  - 最小值: {np.min(inference_times):.2f} ms")

        # 返回预测结果
        return np.array(predictions)


def main():
    """
    主函数
    """
    # 配置路径（根据实际情况修改）
    model_path = "./deploy/model_knee_manual_windows.tar"
    csv_path = "./deploy/kneeData_left.csv"  # 需要准备测试数据CSV文件

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return

    if not os.path.exists(csv_path):
        print(f"警告: CSV文件不存在: {csv_path}")
        print("创建示例CSV文件...")

        # 创建示例数据
        create_sample_csv(csv_path)

    # 创建推理器并运行测试
    try:
        inferencer = SimpleTCNInference(model_path, csv_path)
        predictions = inferencer.run_inference()

        # 可选：保存预测结果
        np.savetxt("predictions.txt", predictions, fmt='%.6f')
        print(f"\n预测结果已保存到 predictions.txt")

    except Exception as e:
        print(f"推理过程出错: {e}")
        import traceback
        traceback.print_exc()


def create_sample_csv(filename: str, n_samples: int = 1000):
    """
    创建示例CSV数据文件

    Args:
        filename: 输出文件名
        n_samples: 样本数量
    """
    # 生成模拟数据
    t = np.linspace(0, 10, n_samples)

    data = {
        'motorPos': -30 + 20 * np.sin(2 * np.pi * 0.5 * t),  # 膝关节角度
        'motorVel': 40 * np.pi * np.cos(2 * np.pi * 0.5 * t),  # 膝关节角速度
        'acc_x': np.random.randn(n_samples) * 0.5,
        'acc_y': np.random.randn(n_samples) * 0.2,
        'acc_z': np.random.randn(n_samples) * 0.1 + 9.8,
        'gyro_x': 80 * np.sin(2 * np.pi * 0.8 * t) + np.random.randn(n_samples) * 10,
        'gyro_y': 4 * np.cos(2 * np.pi * 0.8 * t) + np.random.randn(n_samples) * 0.5,
        'gyro_z': 4.5 * np.sin(2 * np.pi * 0.8 * t + np.pi / 4) + np.random.randn(n_samples) * 0.5,
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"已创建示例CSV文件: {filename}")


if __name__ == "__main__":
    main()