#!/usr/bin/env python
"""
修正版离线推理脚本 - 确保与在线推理结果完全一致
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from collections import deque
import pickle
import os
import sys
from datetime import datetime
from pathlib import Path
from model.ConvTimeNet_backbone import ConvTimeNet_backbone

# 导入ConvTimeNet模型



class OfflineInferenceFixed:
    """
    修正版离线推理器 - 完全复现在线推理逻辑
    """
    def __init__(self, model_path, norm_stats_path, device='cuda'):
        """
        初始化离线推理器
        
        Args:
            model_path: 模型checkpoint路径
            norm_stats_path: 归一化统计数据路径
            device: 推理设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载归一化统计数据
        with open(norm_stats_path, 'rb') as f:
            self.norm_stats = pickle.load(f)
        print("加载归一化统计数据成功")
        
        # 加载模型
        self.model = self.load_model(model_path)
        
        # 输入缓冲区参数（与在线推理保持一致）
        self.input_buffer_size = 218
        self.input_feature_size = 4
        
        # 滤波器参数（与在线推理保持一致）
        self.cutoff_freq = 10.0  # 截止频率 10 Hz
        self.sampling_rate = 200.0  # 采样率 200 Hz
        self.filter_order = 2  # 二阶滤波器
        
        # 设计巴特沃斯滤波器
        nyquist_freq = self.sampling_rate / 2
        normalized_cutoff = self.cutoff_freq / nyquist_freq
        self.b, self.a = scipy_signal.butter(
            self.filter_order, normalized_cutoff, btype='low', analog=False
        )
        
        # 预测缓冲区（用于滤波）
        self.prediction_buffer_size = 30
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        model = ConvTimeNet_backbone(
            c_in=2,
            n_layers=4,
            seq_len=218,
            context_window=218,
            target_window=218,
            patch_len=32,
            stride=16,
            d_model=64,
            d_ff=128,
            dropout=0.2,
            act="gelu",
            enable_res_param=False,
            dw_ks=[5, 5, 7, 7, 13, 13, 19, 19],
            norm='batch',
            re_param=False,
            deformable=True,
            reduced_channels=16,
            revin=False,
            final_out=1,
        ).to(self.device)
        
        # 加载checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"模型加载成功: {model_path}")
        else:
            print(f"警告: 模型文件不存在: {model_path}")
            print("使用随机初始化的模型进行演示")
        
        return model
    
    def apply_filter_vel(self, buffer_array):
        """
        对速度数据应用巴特沃斯滤波器（与在线推理完全一致）
        buffer_array: shape (n, 4)
        """
        if buffer_array.shape[0] < 3:
            return buffer_array
        
        filtered_data = np.zeros_like(buffer_array)
        
        try:
            # 角度数据不滤波（与在线推理一致）
            filtered_data[:, 0] = buffer_array[:, 0]
            filtered_data[:, 1] = buffer_array[:, 1]
            
            # 滤波角速度数据
            filtered_data[:, 2] = scipy_signal.filtfilt(self.b, self.a, buffer_array[:, 2])
            filtered_data[:, 3] = scipy_signal.filtfilt(self.b, self.a, buffer_array[:, 3])
        except Exception as e:
            print(f"滤波失败，使用原始数据: {e}")
            return buffer_array
        
        return filtered_data
    
    def apply_filter_torque(self, buffer_array):
        """
        对扭矩预测结果应用巴特沃斯滤波器
        buffer_array: shape (n, 2)
        """
        if buffer_array.shape[0] < 3:
            return buffer_array
        
        filtered_data = np.zeros_like(buffer_array)
        
        try:
            filtered_data[:, 0] = scipy_signal.filtfilt(self.b, self.a, buffer_array[:, 0])
            filtered_data[:, 1] = scipy_signal.filtfilt(self.b, self.a, buffer_array[:, 1])
        except Exception as e:
            print(f"滤波失败，使用原始数据: {e}")
            return buffer_array
        
        return filtered_data
    
    def preprocess_single_frame(self, sensor_data):
        """
        预处理单帧数据（完全复现在线推理的sensor_callback逻辑）
        
        Args:
            sensor_data: shape (4,) 的传感器数据 [hip_angle_l, hip_angle_r, vel_l, vel_r]
        
        Returns:
            预处理后的数据
        """
        # 这里完全复现在线推理的预处理步骤
        # 注意：输入数据已经是弧度制
        return sensor_data
    
    def run_inference_streaming(self, csv_path, save_results=True):
        """
        流式处理CSV文件（完全模拟在线推理的逐帧处理）
        
        Args:
            csv_path: CSV文件路径
            save_results: 是否保存推理结果
        
        Returns:
            DataFrame包含原始数据和离线推理结果
        """
        # 读取CSV文件
        print(f"读取CSV文件: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 检查必要的列是否存在
        required_cols = ['hip_angle_l', 'hip_angle_r', 
                        'hip_angle_l_velocity', 'hip_angle_r_velocity']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必要的列: {col}")
        
        # 初始化结果列表
        offline_predictions_raw = []
        offline_predictions_filtered = []
        
        # 初始化缓冲区（模拟在线推理的状态）
        input_buffer = deque(maxlen=self.input_buffer_size)
        prediction_buffer = deque(maxlen=self.prediction_buffer_size)
        
        # 逐行处理数据（模拟实时数据流）
        total_rows = len(df)
        print(f"开始流式处理 {total_rows} 行数据...")
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"处理进度: {idx}/{total_rows} ({idx/total_rows*100:.1f}%)")
            
            # 步骤1：提取传感器数据（模拟parse_sensor_data）
            sensor_data = np.array([
                row['hip_angle_l'],
                row['hip_angle_r'],
                row['hip_angle_l_velocity'],
                row['hip_angle_r_velocity']
            ], dtype=np.float32)
            
            # 步骤2：添加到输入缓冲区（模拟sensor_callback中的buffer.append）
            input_buffer.append(sensor_data)
            
            # 步骤3：获取网络输入（模拟get_network_input）
            current_size = len(input_buffer)
            if current_size < self.input_buffer_size:
                # 需要填充
                padding_size = self.input_buffer_size - current_size
                padding = np.zeros((padding_size, self.input_feature_size), dtype=np.float32)
                if current_size > 0:
                    existing_data = np.array(input_buffer, dtype=np.float32)
                    network_input = np.vstack([padding, existing_data])
                else:
                    network_input = padding
            else:
                network_input = np.array(input_buffer, dtype=np.float32)
            
            # 步骤4：数据预处理（完全复现在线推理的预处理）
            # 角度转换（与在线推理完全一致）
            network_input = (network_input * 180 / np.pi) * -1
            
            # 归一化（与在线推理完全一致）
            network_input[:, :2] = network_input[:, :2] / self.norm_stats['input_std'][0]  # 角度
            network_input[:, 2:] = network_input[:, 2:] / self.norm_stats['input_std'][1]  # 角速度
            
            # 应用速度滤波（与在线推理完全一致）
            network_input = self.apply_filter_vel(network_input)
            
            # 步骤5：准备张量并进行推理（与在线推理完全一致）
            sensor_tensor = torch.tensor(network_input, dtype=torch.float).T.unsqueeze(dim=0)
            sensor_tensor = sensor_tensor.to(self.device)
            
            # 重组数据（与在线推理完全一致）
            part1 = sensor_tensor[:, [0, 2], :]  # 左电机数据
            part2 = sensor_tensor[:, [1, 3], :]  # 右电机数据
            result = torch.cat([part1, part2], dim=0)  # 形状: (2, 2, 218)
            
            # 推理
            with torch.no_grad():
                predictions = self.model(result)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
            
            # 步骤6：处理预测结果（与在线推理完全一致）
            predictions = predictions.cpu().numpy().squeeze().T
            predictions = predictions.astype(np.float32)
            
            # 只取最后一行（与在线推理完全一致）
            last_prediction = predictions[-1] * 3  # 缩放因子3
            
            # 保存原始预测
            offline_predictions_raw.append(last_prediction.copy())
            
            # 步骤7：添加到预测缓冲区并进行滤波（与在线推理完全一致）
            prediction_buffer.append(last_prediction.copy())
            
            if len(prediction_buffer) >= 3:
                # 应用滤波器
                buffer_array = np.array(prediction_buffer)
                filtered_buffer = self.apply_filter_torque(buffer_array)
                filtered_prediction = filtered_buffer[-1]
            else:
                filtered_prediction = last_prediction
            
            offline_predictions_filtered.append(filtered_prediction.copy())
        
        print("推理完成!")
        
        # 将结果添加到DataFrame
        offline_predictions_raw = np.array(offline_predictions_raw)
        offline_predictions_filtered = np.array(offline_predictions_filtered)
        
        df['offline_torque_L_raw'] = offline_predictions_raw[:, 0]
        df['offline_torque_R_raw'] = offline_predictions_raw[:, 1]
        df['offline_torque_L_filtered'] = offline_predictions_filtered[:, 0]
        df['offline_torque_R_filtered'] = offline_predictions_filtered[:, 1]
        
        # 保存结果
        if save_results:
            output_path = csv_path.replace('.csv', '_with_offline_inference_fixed.csv')
            df.to_csv(output_path, index=False)
            print(f"结果已保存到: {output_path}")
        
        return df
    
    def visualize_comparison(self, df, save_fig=True, output_dir='./figures'):
        """
        可视化在线和离线推理结果的对比
        
        Args:
            df: 包含在线和离线推理结果的DataFrame
            save_fig: 是否保存图像
            output_dir: 图像保存目录
        """
        # 创建输出目录
        if save_fig and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建图形
        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        fig.suptitle('在线推理 vs 离线推理对比分析（修正版）', fontsize=16)
        
        # 时间轴（假设200Hz采样率）
        time_axis = np.arange(len(df)) / 200.0
        
        # 1. 输入数据展示 - 左髋角度和角速度
        ax = axes[0, 0]
        ax.plot(time_axis, df['hip_angle_l'] * 180 / np.pi, 'b-', label='角度', alpha=0.7)
        ax.set_ylabel('角度 (度)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax2 = ax.twinx()
        ax2.plot(time_axis, df['hip_angle_l_velocity'] * 180 / np.pi, 'r-', label='角速度', alpha=0.7)
        ax2.set_ylabel('角速度 (度/秒)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_xlabel('时间 (s)')
        ax.set_title('左髋输入数据')
        ax.grid(True, alpha=0.3)
        
        # 2. 输入数据展示 - 右髋角度和角速度
        ax = axes[0, 1]
        ax.plot(time_axis, df['hip_angle_r'] * 180 / np.pi, 'b-', label='角度', alpha=0.7)
        ax.set_ylabel('角度 (度)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax2 = ax.twinx()
        ax2.plot(time_axis, df['hip_angle_r_velocity'] * 180 / np.pi, 'r-', label='角速度', alpha=0.7)
        ax2.set_ylabel('角速度 (度/秒)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_xlabel('时间 (s)')
        ax.set_title('右髋输入数据')
        ax.grid(True, alpha=0.3)
        
        # 3. 左髋扭矩对比（滤波后）
        ax = axes[1, 0]
        ax.plot(time_axis, df['motor_torque_L_float'], 'b-', label='在线推理', alpha=0.7)
        ax.plot(time_axis, df['offline_torque_L_filtered'], 'r--', label='离线推理', alpha=0.7)
        if 'torque_L' in df.columns:
            ax.plot(time_axis, df['torque_L'], 'g:', label='编码器读数', alpha=0.5)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('扭矩 (Nm)')
        ax.set_title('左髋扭矩对比（滤波后）')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 右髋扭矩对比（滤波后）
        ax = axes[1, 1]
        ax.plot(time_axis, df['motor_torque_R_float'], 'b-', label='在线推理', alpha=0.7)
        ax.plot(time_axis, df['offline_torque_R_filtered'], 'r--', label='离线推理', alpha=0.7)
        if 'torque_R' in df.columns:
            ax.plot(time_axis, df['torque_R'], 'g:', label='编码器读数', alpha=0.5)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('扭矩 (Nm)')
        ax.set_title('右髋扭矩对比（滤波后）')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 左髋扭矩误差分析
        ax = axes[2, 0]
        error_L = df['motor_torque_L_float'] - df['offline_torque_L_filtered']
        ax.plot(time_axis, error_L, 'g-', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.fill_between(time_axis, error_L, 0, alpha=0.3)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('误差 (Nm)')
        ax.set_title(f'左髋扭矩误差 (MAE: {np.abs(error_L).mean():.6f}, RMSE: {np.sqrt((error_L**2).mean()):.6f})')
        ax.grid(True, alpha=0.3)
        
        # 6. 右髋扭矩误差分析
        ax = axes[2, 1]
        error_R = df['motor_torque_R_float'] - df['offline_torque_R_filtered']
        ax.plot(time_axis, error_R, 'g-', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.fill_between(time_axis, error_R, 0, alpha=0.3)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('误差 (Nm)')
        ax.set_title(f'右髋扭矩误差 (MAE: {np.abs(error_R).mean():.6f}, RMSE: {np.sqrt((error_R**2).mean()):.6f})')
        ax.grid(True, alpha=0.3)
        
        # 7. 散点图 - 左髋
        ax = axes[3, 0]
        ax.scatter(df['motor_torque_L_float'], df['offline_torque_L_filtered'], 
                  alpha=0.5, s=1)
        min_val = min(df['motor_torque_L_float'].min(), df['offline_torque_L_filtered'].min())
        max_val = max(df['motor_torque_L_float'].max(), df['offline_torque_L_filtered'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='理想线 (y=x)')
        
        # 计算R²
        correlation = np.corrcoef(df['motor_torque_L_float'], df['offline_torque_L_filtered'])[0,1]
        r_squared = correlation ** 2
        ax.set_xlabel('在线推理扭矩 (Nm)')
        ax.set_ylabel('离线推理扭矩 (Nm)')
        ax.set_title(f'左髋扭矩相关性 (R²={r_squared:.6f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # 8. 散点图 - 右髋
        ax = axes[3, 1]
        ax.scatter(df['motor_torque_R_float'], df['offline_torque_R_filtered'], 
                  alpha=0.5, s=1)
        min_val = min(df['motor_torque_R_float'].min(), df['offline_torque_R_filtered'].min())
        max_val = max(df['motor_torque_R_float'].max(), df['offline_torque_R_filtered'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='理想线 (y=x)')
        
        # 计算R²
        correlation = np.corrcoef(df['motor_torque_R_float'], df['offline_torque_R_filtered'])[0,1]
        r_squared = correlation ** 2
        ax.set_xlabel('在线推理扭矩 (Nm)')
        ax.set_ylabel('离线推理扭矩 (Nm)')
        ax.set_title(f'右髋扭矩相关性 (R²={r_squared:.6f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # 保存图像
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(output_dir, f'inference_comparison_fixed_{timestamp}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到: {fig_path}")
        
        plt.show()
        
        # 打印详细统计信息
        print("\n" + "=" * 60)
        print("推理一致性详细分析（修正版）")
        print("=" * 60)
        
        print(f"\n左髋扭矩统计:")
        print(f"  在线推理范围: [{df['motor_torque_L_float'].min():.3f}, {df['motor_torque_L_float'].max():.3f}] Nm")
        print(f"  离线推理范围: [{df['offline_torque_L_filtered'].min():.3f}, {df['offline_torque_L_filtered'].max():.3f}] Nm")
        print(f"  平均绝对误差 (MAE): {np.abs(error_L).mean():.6f} Nm")
        print(f"  最大绝对误差: {np.abs(error_L).max():.6f} Nm")
        print(f"  均方根误差 (RMSE): {np.sqrt((error_L**2).mean()):.6f} Nm")
        print(f"  相关系数: {np.corrcoef(df['motor_torque_L_float'], df['offline_torque_L_filtered'])[0,1]:.6f}")
        
        print(f"\n右髋扭矩统计:")
        print(f"  在线推理范围: [{df['motor_torque_R_float'].min():.3f}, {df['motor_torque_R_float'].max():.3f}] Nm")
        print(f"  离线推理范围: [{df['offline_torque_R_filtered'].min():.3f}, {df['offline_torque_R_filtered'].max():.3f}] Nm")
        print(f"  平均绝对误差 (MAE): {np.abs(error_R).mean():.6f} Nm")
        print(f"  最大绝对误差: {np.abs(error_R).max():.6f} Nm")
        print(f"  均方根误差 (RMSE): {np.sqrt((error_R**2).mean()):.6f} Nm")
        print(f"  相关系数: {np.corrcoef(df['motor_torque_R_float'], df['offline_torque_R_filtered'])[0,1]:.6f}")
        
        # 一致性判断
        mae_threshold = 0.01
        correlation_threshold = 0.999
        
        is_consistent_L = (np.abs(error_L).mean() < mae_threshold and 
                          np.corrcoef(df['motor_torque_L_float'], df['offline_torque_L_filtered'])[0,1] > correlation_threshold)
        is_consistent_R = (np.abs(error_R).mean() < mae_threshold and 
                          np.corrcoef(df['motor_torque_R_float'], df['offline_torque_R_filtered'])[0,1] > correlation_threshold)
        
        print(f"\n一致性判断标准:")
        print(f"  - MAE < {mae_threshold} Nm")
        print(f"  - 相关系数 > {correlation_threshold}")
        
        print(f"\n最终判断:")
        print(f"  - 左髋: {'✓ 完全一致' if is_consistent_L else '✗ 存在差异'}")
        print(f"  - 右髋: {'✓ 完全一致' if is_consistent_R else '✗ 存在差异'}")
        
        if is_consistent_L and is_consistent_R:
            print("\n🎉 恭喜！离线推理与在线推理结果完全一致！")
        else:
            print("\n⚠️ 注意：离线和在线推理结果存在细微差异，请检查数据预处理流程。")
        
        print("=" * 60 + "\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='修正版ConvTimeNet离线推理')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--model_path', type=str, 
                       default='checkpoint_and_model/best_model1021.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--norm_stats_path', type=str,
                       default='checkpoint_and_model/norm_stats_5sensors.pkl',
                       help='归一化统计数据路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='推理设备')
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存推理结果')
    parser.add_argument('--save_figures', action='store_true',
                       help='是否保存可视化图像')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='图像输出目录')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"错误: CSV文件不存在: {args.csv_path}")
        return
    
    try:
        # 创建离线推理器
        print("初始化离线推理器（修正版）...")
        inferencer = OfflineInferenceFixed(
            model_path=args.model_path,
            norm_stats_path=args.norm_stats_path,
            device=args.device
        )
        
        # 运行流式推理（完全模拟在线推理）
        print("\n开始流式离线推理...")
        df_results = inferencer.run_inference_streaming(
            csv_path=args.csv_path,
            save_results=args.save_results
        )
        
        # 可视化对比
        print("\n生成对比可视化...")
        inferencer.visualize_comparison(
            df=df_results,
            save_fig=args.save_figures,
            output_dir=args.output_dir
        )
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 如果直接运行脚本（不使用命令行参数），可以在这里设置默认参数
    import sys
    if len(sys.argv) == 1:

        
        # 直接设置参数运行
        test_csv = 'outputs_csv/hip_moment_data_20251024_160658.csv'  # 修改为您的CSV文件路径
        
        sys.argv = [
            'offline_inference_fixed.py',
            '--csv_path', test_csv,
            '--model_path', 'checkpoint_and_model/best_model1021.pth',
            '--norm_stats_path', 'checkpoint_and_model/norm_stats_5sensors.pkl',
            '--device', 'cuda',
            '--save_results',
            '--save_figures',
            '--output_dir', './figures'
        ]
    
    main()