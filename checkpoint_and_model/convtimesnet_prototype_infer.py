#!/usr/bin/env python
"""
Hip Moment Predictor Node
用于接收传感器数据，进行推理，并发布预测结果
"""

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Float32MultiArray
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from hip_tcn import HipMomentTCN
import time
from scipy import signal as scipy_signal
from collections import deque
from model.ConvTimeNet_backbone import ConvTimeNet_backbone
import pickle
import csv
import signal
import sys
from datetime import datetime
import torch.nn.functional as F

class HipMomentPredictor:
    def __init__(self, model_path, device='cuda'):
        # 初始化ROS节点
        rospy.init_node('hip_moment_predictor', anonymous=True)
        
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        # 加载模型
        self.model = self.load_model(model_path)
        
        # 创建发布器 - 发布预测的力矩（只发布最后一行）
        self.prediction_pub = rospy.Publisher('/moment', Float32MultiArray, queue_size=10)
        
        # 创建订阅器 - 接收传感器数据
        rospy.Subscriber('/wgg_msg', Float32MultiArray, self.sensor_callback)
        
        # 统计信息
        self.prediction_count = 0
        self.total_inference_time = 0
        
        # 初始化网络输入缓冲区（存储186帧，每帧4个特征）
        self.input_buffer_size = 218
        self.input_feature_size = 4
        self.input_buffer = deque(maxlen=self.input_buffer_size)
        
        # 初始化预测缓冲区（存储最近30帧）
        self.prediction_buffer_size = 30
        self.prediction_buffer = deque(maxlen=self.prediction_buffer_size)
        
        # 滤波器参数
        self.cutoff_freq = 5.0  # 截止频率 10 Hz
        self.sampling_rate = 200.0  # 采样率 200 Hz
        self.filter_order = 2  # 二阶滤波器
        
        # 设计巴特沃斯滤波器
        nyquist_freq = self.sampling_rate / 2
        normalized_cutoff = self.cutoff_freq / nyquist_freq
        self.b, self.a = scipy_signal.butter(self.filter_order, normalized_cutoff, btype='low', analog=False)
        
        # 数据记录缓冲区
        self.data_records = []  # 存储所有的数据记录
        
        # 注册信号处理器以便在程序退出时保存数据
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        rospy.loginfo("Hip Moment Predictor initialized successfully")
        rospy.loginfo(f"Using input buffer size: {self.input_buffer_size} frames x {self.input_feature_size} features")
        rospy.loginfo(f"Filter parameters - Cutoff: {self.cutoff_freq}Hz, Sampling rate: {self.sampling_rate}Hz")
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        # 注意：这里需要确认模型的input_size是否需要从10改为4
        model = ConvTimeNet_backbone(
            c_in=2,
            n_layers=4,
            seq_len=218,
            context_window = 218,
            target_window = 218,
            patch_len= 32,
            stride=16,
            d_model=64,
            d_ff=128,
            dropout=0.2,
            act="gelu",
            enable_res_param=False,
            dw_ks=[5,5,7,7,13,13,19,19,],  # Depth-wise kernel sizes for each layer
            norm='batch',
            re_param=False,
            deformable=True,
            reduced_channels=16,
            revin = False,
            final_out=1,
        ).to(self.device)
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        inp = torch.randn(2,2,218)
        # model = torch.jit.trace(model, inp)

        model = model.to(self.device)
        model.eval()
        
        rospy.loginfo(f"Model loaded from: {model_path}")
        return model
        
    def parse_sensor_data(self, msg):
        """
        解析Float32MultiArray消息,取前4个数据
        """
        # 从一维数据中取前4个值
        if len(msg.data) < 4:
            rospy.logwarn(f"Received data has less than 4 values: {len(msg.data)}")
            return None
        
        sensor_data = np.array(msg.data[:4], dtype=np.float32)
        enc_torque = np.array(msg.data[-2:], dtype=np.float32) / 100
        return sensor_data, enc_torque
    
    def get_network_input(self):
        """
        获取网络输入数据，形状为(186, 4)
        如果数据不足186帧, 用零向量填充前面的部分
        """
        current_size = len(self.input_buffer)
        
        if current_size < self.input_buffer_size:
            # 需要填充
            padding_size = self.input_buffer_size - current_size
            # 创建零填充
            padding = np.zeros((padding_size, self.input_feature_size), dtype=np.float32)
            # 将现有数据转换为numpy数组
            if current_size > 0:
                existing_data = np.array(self.input_buffer, dtype=np.float32)
                # 拼接：零填充在前，实际数据在后
                network_input = np.vstack([padding, existing_data])
            else:
                network_input = padding
        else:
            # 缓冲区已满，直接使用
            network_input = np.array(self.input_buffer, dtype=np.float32)
        
        return network_input
        
    def create_multiarray_msg(self, data):
        """
        创建Float32MultiArray消息
        现在只处理一维数据（最后两个力矩值）
        """
        msg = Float32MultiArray()
        
        # 设置维度信息（一维数组）
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "joint_moments"
        msg.layout.dim[0].size = len(data)
        msg.layout.dim[0].stride = len(data)
        
        # 填充数据
        msg.data = data.tolist()
        
        return msg
    
    def apply_filter(self, buffer_array):
        """
        对缓冲区数据应用巴特沃斯滤波器
        buffer_array: shape (n, 2), n <= 30
        """
        if buffer_array.shape[0] < 3:  # filtfilt需要至少3个样本点
            return buffer_array
        
        filtered_data = np.zeros_like(buffer_array)
        
        # 对左右髋关节力矩分别进行滤波
        try:
            # 滤波左髋关节力矩
            filtered_data[:, 0] = scipy_signal.filtfilt(self.b, self.a, buffer_array[:, 0])
            # 滤波右髋关节力矩
            filtered_data[:, 1] = scipy_signal.filtfilt(self.b, self.a, buffer_array[:, 1])
        except Exception as e:
            rospy.logwarn(f"Filter failed, using original data: {e}")
            return buffer_array
        
        return filtered_data

    def apply_filter_vel(self, buffer_array):
        """
        对缓冲区数据应用巴特沃斯滤波器
        buffer_array: shape (n, 4), n <= 30
        """
        if buffer_array.shape[0] < 3:  # filtfilt需要至少3个样本点
            return buffer_array
        
        filtered_data = np.zeros_like(buffer_array)
        
        # 对左右髋关节力矩分别进行滤波
        try:
            filtered_data[:, 0] = buffer_array[:, 0]
            filtered_data[:, 1] = buffer_array[:, 1]
            
            # 滤波左电机角速度
            filtered_data[:, 2] = scipy_signal.filtfilt(self.b, self.a, buffer_array[:, 2])
            # 滤波右电机角速度
            filtered_data[:, 3] = scipy_signal.filtfilt(self.b, self.a, buffer_array[:, 3])
        except Exception as e:
            rospy.logwarn(f"Filter failed, using original data: {e}")
            return buffer_array
        
        return filtered_data
        
    def signal_handler(self, sig, frame):
        """处理退出信号，保存数据到CSV"""
        rospy.loginfo("\nReceived interrupt signal, saving data to CSV...")
        self.save_data_to_csv()
        rospy.loginfo("Data saved successfully. Exiting...")
        sys.exit(0)
    
    def save_data_to_csv(self):
        """将记录的数据保存到CSV文件"""
        if not self.data_records:
            rospy.logwarn("No data to save.")
            return
        
        # 生成带时间戳的文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs_csv/hip_moment_data_{timestamp_str}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                # 定义CSV列名
                fieldnames = [
                    'timestamp',
                    'hip_angle_l', 
                    'hip_angle_r',
                    'hip_angle_l_velocity', 
                    'hip_angle_r_velocity',
                    'motor_torque_L_float', 
                    'motor_torque_R_float',
                    'torque_L',
                    'torque_R'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # 写入所有记录
                for record in self.data_records:
                    writer.writerow(record)
            
            rospy.loginfo(f"Saved {len(self.data_records)} records to {filename}")
            
        except Exception as e:
            rospy.logerr(f"Error saving data to CSV: {e}")
            import traceback
            traceback.print_exc()
        
    def sensor_callback(self, msg):
        """处理传感器数据并进行预测"""
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 解析传感器数据（取前4个值）
            sensor_data,torque = self.parse_sensor_data(msg)
            if sensor_data is None:
                return
            
            # 将新数据添加到输入缓冲区
            self.input_buffer.append(sensor_data)
            
            # 获取网络输入（186, 4）
            network_input = self.get_network_input()

            network_input = (network_input*180/np.pi) * -1

            network_input[:, :2] = (network_input[:,:2] ) / norm_stats['input_std'][0]  # 0-2是角度数据
            network_input[:, 2:] = (network_input[:, 2:] ) / norm_stats['input_std'][1] # 2-4是角速度数据
            # TODO： 归一化需要在外面做
            # network_input[:,:2] = network_input[:,:2] - 180
            # network_input = (network_input) * np.pi / 180
            # network_input *= -1

            network_input = self.apply_filter_vel(network_input)

            # 准备数据并传输到GPU
            # 注意：需要确认模型期望的输入形状
            # 假设模型期望输入形状为 (batch_size, features, time_steps)
            sensor_tensor = torch.tensor(network_input, dtype=torch.float).T.unsqueeze(dim=0)
            # sensor_tensor = F.interpolate(sensor_tensor, size=218, mode='linear', align_corners=False)

            sensor_tensor = sensor_tensor.to(self.device)

            part1 = sensor_tensor[:, [0, 2], :]  # 左电机数据形状: (1, 2, 218)
            part2 = sensor_tensor[:, [1, 3], :]  # 右电机形状: (1, 2, 218)

            # 沿着第1维连接，然后重塑
            result = torch.cat([part1, part2], dim=0)  # 形状: (1, 4, 218)
            # result = result.view(2,2,218)
            # rospy.loginfo(f"sensor_tensor;{result.shape}")
            # rospy.loginfo(f"sensor_tensor;{result[0,0,-1:]}")
            
            # 进行推理
            with torch.no_grad():
                predictions = self.model(result)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()  # 确保GPU计算完成
            
            # 将预测结果传回CPU并转换为float32
            predictions = predictions.cpu().numpy().squeeze().T
            predictions = predictions.astype(np.float32)  # 确保是float32类型
            # rospy.loginfo(f"predictions;{predictions.shape}")

            
            # 只取最后一行（最新的预测结果）
            last_prediction = predictions[-1] * 5 # 形状应为 (2,)
            rospy.loginfo(f"last prediction data: {last_prediction}")
            
            # 将新的预测添加到预测缓冲区
            self.prediction_buffer.append(last_prediction.copy())
            
            # 获取要发布的数据
            if len(self.prediction_buffer) >= 3:  # 有足够的数据进行滤波
                # 将缓冲区转换为numpy数组
                buffer_array = np.array(self.prediction_buffer)  # shape: (n, 2), n <= 30
                
                # 应用滤波器
                filtered_buffer = self.apply_filter(buffer_array)
                
                # 获取滤波后的最新数据（最后一行）
                filtered_last_prediction = filtered_buffer[-1]
                
                # 发布滤波后的数据
                prediction_to_publish = filtered_last_prediction
            else:
                # 缓冲区数据不足，直接发布原始数据
                prediction_to_publish = last_prediction
                rospy.loginfo_once("Prediction buffer filling, using unfiltered data temporarily...")
            
            # 计算推理时间
            inference_time = time.time() - start_time
            self.prediction_count += 1
            self.total_inference_time += inference_time
            
            # 每100次预测打印一次统计信息
            if self.prediction_count % 100 == 0:
                avg_time = (self.total_inference_time / self.prediction_count) * 1000
                rospy.loginfo(f"Predictions: {self.prediction_count}, "
                            f"Avg inference time: {avg_time:.2f}ms, "
                            f"Input buffer: {len(self.input_buffer)}/{self.input_buffer_size}, "
                            f"Prediction buffer: {len(self.prediction_buffer)}/{self.prediction_buffer_size}")
                rospy.loginfo(f"Latest filtered prediction: {prediction_to_publish}")
                rospy.loginfo(f"Latest sensor data: {sensor_data}")
            
            # 发布预测结果（滤波后的最后两个力矩值）
            prediction_msg = self.create_multiarray_msg(prediction_to_publish)

            self.prediction_pub.publish(prediction_msg)
            
            # 记录数据用于后续保存到CSV
            data_record = {
                'timestamp': time.time(),
                'hip_angle_l': float(sensor_data[0]),
                'hip_angle_r': float(sensor_data[1]),
                'hip_angle_l_velocity': float(sensor_data[2]),
                'hip_angle_r_velocity': float(sensor_data[3]),
                'motor_torque_L_float': float(prediction_to_publish[0]),
                'motor_torque_R_float': float(prediction_to_publish[1]),
                'torque_L':float(torque[0]),
                'torque_R':float(torque[1]),
            }
            self.data_records.append(data_record)
            
        except Exception as e:
            rospy.logerr(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            
    def run(self):
        """主循环"""
        rospy.loginfo("Hip Moment Predictor is running...")
        rospy.loginfo("Waiting for sensor data on /wgg_msg topic...")
        
        # 定期打印状态
        rate = rospy.Rate(0.1)  # 10秒一次
        while not rospy.is_shutdown():
            if self.prediction_count > 0:
                avg_time = (self.total_inference_time / self.prediction_count) * 1000
                rospy.loginfo(f"Status - Total predictions: {self.prediction_count}, "
                            f"Average inference time: {avg_time:.2f}ms, "
                            f"Input buffer: {len(self.input_buffer)}/{self.input_buffer_size} frames, "
                            f"Prediction buffer: {len(self.prediction_buffer)}/{self.prediction_buffer_size} frames")
            rate.sleep()

if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = r'checkpoint_and_model/best_model1021.pth'
    DEVICE = 'cuda'  # 或 'cpu'
    path = 'checkpoint_and_model/norm_stats_5sensors.pkl'
    
    global norm_stats
    with open(path, 'rb') as f:
        norm_stats = pickle.load(f)
    
    try:
        predictor = HipMomentPredictor(model_path=MODEL_PATH, device=DEVICE)
        predictor.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()