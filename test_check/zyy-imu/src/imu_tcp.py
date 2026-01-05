import socket
import struct
from collections import namedtuple
import datetime  # 新增导入

# IMU数据结构
IMUStruct = namedtuple('IMUData', [
    'header1', 'header2',  # 帧头2B
    'idx',                 # uint32 (4B)
    'q_x', 'q_y', 'q_z', 'q_w',  # 4×float (16B)
    'acc_x', 'acc_y', 'acc_z',   # 3×float (12B)
    'footer1', 'footer2'   # 帧尾2B
])

class IMUReceiver:
    def __init__(self, host='0.0.0.0', port=8086):
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.buffer = b''
        self.IMU_SIZE = 36  # 2+4+4+16+12+2=36字节
        self.FRAME_SIZE = 6 * self.IMU_SIZE  # 每帧216字节
        
    def start_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Server started on {self.host}:{self.port}")
        self.conn, addr = self.sock.accept()
        print(f"Client connected: {addr}")

    def receive_data(self):
        try:
            while True:
                data = self.conn.recv(4096)
                if not data:
                    print("Client disconnected")
                    break
                    
                self.buffer += data
                
                # 处理完整帧
                while len(self.buffer) >= self.FRAME_SIZE:
                    frame = self.buffer[:self.FRAME_SIZE]
                    self.buffer = self.buffer[self.FRAME_SIZE:]
                    self.process_frame(frame)
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.close()

    def parse_imu(self, imu_bytes):
        """解析36字节的IMU数据"""
        try:
            # 格式: 2B header + 1f timestamp + 1I idx + 4f quat + 3f acc + 2B footer
            unpacked = struct.unpack('<2B I 4f 3f 2B', imu_bytes)
            return IMUStruct._make(unpacked)
        except struct.error as e:
            print(f"解析失败: {e}")
            print(f"数据样本 (hex): {imu_bytes.hex(' ')}")
            print(f"预期长度: 36字节 | 实际长度: {len(imu_bytes)}字节")
            return None

    def process_frame(self, frame):
        """处理一帧数据"""
        # 获取当前时间戳（时分秒和微秒）
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
        print(f"\n=== IMU数据帧 @ {timestamp} ===")
        print("| IMU | Header | ID    | Quaternion(w,x,y,z)          | Acceleration(x,y,z)   | Footer |")
        print("|-----|--------|-------|------------------------------|-----------------------|--------|")
        
        for i in range(6):
            imu_data = frame[i*self.IMU_SIZE : (i+1)*self.IMU_SIZE]
            imu = self.parse_imu(imu_data)
            if imu:
                print(
                    f"| {i+1:2}  | {imu.header1:02X}{imu.header2:02X}  "
                    f"| {imu.idx:5}  "
                    f"| ({imu.q_w:.4f}, {imu.q_x:.4f}, {imu.q_y:.4f}, {imu.q_z:.4f}) "
                    f"| ({imu.acc_x:.4f}, {imu.acc_y:.4f}, {imu.acc_z:.4f}) "
                    f"| {imu.footer1:02X}{imu.footer2:02X} |"
                )

    def close(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()
        print("Connection closed")

if __name__ == '__main__':
    receiver = IMUReceiver()
    receiver.start_server()
    receiver.receive_data()

##############################保存数据及时间戳
import socket
import struct
from collections import namedtuple
import datetime
import csv  # 新增导入
import os   # 新增导入

# 定义IMU数据结构
IMUStruct = namedtuple('IMUData', [
    'header1', 'header2',  # 2B
    'idx',                 # uint32 (4B)
    'q_x', 'q_y', 'q_z', 'q_w',  # 4×float (16B)
    'acc_x', 'acc_y', 'acc_z',   # 3×float (12B)
    'footer1', 'footer2'   # 2B
])

class IMUReceiver:
    def __init__(self, host='0.0.0.0', port=7788):
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.buffer = b''
        self.IMU_SIZE = 36  # 2+4+4+16+12+2=36字节
        self.FRAME_SIZE = 6 * self.IMU_SIZE  # 每帧216字节
        self.csv_file = None
        self.csv_writer = None
        self.setup_csv()  # 初始化CSV文件
        
    def setup_csv(self):
        """创建CSV文件并写入表头"""
        # 创建data目录如果不存在
        os.makedirs('data', exist_ok=True)
        
        # 使用时间戳作为文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"imu_data_{timestamp}.csv"
        
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 写入CSV表头
        header = [
            'timestamp', 'imu_id', 'frame_idx',
            'q_w', 'q_x', 'q_y', 'q_z',
            'acc_x', 'acc_y', 'acc_z'
        ]
        self.csv_writer.writerow(header)
        print(f"数据将保存到: {filename}")

    def start_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Server started on {self.host}:{self.port}")
        self.conn, addr = self.sock.accept()
        print(f"Client connected: {addr}")

    def receive_data(self):
        try:
            while True:
                data = self.conn.recv(4096)
                if not data:
                    print("Client disconnected")
                    break

                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")  
                self.buffer += data
            

                # 处理完整帧
                while len(self.buffer) >= self.FRAME_SIZE:
                    frame = self.buffer[:self.FRAME_SIZE]
                    self.buffer = self.buffer[self.FRAME_SIZE:]
                    self.process_frame(frame, timestamp)  
                    
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.close()

    def parse_imu(self, imu_bytes):
        """解析36字节的IMU数据"""
        try:
            # 格式: 2B header + 1f timestamp + 1I idx + 4f quat + 3f acc + 2B footer
            unpacked = struct.unpack('<2B I 4f 3f 2B', imu_bytes)
            return IMUStruct._make(unpacked)
        except struct.error as e:
            print(f"解析失败: {e}")
            print(f"数据样本 (hex): {imu_bytes.hex(' ')}")
            print(f"预期长度: 36字节 | 实际长度: {len(imu_bytes)}字节")
            return None

    def process_frame(self, frame,timestamp):
        """处理一帧数据"""
        # 获取当前时间戳（精确到微秒）
        
        print(f"\n=== IMU数据帧 @ {timestamp} ===")
        print("| IMU | Header | ID    | Quaternion(w,x,y,z)          | Acceleration(x,y,z)   | Footer |")
        print("|-----|--------|-------|------------------------------|-----------------------|--------|")
        
        for i in range(6):
            imu_data = frame[i*self.IMU_SIZE : (i+1)*self.IMU_SIZE]
            imu = self.parse_imu(imu_data)
            if imu:
                # 打印到控制台
                print(
                    f"| {i+1:2}  | {imu.header1:02X}{imu.header2:02X}  "
                    f"| {imu.idx:5}  "
                    f"| ({imu.q_w:.4f}, {imu.q_x:.4f}, {imu.q_y:.4f}, {imu.q_z:.4f}) "
                    f"| ({imu.acc_x:.4f}, {imu.acc_y:.4f}, {imu.acc_z:.4f}) "
                    f"| {imu.footer1:02X}{imu.footer2:02X} |"
                )
                
                # 写入CSV文件
                self.write_to_csv(timestamp, i+1, imu)

    def write_to_csv(self, timestamp, imu_id, imu_data):
        """将IMU数据写入CSV文件"""
        row = [
            timestamp,      # 时间戳
            imu_id,         # IMU编号（1-6）
            imu_data.idx,   # 帧索引
            imu_data.q_w,   # 四元数q
            imu_data.q_x,
            imu_data.q_y,
            imu_data.q_z,
            imu_data.acc_x, # 加速度
            imu_data.acc_y,
            imu_data.acc_z
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # 确保数据立即写入文件

    def close(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()
        if self.csv_file:
            self.csv_file.close()
        print("Connection closed and CSV file saved")

if __name__ == '__main__':
    receiver = IMUReceiver()
    receiver.start_server()
    receiver.receive_data()




# ####################################动捕联调
# import socket
# import struct
# from collections import namedtuple
# import datetime
# import csv
# import os
# from motioncap import send_capture_start_message, send_capture_stop_message  # 导入动捕控制函数

# # 定义IMU数据结构
# IMUStruct = namedtuple('IMUData', [
#     'header1', 'header2',  # 2B
#     'idx',                 # uint32 (4B)
#     'q_x', 'q_y', 'q_z', 'q_w',  # 4×float (16B)
#     'acc_x', 'acc_y', 'acc_z',   # 3×float (12B)
#     'footer1', 'footer2'   # 2B
# ])

# class IMUReceiver:
#     def __init__(self, host='0.0.0.0', port=7788):
#         self.host = host
#         self.port = port
#         self.sock = None
#         self.conn = None
#         self.buffer = b''
#         self.IMU_SIZE = 36
#         self.FRAME_SIZE = 6 * self.IMU_SIZE
#         self.csv_file = None
#         self.csv_writer = None
#         self.capture_active = False  # 动捕采集状态标志
#         self.setup_csv()
        
#         ############################### 动捕配置参数
#         self.motion_capture_ip = '10.1.1.198'
#         self.motion_capture_port = 7060
#         self.capture_name_prefix = "IMU_Sync_"

#     def setup_csv(self):
#         """创建CSV文件并写入表头"""
#         os.makedirs('data', exist_ok=True)
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"imu_data_{timestamp}.csv"
#         self.csv_file = open(filename, 'w', newline='')
#         self.csv_writer = csv.writer(self.csv_file)
#         header = [
#             'timestamp', 'imu_id', 'frame_idx',
#             'q_w', 'q_x', 'q_y', 'q_z',
#             'acc_x', 'acc_y', 'acc_z'
#         ]
#         self.csv_writer.writerow(header)
#         print(f"数据将保存到: {filename}")

#     def start_server(self):
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         self.sock.bind((self.host, self.port))
#         self.sock.listen(1)
#         print(f"Server started on {self.host}:{self.port}")
#         self.conn, addr = self.sock.accept()
#         print(f"Client connected: {addr}")

#     def receive_data(self):
#         try:
#             while True:
#                 data = self.conn.recv(4096)
#                 if not data:
#                     print("Client disconnected")
#                     break
                    
#                 self.buffer += data
                
#                 while len(self.buffer) >= self.FRAME_SIZE:
#                     frame = self.buffer[:self.FRAME_SIZE]
#                     self.buffer = self.buffer[self.FRAME_SIZE:]
#                     self.process_frame(frame)
                    
#         except Exception as e:
#             print(f"Error: {e}")
#         finally:
#             self.close()

#     def parse_imu(self, imu_bytes):
#         """解析36字节的IMU数据"""
#         try:
#             unpacked = struct.unpack('<2B I 4f 3f 2B', imu_bytes)
#             return IMUStruct._make(unpacked)
#         except struct.error as e:
#             print(f"解析失败: {e}")
#             print(f"数据样本 (hex): {imu_bytes.hex(' ')}")
#             print(f"预期长度: 36字节 | 实际长度: {len(imu_bytes)}字节")
#             return None

#     def process_frame(self, frame):
#         """处理一帧数据（新增动捕触发逻辑）"""
#         timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
        
#         # 每次收到新帧时触发动捕采集
#         if not self.capture_active:
#             self.start_motion_capture(timestamp)
        
#         print(f"\n=== IMU数据帧 @ {timestamp} ===")
#         print("| IMU | Header | ID    | Quaternion(w,x,y,z)          | Acceleration(x,y,z)   | Footer |")
#         print("|-----|--------|-------|------------------------------|-----------------------|--------|")
        
#         for i in range(6):
#             imu_data = frame[i*self.IMU_SIZE : (i+1)*self.IMU_SIZE]
#             imu = self.parse_imu(imu_data)
#             if imu:
#                 print(
#                     f"| {i+1:2}  | {imu.header1:02X}{imu.header2:02X}  "
#                     f"| {imu.idx:5}  "
#                     f"| ({imu.q_w:.4f}, {imu.q_x:.4f}, {imu.q_y:.4f}, {imu.q_z:.4f}) "
#                     f"| ({imu.acc_x:.4f}, {imu.acc_y:.4f}, {imu.acc_z:.4f}) "
#                     f"| {imu.footer1:02X}{imu.footer2:02X} |"
#                 )
#                 self.write_to_csv(timestamp, i+1, imu)

#     def start_motion_capture(self, timestamp):
#         """启动动捕采集"""
#         capture_name = f"{self.capture_name_prefix}{timestamp.replace(':', '-')}"
#         send_capture_start_message(
#             self.motion_capture_ip,
#             self.motion_capture_port,
#             name_value=capture_name
#         )
#         self.capture_active = True
#         print(f"已触发动捕采集: {capture_name}")

#     def stop_motion_capture(self):
#         """停止动捕采集"""
#         if self.capture_active:
#             send_capture_stop_message(
#                 self.motion_capture_ip,
#                 self.motion_capture_port
#             )
#             self.capture_active = False
#             print("已停止动捕采集")

#     def write_to_csv(self, timestamp, imu_id, imu_data):
#         """将IMU数据写入CSV文件"""
#         row = [
#             timestamp, imu_id, imu_data.idx,
#             imu_data.q_w, imu_data.q_x, imu_data.q_y, imu_data.q_z,
#             imu_data.acc_x, imu_data.acc_y, imu_data.acc_z
#         ]
#         self.csv_writer.writerow(row)
#         self.csv_file.flush()

#     def close(self):
#         self.stop_motion_capture()  # 程序退出时停止采集
#         if self.conn:
#             self.conn.close()
#         if self.sock:
#             self.sock.close()
#         if self.csv_file:
#             self.csv_file.close()
#         print("Connection closed and CSV file saved")

# if __name__ == '__main__':
#     receiver = IMUReceiver()
#     try:
#         receiver.start_server()
#         receiver.receive_data()
#     except KeyboardInterrupt:
#         print("\n用户中断程序")


