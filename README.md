## 膝关节
### 数据协议
```
数据类型：Float64MultiArray
上传数据：
       raw_data = {
    'motorPos': msg.data[0],
    'motorVel': msg.data[1],
    'acc_x': msg.data[2],
    'acc_y': msg.data[3],
    'acc_z': msg.data[4],
    'gyro_x': msg.data[5],
    'gyro_y': msg.data[6],
    'gyro_z': msg.data[7],
    'moment': msg.data[8],  # 返回值（用于对比）
    'timestamp_sensor': msg.data[9] if len(msg.data) > 9 else 0,
    'timestamp_back': msg.data[10] if len(msg.data) > 10 else 0,
    }

下发数据：
    [力矩，时间戳]

```

## 髋关节
### 数据协议
```
数据类型：Float64MultiArray

        sensor_data = np.array(msg.data[:4], dtype=np.float32)
        enc_torque = np.array(msg.data[-2:], dtype=np.float32) / 100


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

```

        # 创建发布器 - 发布预测的力矩（只发布最后一行）
        self.prediction_pub = rospy.Publisher('/moment', Float32MultiArray, queue_size=10)
        
        # 创建订阅器 - 接收传感器数据
        rospy.Subscriber('/wgg_msg', Float32MultiArray, self.sensor_callback)
### 运行时数据流
```
1. ROS订阅器接收传感器数据
   ↓
2. 触发 sensor_callback_hip() 回调
   ↓
3. 数据预处理：
   - 弧度转角度
   - 映射到标准格式
   ↓
4. 调用 engine.process_frame_hip()
   - 分别缓存左右侧数据
   - 构建双侧输入张量
   - 执行模型推理
   ↓
5. 后处理：
   - 应用巴特沃斯滤波（如启用）
   - 应用非线性滤波（如启用）
   ↓
6. PublishWorker.run_hip() 发布：
   - 使用 Float32MultiArray
   - 发布频率 100Hz
   - 不包含时间戳
```



