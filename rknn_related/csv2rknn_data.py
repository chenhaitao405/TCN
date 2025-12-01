import os
import pandas as pd
import numpy as np
from collections import deque


def main():
    csv_path = "./deploy/kneeData_left.csv"
    save_path = "./deploy/datasets"
    txt_path = "./deploy/kneedata_left.txt"
    want_sample_num = 100
    window_size = 248
    data_queue = deque(maxlen=window_size)
    
    if not os.path.exists(csv_path):
        print(f"警告: CSV文件不存在: {csv_path}")
        # 导入csv文件
        exit()
    df = pd.read_csv(csv_path)
    required_columns = ['gyro_x', 'gyro_y', 'gyro_z',
                        'acc_x', 'acc_y', 'acc_z',
                        'motorPos', 'motorVel']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必需的列: {col}")
    
    data = df[required_columns].to_numpy()
    
    sample_interval = (len(data) + 1 - window_size) // want_sample_num 
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    count = 0
    with open(txt_path, "w") as f:
        for idx, sensor_input in enumerate(data):
            if len(data_queue) >= 248:
                data_queue.popleft()
            data_queue.append(sensor_input.reshape(1,8))
            
            if len(data_queue) == 248:
                if count % sample_interval == 0:
                    file_name = str(idx) + ".npy"
                    file_path = os.path.join(save_path, file_name)
                    save_data = np.stack(data_queue, axis=-1)
                    np.save(file_path, save_data)
                    f.write(os.path.join("./dataset", file_name) + "\n")
                count += 1
            
        
    
if __name__ == "__main__":
    main()
    
    
    