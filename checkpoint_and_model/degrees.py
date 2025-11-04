import pandas as pd
import numpy as np

# 读取CSV文件
input_file = 'hip1_work.csv'  # 修改为你的CSV文件路径
output_file = 'hip1_work_degree.csv'  # 输出文件路径

# 读取数据
df = pd.read_csv(input_file)

# 将第2-5列从弧度转换为角度
# 列索引：1, 2, 3, 4 (从0开始计数)
columns_to_convert = ['hip_angle_l', 'hip_angle_r', 'hip_angle_l_velocity', 'hip_angle_r_velocity']

for col in columns_to_convert:
    df[col] = np.degrees(df[col])

# 保存转换后的数据
df.to_csv(output_file, index=False)

print(f"转换完成！")
print(f"已将以下列从弧度转换为角度：{columns_to_convert}")
print(f"结果已保存到：{output_file}")

# 显示前几行数据预览
print("\n转换后的数据预览：")
print(df.head())