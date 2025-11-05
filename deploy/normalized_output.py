import pandas as pd
import numpy as np


def normalize_column(data):
    """
    对数据进行Min-Max归一化到[0, 1]区间
    公式: (x - min) / (max - min)
    """
    min_val = data.min()
    max_val = data.max()

    if max_val == min_val:
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)


# 读取CSV文件
df = pd.read_csv('1-8walk.csv')

# 获取前两列的列名
col1 = df.columns[0]
col2 = df.columns[1]

print(f"归一化前的数据:")
print(df[[col1, col2]].head())
print(f"\n{col1} 范围: [{df[col1].min():.6f}, {df[col1].max():.6f}]")
print(f"{col2} 范围: [{df[col2].min():.6f}, {df[col2].max():.6f}]")

# 对前两列进行归一化
df[f'{col1}_normalized'] = normalize_column(df[col1])
df[f'{col2}_normalized'] = normalize_column(df[col2])

print(f"\n归一化后的数据:")
print(df[[col1, f'{col1}_normalized', col2, f'{col2}_normalized']].head())

# 保存结果到新的CSV文件
df.to_csv('normalized_output.csv', index=False)
print(f"\n结果已保存到 'normalized_output.csv'")

# 如果你想替换原列而不是新增列，使用以下代码:
# df[col1] = normalize_column(df[col1])
# df[col2] = normalize_column(df[col2])