

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv(r'G_v8d0_h5d0_c0d2_0.csv')

# 输出所有列名，检查是否有 'coherence' 列及 D_20220109 到 D_20221223 列
print("CSV 文件的列名：", df.columns)

# 保留 'coherence' 列到小数点后15位
df['coherence'] = df['coherence'].round(15)

# 筛选出 'coherence' 列大于 0.3 的数据
filtered_df = df[df['coherence'] > 0.4]

# 提取 'D_20220109' 到 'D_20221223' 的列
d_columns = [col for col in df.columns if col.startswith('D_2022')]

# 提取符合条件的数据（筛选出的 D 列与 'coherence' 列）
result_df = filtered_df[d_columns + ['coherence']]

#行列互换
result_transposed = result_df.T
print(f"提取后的数据共有 {result_transposed.shape[1]} 列数据")
# 保存结果到新文件
result_transposed.to_csv('transposed_result.csv', header=False)

# result_df.to_csv('transposed_result.csv', header=False)

print("处理完成，结果已保存为 'transposed_result.csv'")
