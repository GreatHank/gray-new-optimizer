import numpy as np

# 强度数据
intensities = np.array([
0.262632,
0.224812,
0.238922,
0.262562,
0.258145,
0.245948,
0.296980,
0.244278,
0.249336

















])

# 原始权重
custom_weights = (np.array
(
[[10.78, 5.84, 7.87, 5.3, 4.72, 15.05, 7.0, 3.33, 6.22]]
))

# 归一化补偿公式
max_intensity = intensities.max()
new_weights = custom_weights * (max_intensity / intensities)

# 保留两位小数
new_weights = np.round(new_weights, 2)

print(new_weights.tolist())
