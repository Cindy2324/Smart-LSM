import numpy as np
import matplotlib.pyplot as plt

# 更新数据
operations = ["PUT (100)", "GET (100)", "DEL (100)", "PUT (1M)", "GET (1M)", "DEL (1M)"]
throughput = [207307, 4245560, 845266, 98966, 4909830, 4212730]
latency = [4.82e-06, 2.36e-07, 1.18e-06, 1.01e-05, 2.04e-07, 2.37e-07]

# 画图
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左 y 轴：吞吐量
ax1.bar(operations, throughput, color=['blue', 'green', 'red', 'blue', 'green', 'red'])
ax1.set_ylabel("Throughput (ops/sec)", color="black")
ax1.set_xlabel("Operation")
ax1.set_title("Performance of LSM-KVStore")

# 右 y 轴：平均时延
ax2 = ax1.twinx()
ax2.plot(operations, latency, color="black", marker="o", linestyle="dashed", label="Latency")
ax2.set_ylabel("Latency (sec/op)", color="black")
ax2.set_yscale("log")  # 使用对数坐标轴，突出差异

# 显示图例
fig.tight_layout()
plt.show()