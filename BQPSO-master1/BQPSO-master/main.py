from BQPSO import BQPSO
import pandas as pd
import matplotlib.pyplot as plt  # 添加此行
import numpy as np
def read_power_data(file_path):
    df = pd.read_excel(file_path)
    power_data = df['实际发电功率（mw）'].dropna().values  # 替换为实际发电功率所在的列名
    return power_data

file_path = 'D:/vs code/cworkplace/project/2019newdata.xls'
power_data = read_power_data(file_path)
signal = power_data[:5000]

noP = 30
noV = 2
Max_iteration = 200
gBest, gBestScore, fitness_values = BQPSO(noP, noV, Max_iteration, signal)
print(gBest)
print(gBestScore)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# 绘制适应度变化图
plt.figure(figsize=(8, 4))  # 设置图形大小
plt.plot(fitness_values, marker='o', linestyle='-', color='blue', markersize=5)

plt.xlabel('迭代次数', fontsize=14)  # x轴标签
plt.ylabel('适应度', fontsize=14)  # y轴标签
plt.title('适应度变化图', fontsize=16)  # 图表标题

# 设置y轴范围
plt.ylim(np.min(fitness_values) - 0.005, np.max(fitness_values) + 0.005)

# 添加网格
plt.grid()

# 设置x轴刻度
plt.xticks(ticks=range(1, len(fitness_values) + 1, 1))  # 每个刻度都显示
plt.yticks(fontsize=12)  # y轴刻度字体大小

plt.show()

# pdf_file_path = 'bqvmdssa2.pdf'
# plt.savefig(pdf_file_path, format='pdf')

