import numpy as np

def Qinitialization(N, L):
    # 初始化Qbit矩阵的两个分量
    Qbit = np.zeros((N, L, 2))

    # 生成均值为0，标准差为1的正态分布随机数
    Qbit[:, :, 0] = np.random.randn(N, L)  # alpha分量
    Qbit[:, :, 1] = np.random.randn(N, L)  # beta分量

    # 计算 Qbit 的振幅 |alpha|^2 + |beta|^2 = 1，归一化
    AA = np.sqrt(Qbit[:, :, 0]**2 + Qbit[:, :, 1]**2)
    Qbit[:, :, 0] = Qbit[:, :, 0] / AA
    Qbit[:, :, 1] = Qbit[:, :, 1] / AA

    return Qbit

