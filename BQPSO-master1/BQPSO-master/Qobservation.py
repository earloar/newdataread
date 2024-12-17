import numpy as np

def Qobservation(Qbit, N, dim, cnt):
    """
    量子比特解码到实际变量范围
    :param Qbit: 量子比特矩阵
    :param N: 搜索代理数量
    :param dim: 变量维度
    :param ranges: 变量的实际范围 [[min1, max1], [min2, max2], ...]
    :return: 解码后的实际位置矩阵
    """
    ranges = [[3, 15], [100, 2500]]
    # ranges = [[0, 10], [0, 10]]
    X = np.zeros((N, dim))

    # 解码量子比特，将 [-1, 1] 的量子比特状态转换为实际范围
    if cnt == 1:
        for j in range(dim):
            min_val = ranges[j][0]
            max_val = ranges[j][1]
            X[:, j] = (((Qbit[:, j, 1] + Qbit[:, j, 0]) / 2) ** 2) * (max_val - min_val) + min_val
    if cnt == 2:
        for j in range(dim):
            min_val = ranges[j][0]
            max_val = ranges[j][1]
            X[:, j] = (Qbit[:, j, 1] ** 2) * (max_val - min_val) + min_val

    return X