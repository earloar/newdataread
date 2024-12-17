import numpy as np
from Qinitialization import Qinitialization
from Qobservation import Qobservation
from Qmove import Qmove
from MyCost import MyCost
import random
from scipy.stats import levy

def gliding_distance():
        lift = 0.9783723933835806 / random.uniform(0.675, 1.5)
        drag = 1.630620655639301
        return 8.0 / (18 * drag / lift)
    
def Qinitialization_single():
    # 生成均值为0，标准差为1的正态分布随机数，用于初始化alpha和beta
    r = 1  # 振幅是1，确保归一化
    theta = np.random.uniform(0, 2 * np.pi)  # 随机选择角度

    # 将极坐标转换为笛卡尔坐标：alpha和beta
    alpha = r * np.cos(theta)
    beta = r * np.sin(theta)

    # 组合成一个量子比特
    qbit = np.array([alpha, beta])

    return qbit  # 返回量子比特
    
def calculate_delta_theta(qbit2, qbit1):
    """
    计算 delta_theta，即量子比特 qbit1 和 qbit2 之间的角度变化。
    
    :param qbit1: 量子比特 1, 包含 [α, β]
    :param qbit2: 量子比特 2, 包含 [α', β']
    :return: 计算得到的 delta_theta
    """
    alpha, beta = qbit1
    alpha_prime, beta_prime = qbit2

    # 计算 cos(theta) 和 sin(theta)
    cos_theta = (alpha_prime * alpha + beta_prime * beta) / (alpha**2 + beta**2)
    sin_theta = (beta_prime * alpha - alpha_prime * beta) / (alpha**2 + beta**2)
    # 计算角度 theta
    delta_theta = np.arctan2(sin_theta, cos_theta)

    return delta_theta
    
def BQPSO(N, dim, max_it, signal):
    Qbit = Qinitialization(N, dim)
    V = np.zeros((N, dim))
    theta_min = 0.005 * np.pi
    theta_max = 0.1 * np.pi
    
    best_fitness_values = []
    for it in range(1, max_it + 1):
        if it == 1:
            delta_theta = np.zeros((N, dim))
            X1 = Qobservation(Qbit, N, dim, 1)
            X2 = Qobservation(Qbit, N, dim, 2)
            if MyCost(X1[0], signal) > MyCost(X2[0], signal):
                X = X2
            else:
                X = X1
            fitness = np.array([MyCost(X[aa], signal) for aa in range(N)])
            sorted_indices = np.argsort(fitness)
            X = X[sorted_indices]
            Qbit = Qbit[sorted_indices]
        # 声明松鼠在不同树上的分布
        n1 = 1  # 山核桃树上的松鼠数量
        n2 = 3  # 橡树上的松鼠数量
        n3 = 16   # 普通树上的松鼠数量
        g_c = 1.9
        for i in range(n1, N):
            for j in range(dim):
                R1 = random.random()  # 更新橡树和普通树上的松鼠
                if R1 >= 0.2:
                    if i < n1 + n2:  # 橡树上的松鼠
                        delta_theta[i][j] = gliding_distance() * g_c * calculate_delta_theta(Qbit[0][j], Qbit[i][j])
                    elif i < n1 + n2 + n3 / 2:  # 普通树上的松鼠
                        delta_theta[i][j] = gliding_distance() * g_c * calculate_delta_theta(Qbit[random.randint(1, n2)][j], Qbit[i][j])
                    else:
                        delta_theta[i][j] = gliding_distance() * g_c * calculate_delta_theta(Qbit[0][j], Qbit[i][j])
                else:
                    Qbit[i][j] = Qinitialization_single()
                    delta_theta[i][j] = 0
        # 计算季节常数
        sum = 0
        for i in range(1, n1 + n2):
            for j in range(0, 2):
                sum = sum + (X[0][j] - X[i][j]) ** 2

        Sc = np.sqrt(sum)
        Smin = 1e-5 / (365 ** (2.5 * it / max_it))
        
        # 季节性监测条件
        if Sc < Smin:
            print(1)
            # 随机重新定位松鼠
            for i in range(N - n3, N):
                for j in range(dim):
                    delta_theta[i][j] = theta_min + levy.rvs(size=1) * (theta_max - theta_min)
                    
        Qbit = Qmove(Qbit, N, dim, delta_theta)
        X1 = Qobservation(Qbit, N, dim, 1)
        X2 = Qobservation(Qbit, N, dim, 2)
        if MyCost(X1[0], signal) > MyCost(X2[0], signal):
            X = X2
        else:
            X = X1
        fitness = np.array([MyCost(X[aa], signal) for aa in range(N)])
        sorted_indices = np.argsort(fitness)
        Qbit = Qbit[sorted_indices]
        delta_theta = delta_theta[sorted_indices]
        X = X[sorted_indices]
        best_fitness_values.append(MyCost(X[0], signal))
        print("第", it, "次迭代，最优解：", X[0], " 适应度：", MyCost(X[0], signal))

    return X[0], MyCost(X[0], signal), best_fitness_values