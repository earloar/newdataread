import numpy as np
from vmdpy import VMD
import pandas as pd
from scipy.signal import hilbert

def calculate_envelope_entropy(imf):
    K = imf.shape[0]
    features = np.zeros(K)

    for i in range(K):
        # 对分解得到的 IMF 分量进行希尔伯特变换，得到幅值
        amp = np.abs(hilbert(imf[i, :]))
        amp_n = amp / np.sum(amp)

        # 计算包络熵
        env_e = -np.sum(amp_n[amp_n > 0] * np.log(amp_n[amp_n > 0]))  # 避免log(0)
        features[i] = env_e

    fitness = np.min(features)  # 适应度函数值，目的就是使该值最小
    return fitness

def MyCost(X, signal):
    K, alpha = int(X[0]), int(X[1])
    # x1, x2 = X[0],X[1]
    # print(K, alpha)
    u, _, _ = VMD(signal, alpha, 0, K, 0, 1, 1e-7)
    # return calculate_envelope_entropy(u)
    fitness = np.zeros(K)
    for i in range(K):
        # 计算 Hilbert 包络
        xx = np.abs(hilbert(u[i, :]))  # 计算 Hilbert 包络
        
        # 归一化
        xxx = xx / np.sum(xx)
        
        # 计算最小包络熵
        ssum = 0
        for ii in range(len(xxx)):
            if xxx[ii] > 0:  # 避免 log(0) 的情况
                bb = xxx[ii] * np.log(xxx[ii])
                ssum += bb
        
        fitness[i] = -ssum

    # 找到最小的 fitness 值
    ff = np.min(fitness)
    return ff
    # dim = len(X)  # 获取 X 的长度，dim 为维度
    # o = -20 * np.exp(-0.2 * np.sqrt(np.sum(X**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * X)) / dim) + 20 + np.exp(1)
    # return o
    # o = np.sum(np.abs(X)) + np.prod(np.abs(X))
    # return o
    # o = np.sum(-X * np.sin(np.sqrt(np.abs(X))))
    # return o
    # dim = 2  # 将 dim 设置为 2
    # return np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X)) + 10 * dim
    # dim = len(X)  # 获取 X 的长度，dim 为维度
    # o = -20 * np.exp(-0.2 * np.sqrt(np.sum(X**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * X)) / dim) + 20 + np.exp(1)
    # return o
