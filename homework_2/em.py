import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 读取身高数据
df = pd.read_csv('height_data.csv')

# 取出身高数据并进行标准化
X = df['height'].values
data_mean = np.mean(X)
data_std = np.std(X)
X = (X - data_mean) / data_std

# 初始化模型参数
K = 2
N = len(X)
mu = np.random.rand(K) * 2 - 1
sigma = np.ones(K)
pi = np.ones(K) / K   # 初始化每个高斯分布的权重

# 定义高斯分布密度函数
def gaussian(X, mu, sigma):
    return np.exp(-(X - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

# E步
def E_step(X, pi, mu, sigma):
    r = np.zeros((N, K))
    for k in range(K):
        r[:, k] = pi[k] * gaussian(X, mu[k], sigma[k])
    r = r / np.sum(r, axis=1, keepdims=True)
    return r

# M步
def M_step(X, r):
    K = r.shape[1]
    pi = np.mean(r, axis=0)
    mu = np.zeros(K)
    sigma = np.zeros(K)
    for k in range(K):
        mu[k] = np.sum(r[:, k] * X) / np.sum(r[:, k])
        sigma[k] = np.sqrt(np.sum(r[:, k] * (X - mu[k])**2) / np.sum(r[:, k]))
    return pi, mu, sigma

# EM算法
def EM(X, pi, mu, sigma, max_iter=100, tol=1e-4):
    log_likelihood = []
    for i in range(max_iter):
        r = E_step(X, pi, mu, sigma)
        pi, mu, sigma = M_step(X, r)
        ll = np.sum(np.log(np.sum(pi[k] * gaussian(X, mu[k], sigma[k]) for k in range(K))))
        log_likelihood.append(ll)
        if i > 0 and np.abs(ll - log_likelihood[-2]) < tol:
            break
    return pi, mu, sigma, log_likelihood

# 运行EM算法
pi, mu, sigma, log_likelihood = EM(X, pi, mu, sigma)

# 打印结果
print('pi:', pi)
print('mu:', mu * data_std + data_mean)
print('sigma:', sigma * data_std)

# 绘制身高数据分布和估计结果
plt.hist(X, bins=50, density=True, alpha=0.5, color='blue')
x = np.linspace(-4, 4, 2000)
y = sum(pi[k] * gaussian(x, mu[k], sigma[k]) for k in range(K))
plt.plot(x, y, color='red')
plt.show()
