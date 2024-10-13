---
single: post
title:  "高斯分布的数学推导详解"
date:   2024-10-12 22:50:22 +1000
categories: 
    - machine-learning
author: Hui
---


## 引言
高斯分布，又称正态分布，是概率论和统计学中最重要的分布之一。

## 高斯分布的定义
高斯分布的概率密度函数（PDF）形式为：

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

其中，$\mu$ 为均值，$\sigma^2$ 为方差。

## 高斯分布的推导

### 从最大熵原理推导
在已知均值和方差的情况下，寻找满足这些条件的概率密度函数，使得熵最大。

### 熵的定义
连续随机变量 $X$ 的微分熵定义为：

$$
H(X) = - \int_{-\infty}^{\infty} f(x) \ln f(x) \, dx
$$

### 约束条件

1. **概率密度函数的归一化条件：**

   $$
   \int_{-\infty}^{\infty} f(x) \, dx = 1
   $$

2. **均值为 $\mu$：**

   $$
   \int_{-\infty}^{\infty} x f(x) \, dx = \mu
   $$

3. **方差为 $\sigma^2$：**

   $$
   \int_{-\infty}^{\infty} (x - \mu)^2 f(x) \, dx = \sigma^2
   $$

### 拉格朗日乘子法
构建拉格朗日函数：

$$
L = - \int_{-\infty}^{\infty} f(x) \ln f(x) \, dx + \lambda_0 \left( \int_{-\infty}^{\infty} f(x) \, dx - 1 \right) + \lambda_1 \left( \int_{-\infty}^{\infty} x f(x) \, dx - \mu \right) + \lambda_2 \left( \int_{-\infty}^{\infty} (x - \mu)^2 f(x) \, dx - \sigma^2 \right)
$$

对 $f(x)$ 求函数的变分，使 $L$ 取极值：

$$
\delta L = - \int_{-\infty}^{\infty} \left( \ln f(x) + 1 - \lambda_0 - \lambda_1 x - \lambda_2 (x - \mu)^2 \right) \delta f(x) \, dx = 0
$$

由于 $\delta f(x)$ 任意，因此：

$$
\ln f(x) = -1 + \lambda_0 + \lambda_1 x + \lambda_2 (x - \mu)^2
$$

解得：

$$
f(x) = \exp\left( \lambda_0 - 1 + \lambda_1 x + \lambda_2 (x - \mu)^2 \right)
$$

### 确定拉格朗日乘子
为了使 $f(x)$ 正且可积，$\lambda_2$ 必须为负数。设：

$$
\lambda_2 = -\frac{1}{2\sigma^2}
$$

令 $\lambda_1 = 0$（由于对称性，均值 $\mu$ 已包含在函数中），则：

$$
f(x) = C \exp\left( - \frac{(x - \mu)^2}{2\sigma^2} \right)
$$

其中 $C = \exp(\lambda_0 - 1)$。

利用归一化条件确定 $C$：

$$
\int_{-\infty}^{\infty} f(x) \, dx = C \int_{-\infty}^{\infty} \exp\left( - \frac{(x - \mu)^2}{2\sigma^2} \right) \, dx = 1
$$

计算积分：

$$
\int_{-\infty}^{\infty} \exp\left( - \frac{(x - \mu)^2}{2\sigma^2} \right) \, dx = \sigma \sqrt{2\pi}
$$

因此：

$$
C = \frac{1}{\sigma \sqrt{2\pi}}
$$

最终得到高斯分布的概率密度函数：

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( - \frac{(x - \mu)^2}{2\sigma^2} \right)
$$

### 从中心极限定理推导
中心极限定理表明，大量独立同分布的随机变量之和趋近于正态分布。

假设有 $n$ 个独立同分布的随机变量 $X_i$，其均值为 $\mu$，方差为 $\sigma^2$。定义标准化的和为：

$$
S_n = \frac{\sum_{i=1}^{n} X_i - n\mu}{\sigma\sqrt{n}}
$$

当 $n \to \infty$ 时，$S_n$ 的分布趋于标准正态分布：

$$
\lim_{n \to \infty} P(S_n \leq z) = \Phi(z) = \int_{-\infty}^{z} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} \, dt
$$

因此，高斯分布是大量独立随机变量之和的极限分布。

### 通过微分方程推导
假设概率密度函数 $f(x)$ 满足以下微分方程：

$$
\frac{d}{dx} \left( \ln f(x) \right) = - \frac{x - \mu}{\sigma^2}
$$

解此微分方程：

$$
\ln f(x) = - \frac{(x - \mu)^2}{2\sigma^2} + C
$$

因此：

$$
f(x) = \exp\left( - \frac{(x - \mu)^2}{2\sigma^2} + C \right) = C' \exp\left( - \frac{(x - \mu)^2}{2\sigma^2} \right)
$$

利用归一化条件确定 $C'$，最终得到高斯分布的概率密度函数。

## 结论
通过以上推导，我们得到了高斯分布的概率密度函数：

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( - \frac{(x - \mu)^2}{2\sigma^2} \right)
$$
