---
single: post
title:  "球形高斯分布"
date:   2024-10-05 22:50:22 +1000
categories: 
    - machine-learning
author: Hui
---

# 球形高斯分布的详细推导

## 1. 从一般多元正态分布开始

多元正态分布的概率密度函数为:

$$f(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$

其中:
- $x$ 是 $n$ 维随机向量
- $\mu$ 是均值向量
- $\Sigma$ 是协方差矩阵
- $ |\Sigma| $ 是 $\Sigma$ 的行列式

## 2. 引入球形高斯的假设

对于球形高斯分布,我们假设:

1. 所有维度的方差相等,设为 $\sigma^2$
2. 维度之间相互独立

这意味着协方差矩阵 $\Sigma = \sigma^2 I$，其中 $I$ 是 $n \times n$ 单位矩阵。

## 3. 简化协方差矩阵

将 $\Sigma = \sigma^2 I$ 代入一般多元正态分布的公式，我们可以简化以下部分:

1. 协方差矩阵的行列式:
   $$|\Sigma| = |\sigma^2 I| = (\sigma^2)^n$$

2. 协方差矩阵的逆:
   $$\Sigma^{-1} = (\sigma^2 I)^{-1} = \frac{1}{\sigma^2} I$$

## 4. 简化概率密度函数

将简化后的协方差矩阵代入原始公式:

$$
f(x) = \frac{1}{(2\pi)^{n/2} (\sigma^2)^n} \exp \left( -\frac{1}{2} (x - \mu)^T \frac{1}{\sigma^2} I (x - \mu) \right)
$$

## 5. 进一步简化

1. 简化分母:
   $$(\sigma^2)^n = \sigma^n$$

2. 简化指数项:
   $$(x - \mu)^T \frac{1}{\sigma^2} I (x - \mu) = \frac{1}{\sigma^2} (x - \mu)^T (x - \mu) = \frac{1}{\sigma^2} ||x - \mu||^2$$

其中 $ ||x - \mu||^2 $ 表示 $x$ 和 $\mu$ 之间的欧几里得距离的平方。

## 6. 最终形式

经过以上步骤,我们得到球形高斯分布的概率密度函数:

$$f(x) = \frac{1}{(2\pi)^{n/2} \sigma^n} \exp \left( -\frac{1}{2\sigma^2} ||x - \mu||^2 \right)$$

## 7. 解释

- 分母中的 $(2\pi)^{n/2} \sigma^n$ 是归一化常数,确保整个概率密度函数在所有可能的$x$值上积分等于1。
- 指数项 $ \exp \left( -\frac{1}{2\sigma^2} ||x - \mu||^2 \right) $ 描述了概率密度如何随着点$x$离均值$\mu$的距离增加而减小。
- $\sigma$ 控制分布的"宽度"或"分散程度"。较大的$\sigma$值会使分布更"扁平",较小的$\sigma$值会使分布更"尖峰"。
