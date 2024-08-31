---
single: post
title:  "逻辑回归"
date:   2024-08-31 22:50:22 +1000
categories: 
    - machine-learning
author: Hui
---


## 逻辑回归
在逻辑回归模型中，对于二分类问题，目标变量 `y` 可以取 0 或 1 的值，其中 1 代表属于类别 $c_1$，0 代表属于类别 $c_2$。模型的目的是基于输入特征 `x` 来预测 `y` 的概率 $p(y=1|x)$，这个概率由 Sigmoid 函数 `σ` 来计算，具体如下：

$$
p(y=1|x) = \sigma(\theta^T x)
$$

这里，`θ` 是模型的参数向量，$\theta^T x$ 是参数和特征向量的点积。Sigmoid 函数定义为：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### 构建似然函数

在逻辑回归中，似然函数的构建基于最大化观测数据（训练数据集）的概率。对于一个给定的数据集 ${(x_1, y_1),(x_2,y_2), ..., (x_N, y_N) }$，每一个 $(x_i, y_i)$ 都是独立同分布的。因此，整个数据集的联合概率（似然函数）是每个单独数据点概率的乘积：

假设 $ŷ_i = σ(θ^T x_i)$ 是模型预测的概率，即 $y_i = 1$ 的概率。那么：

- 如果 $y_i = 1$，观测到这个结果的概率是 $ŷ_i$。
- 如果 $y_i = 0$，观测到这个结果的概率是 $1 - ŷ_i$。

因此，对于单个观测 $(x_i, y_i)$，其概率可以表达为：

$$
p(y_i | x_i) = ŷ_i^{y_i} (1 - ŷ_i)^{1 - y_i}
$$

对于整个数据集，似然函数 $L(θ)$ 是所有单个数据点概率的乘积：

$$
L(\theta) = \prod_{i=1}^{N} \left[ ŷ_i^{y_i} (1 - ŷ_i)^{1 - y_i} \right]
$$

通常，为了便于计算，我们会取对数似然函数：

$$
\log L(\theta) = \sum_{i=1}^{N} \left[ y_i \log(ŷ_i) + (1 - y_i) \log(1 - ŷ_i) \right]
$$

这个对数似然函数是逻辑回归模型中使用的损失函数，通常称为交叉熵损失。模型训练的目标是通过调整参数 `θ` 来最大化这个对数似然函数，或者等价地最小化负对数似然函数。


对于交叉熵损失函数的导数计算，我们首先需要理解Sigmoid函数（σ(z)）的导数是如何得来的。

要求其导数，我们可以直接对这个函数进行求导：

1. 首先，令 $u = 1 + e^{-z}$，则 $\sigma(z) = \frac{1}{u}$。
2. 应用链式法则，我们先求 $u$ 关于 $z$ 的导数，再求 $\frac{1}{u}$ 的导数：
   - $\frac{du}{dz} = -e^{-z}$ （因为 $e^{-z}$ 关于 $z$ 的导数是 $-e^{-z}$）
   - 应用链式法则，我们有 $$\frac{d\sigma}{dz} = \frac{d}{dz}\left(\frac{1}{u}\right) = -\frac{1}{u^2}\frac{du}{dz} = -\frac{1}{(1+e^{-z})^2}(-e^{-z})$$

1. 代入 $e^{-z}$ 以简化表达式：
   - $$ \frac{d\sigma}{dz} = \frac{e^{-z}}{(1 + e^{-z})^2} $$

2. 使用Sigmoid函数的定义式 $\sigma(z) = \frac{1}{1 + e^{-z} }$，我们可以将 $e^{-z}$ 用 $\sigma(z)$ 表达：
   - $e^{-z} = \frac{1 - \sigma(z)}{\sigma(z)}$（通过移项和代数变换得到）

3. 最终，我们将 $e^{-z}$ 代入 $\frac{d\sigma}{dz}$ 的表达式：
   - $$ \frac{d\sigma}{dz} = \frac{\frac{1 - \sigma(z)}{\sigma(z)}}{(1 + \frac{1 - \sigma(z)}{\sigma(z)})^2} $$
   - 简化这个表达式，我们会得到 $\sigma(z)(1 - \sigma(z))$

因此，Sigmoid函数的导数是

$$ \sigma(z)(1 - \sigma(z)) $$

这个结果对于后续计算交叉熵损失函数的梯度非常关键，因为它提供了预测值 $\hat{y}_n$ 关于参数 $\theta$ 的偏导数的部分。


对于二分类问题的交叉熵损失函数 $L(\theta)$，其定义为：


$$
L(\theta) = -\sum_{n=1}^N \left( y_n \log(\hat{y}_n) + (1-y_n) \log(1-\hat{y}_n) \right)
$$

其中 $y_n$ 是实际的标签，$\hat{y}_n = \sigma(\theta^T x_n)$是通过模型预测出的概率，$\sigma(z)$ 是Sigmoid函数。

### 计算损失函数对参数 $\theta$ 的梯度：

#### 1. Sigmoid函数的导数
Sigmoid函数 \(\sigma(z)\) 的导数是：

$$
\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))
$$

#### 2. 预测概率 $\hat{y}_n$ 对 $\theta$的偏导数
应用链式法则，预测概率 $\hat{y}_n$ 对 $\theta$ 的偏导数为：

$$
\frac{\partial \hat{y}_n}{\partial \theta} = \hat{y}_n(1 - \hat{y}_n) x_n
$$

#### 3. 损失函数 $L(\theta)$ 对 $\theta$ 的偏导数
将 $\hat{y}_n$ 的偏导数代入损失函数的偏导数计算中：

$$
\frac{\partial L(\theta)}{\partial \theta} = -\sum_{n=1}^N \left( y_n \frac{1}{\hat{y}_n} \frac{\partial \hat{y}_n}{\partial \theta} + (1-y_n) \frac{-1}{1-\hat{y}_n} \frac{\partial \hat{y}_n}{\partial \theta} \right)
$$

简化得到：

$$
\frac{\partial L(\theta)}{\partial \theta} = -\sum_{n=1}^N \left( y_n \frac{1}{\hat{y}_n} \hat{y}_n (1-\hat{y}_n) x_n - (1-y_n) \frac{1}{1-\hat{y}_n} \hat{y}_n (1-\hat{y}_n) x_n \right)
$$

$$
\frac{\partial L(\theta)}{\partial \theta} = \sum_{n=1}^N (\hat{y}_n - y_n) x_n
$$

因此，最终我们得到了损失函数的梯度为：

$$
\nabla L(\theta) = \sum_{n=1}^N (\hat{y}_n - y_n) x_n
$$


### 没有闭式解（No Closed Form Solution）

在机器学习中，如果一个算法或模型有闭式解（Closed Form Solution），这意味着我们可以通过直接求解方程来找到模型参数的最优解，而不需要迭代方法。例如，线性回归有闭式解，可以通过解析方法（如最小二乘法）直接计算出权重。

然而，逻辑回归没有闭式解。这是因为其损失函数（通常是交叉熵损失）导致的参数优化问题不能通过简单的代数方程解决。因此，我们通常使用迭代优化算法（如梯度下降法）来找到损失函数的最小值。
### 是凸函数（Is Convex）

关于函数的凸性，如果一个函数是凸的，那么这个函数的任意两点上的线段都在函数的图像之上。对于优化问题，这意味着函数有一个全局最小值，而不是多个局部最小值，这在寻找最优解时是一个非常有利的特性。

在逻辑回归中，如果我们使用交叉熵作为损失函数，并假设数据线性可分，那么这个损失函数关于参数（权重）是凸的。这种凸性质保证了使用梯度下降法等优化算法可以有效地找到全局最优解.
