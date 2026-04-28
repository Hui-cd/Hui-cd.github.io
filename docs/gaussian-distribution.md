# Gaussian Distribution

> 概率论中最重要的连续分布，从最大熵原理、中心极限定理、微分方程三个角度推导。

## Overview

高斯分布（正态分布）是机器学习和统计学的基石。它的重要性不仅体现在数学上的优雅（均值和方差完全刻画分布），更在于**中心极限定理**保证了它在自然界中的普适性。

辉少的博客从三个不同角度推导了高斯分布的 PDF，展示了其深刻的数学根源。

## Key Facts / Claims

### 概率密度函数

- 一维高斯：$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$
- 多维高斯：$f(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$

### 推导一：最大熵原理

- 约束：已知均值 $\mu$ 和方差 $\sigma^2$，求使熵最大的分布
- 拉格朗日函数：$L = -\int f \ln f \, dx + \lambda_0(\int f dx - 1) + \lambda_1(\int xf dx - \mu) + \lambda_2(\int (x-\mu)^2 f dx - \sigma^2)$
- 变分求解得：$\ln f(x) = -1 + \lambda_0 + \lambda_1 x + \lambda_2 (x-\mu)^2$
- 代入约束确定系数，最终得到高斯 PDF
- **核心洞察**：高斯分布是在给定一阶、二阶矩约束下"最不确定"（熵最大）的分布

### 推导二：中心极限定理

- $n$ 个独立同分布随机变量之和的标准化形式：$S_n = \frac{\sum X_i - n\mu}{\sigma\sqrt{n}}$
- 当 $n \to \infty$ 时，$S_n \to \mathcal{N}(0,1)$
- **核心洞察**：大量微小独立因素的叠加必然趋于高斯分布

### 推导三：微分方程

- 假设 $\frac{d}{dx}(\ln f(x)) = -\frac{x-\mu}{\sigma^2}$
- 解得 $\ln f(x) = -\frac{(x-\mu)^2}{2\sigma^2} + C$
- **核心洞察**：对数密度的线性变化率假设直接导出高斯形式

## Related

- [[kl-divergence]] — 两个高斯分布之间的 KL 散度推导
- [[spherical-gaussian]] — 协方差矩阵为 $\sigma^2 I$ 的特殊情形
- [[gaussian-posterior]] — 高斯先验 + 线性高斯似然的后验推导
- [[machine-learning-basics]] — Fisher LDA 中假设类条件分布为高斯
- [[diffusion-model]] — 前向扩散过程的核心就是高斯噪声注入

## Sources

- [高斯分布的数学推导详解](https://hui-cd.github.io/2024/10/12/Gaussian-distribution/) — 辉少的博客原文
