# Spherical Gaussian

> 协方差矩阵为 $\sigma^2 I$ 的多元高斯分布，各维度独立同方差。

## Overview

球形高斯分布（Spherical Gaussian）是多元高斯分布的简化形式。假设所有维度方差相等且相互独立，协方差矩阵退化为 $\Sigma = \sigma^2 I$。

这种简化在计算上大幅降低了复杂度（行列式和逆矩阵的计算变得平凡），同时在许多实际场景中（如各向同性噪声）是合理的近似。

## Key Facts / Claims

### 从多元高斯简化

- 一般多元高斯：$f(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$
- 球形假设：$\Sigma = \sigma^2 I$
- 行列式：$|\Sigma| = (\sigma^2)^n = \sigma^{2n}$
- 逆矩阵：$\Sigma^{-1} = \frac{1}{\sigma^2} I$

### 简化后的 PDF

$$
f(x) = \frac{1}{(2\pi)^{n/2} \sigma^n} \exp\left( -\frac{1}{2\sigma^2} ||x - \mu||^2 \right)
$$

- 分母：$(2\pi)^{n/2} \sigma^n$ 是归一化常数
- 指数项：仅依赖于 $x$ 到均值 $\mu$ 的**欧几里得距离平方**
- 等概率面：以 $\mu$ 为中心的球面（故名"球形"）

### 几何解释

- 概率密度随距离 $||x - \mu||$ 指数衰减
- $\sigma$ 控制分布的"宽度"：大 $\sigma$ 更扁平，小 $\sigma$ 更尖峰
- 各维度完全对称，无方向偏好

## Related

- [[gaussian-distribution]] — 一般多元高斯分布的完整推导
- [[gaussian-posterior]] — 高斯先验的后验更新
- [[machine-learning-basics]] — 贝叶斯分类器中的类条件分布假设
- [[diffusion-model]] — DDPM 中假设噪声为各向同性高斯

## Sources

- [球形高斯分布](https://hui-cd.github.io/2024/10/06/SphericalGaussian/) — 辉少的博客原文
