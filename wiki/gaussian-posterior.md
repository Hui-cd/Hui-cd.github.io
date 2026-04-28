# Gaussian Posterior

> 高斯先验 + 线性高斯似然的后验分布推导，贝叶斯推断的解析范例。

## Overview

当先验和似然都是高斯分布时，后验分布也有**闭式解析解**——仍然是高斯分布。这是贝叶斯推断中最优雅的结果之一，广泛应用于卡尔曼滤波、高斯过程、变分推断等领域。

辉少的博客完整推导了一维情形：先验 $p(x) = \mathcal{N}(\mu_A, \sigma_A^2)$，似然 $p(y|x) = \mathcal{N}(ax, \sigma_B^2)$。

## Key Facts / Claims

### 后验分布形式

- 后验：$p(x|y) = \mathcal{N}(\tilde{\mu}, \tilde{\sigma}^2)$
- 后验精度（方差的倒数）：$\tilde{\sigma}^{-2} = \sigma_A^{-2} + a^2\sigma_B^{-2}$
- 后验均值：$\tilde{\mu} = \tilde{\sigma}^2 \left(\sigma_A^{-2}\mu_A + a\sigma_B^{-2}y\right)$

### 推导关键步骤

1. 贝叶斯定理：$p(x|y) \propto p(y|x)p(x)$
2. 指数部分展开：$-\frac{(y-ax)^2}{2\sigma_B^2} - \frac{(x-\mu_A)^2}{2\sigma_A^2}$
3. 整理 $x^2$ 项系数：$A = \frac{a^2}{\sigma_B^2} + \frac{1}{\sigma_A^2}$
4. 整理 $x$ 项系数：$B = \frac{2ay}{\sigma_B^2} + \frac{2\mu_A}{\sigma_A^2}$
5. 完成平方：$-\frac{A}{2}(x - \frac{B}{2A})^2 + \text{const}$
6. 读出高斯参数：$\tilde{\sigma}^2 = \frac{1}{A}$，$\tilde{\mu} = \frac{B}{2A}$

### 直观理解

- **精度相加**：后验精度 = 先验精度 + 似然精度（加权）
- **均值加权**：后验均值是先验均值和观测值的加权平均，权重与各自的精度成正比
- 观测噪声越小（$\sigma_B^2$ 越小），后验越向观测值 $y/a$ 靠拢

## Related

- [[gaussian-distribution]] — 高斯分布的基础推导
- [[spherical-gaussian]] — 高斯分布的简化形式
- [[machine-learning-basics]] — 贝叶斯分类器的核心思想
- [[vae]] — 变分自编码器中编码器输出高斯后验 $q(z|x)$

## Sources

- [高斯先验和线性高斯似然的后验分布推导](https://hui-cd.github.io/2024/08/31/Derivation-of-the-posterior-distribution/) — 辉少的博客原文
