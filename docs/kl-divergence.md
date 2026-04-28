# KL Divergence

> 度量两个概率分布差异的非对称统计量，在 VAE、扩散模型、信息论中广泛应用

## Overview

KL 散度（Kullback-Leibler divergence）是机器学习中最重要的分布距离度量之一。辉少的博客详细推导了两个正态分布之间的 KL 散度闭式解，这在 VAE 的 ELBO、扩散模型的变分下界中反复出现。

## Key Facts / Claims

### 定义
$$\text{KL}(P \parallel Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

### 正态分布的 KL 散度（闭式解）
对于 $P \sim \mathcal{N}(\mu_P, \sigma_P^2)$ 和 $Q \sim \mathcal{N}(\mu_Q, \sigma_Q^2)$：
$$\text{KL}(P \parallel Q) = \log \frac{\sigma_Q}{\sigma_P} + \frac{\sigma_P^2 + (\mu_P - \mu_Q)^2}{2\sigma_Q^2} - \frac{1}{2}$$

### 推导要点
1. 展开 $\log \frac{p(x)}{q(x)}$ 的对数项
2. 利用正态分布的期望和方差：$E[x] = \mu_P$，$E[x^2] = \sigma_P^2 + \mu_P^2$
3. 积分后得到闭式表达式

### 应用场景
- **VAE**：ELBO 中的 $D_{KL}(q(z|x) || p(z))$ 项
- **扩散模型**：ELBO 分解中的 KL 项
- **变分推断**：衡量近似后验与真实后验的差距
- **信息论**：相对熵，描述编码效率损失

## Related
- [[diffusion-model]] — ELBO 推导中大量使用 KL 散度
- [[vae]] — 潜变量模型的核心损失组件
- [[machine-learning-basics]] — 基础概率统计
- [[llm-rl-algorithms]] — PPO/DPO/GRPO 中的 KL 约束
- [[sft-vs-rlhf]] — RLHF 中的 KL 惩罚

## Counter-arguments & Data Gaps
- KL 散度非对称：$KL(P||Q) \neq KL(Q||P)$，某些场景需要 JS 散度或 Wasserstein 距离
- 两个分布无重叠时 KL 散度为无穷大，训练不稳定
- 多维正态分布的 KL 散度推导（博客仅覆盖了一维）

## Sources
- [Kullback-Leibler divergence](../sources/2024-09-18-Kullback-LeiblerDivergence.md) — 2024-09-18
