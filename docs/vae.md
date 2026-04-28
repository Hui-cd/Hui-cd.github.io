# VAE

> 变分自编码器：通过潜变量和变分推断学习数据分布的生成模型

## Overview

VAE（Variational Autoencoder）是深度生成模型的奠基工作之一。与自回归模型和 GAN 不同，VAE 通过引入潜变量 $z$ 和变分推断，将生成问题转化为优化 ELBO（证据下界）。

## Key Facts / Claims

### 核心思想
- 数据 $x$ 由潜变量 $z$ 通过 Decoder $p_\theta(x|z)$ 生成
- 后验 $p(z|x)$ 难以计算，用变分分布 $q_\phi(z|x)$ 近似
- 优化 ELBO：$\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$

### ELBO 的两项
1. **重构项**：$\mathbb{E}[\log p_\theta(x|z)]$ — 解码质量
2. **KL 项**：$D_{KL}(q_\phi(z|x) || p(z))$ — 近似后验与先验的匹配

### 重参数化技巧
- $z = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0,I)$
- 使梯度能反向传播通过采样操作

### 与扩散模型的关系
- VAE 的 Encoder-Decoder 结构启发了扩散模型的设计
- 扩散模型可视为多步 VAE，每一步去噪相当于一次解码
- VAE 的 KL 项与扩散模型 ELBO 中的 KL 项数学形式相同

## Related
- [[diffusion-model]] — 多步去噪的扩展
- [[kl-divergence]] — ELBO 中的核心组件
- [[machine-learning-basics]] — 概率基础
- [[flow-matching]] — 另一种生成模型范式
- [[external-blogs]] — Yang Song 的生成模型博客

## Counter-arguments & Data Gaps
- VAE 生成质量通常不如扩散模型和 GAN
- 后验塌陷（posterior collapse）：Decoder 忽略潜变量
- 与 Flow-based 模型的比较

## Sources
- [vae](../sources/vae.md) — 未标注日期
