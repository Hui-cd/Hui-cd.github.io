# Diffusion Model

> 通过前向加噪和反向去噪学习数据分布的生成模型，涵盖 DDPM、Score Matching、SDE 统一视角

## Overview

扩散模型是当前最强大的生成模型之一。辉少的博客从三个层次覆盖了扩散模型：DDPM 的离散推导、DDPM 与 Score Matching 的统一（SDE 视角）、以及具体实现细节。核心思想是通过可逆的随机过程将数据分布转化为高斯分布，再学习反向过程恢复数据。

## Key Facts / Claims

### DDPM 核心推导
1. **ELBO 分解**：
   $$L_{\text{VLB}} = D_{KL}(q(x_T|x_0)||p(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1)$$

2. **后验分布**（高斯闭式）：
   $$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t I)$$
   $$\tilde{\mu}_t = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0$$

3. **噪声预测目标**：
   $$L_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2]$$

### DDPM 与 Score Matching 统一
- **VP-SDE**（DDPM 连续极限）：$dX_t = -\frac{1}{2}\beta(t)X_t dt + \sqrt{\beta(t)}dW_t$
- **VE-SDE**（NCSN/EDM）：$dX_t = \sqrt{\frac{d}{dt}\sigma^2(t)}dW_t$
- **重参数化**：$x_t = \alpha(t)x_0 + \sigma(t)\epsilon$
- **Score 与噪声的等价**：$s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sigma(t)}$

### 反向动力学
- **反向 SDE**：$dX_t = [f - g^2\nabla\log p_t]dt + g d\bar{W}_t$
- **概率流 ODE**（确定性）：$\frac{dX_t}{dt} = f - \frac{1}{2}g^2\nabla\log p_t$

### 从 $x_t$ 恢复 $x_0$
- VP：$\hat{x}_0 = \frac{x_t - \sigma\hat{\epsilon}_\theta}{\alpha}$
- VE：$\hat{x}_0 = x_t - \sigma\hat{\epsilon}_\theta$

## Related
- [[vae]] — 另一大潜变量生成模型
- [[transformer]] — 扩散模型中常用的骨干网络
- [[flash-attention]] — 长序列训练时的注意力优化
- [[flow-matching]] — 另一种生成模型训练范式（速度场回归）
- [[kl-divergence]] — ELBO 中的核心组件
- [[external-blogs]] — Yang Song 的 Score Matching 博客

## Counter-arguments & Data Gaps
- 采样速度仍是瓶颈（需要多步去噪）
- 与 GAN、Flow-based 模型的全面比较
- 条件生成（classifier-free guidance）的理论分析

## Sources
- [DDPM 数学推导](../sources/2025-12-15-ddpm.md) — 2025-12-15
- [DDPM 与 Score Matching 的统一推导](../sources/2025-07-29-ddpm-score-matching.md) — 2025-07-29
- [DiffusionModel](../sources/DiffusionModel.md) — 未标注日期
