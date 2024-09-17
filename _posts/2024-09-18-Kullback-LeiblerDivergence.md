---
single: post
title:  "Kullback-Leibler divergence"
date:   2024-09-18 02:26:22 +1000
categories: 
    - deep-learning
author: Hui
---

## 正态分布的KL散度推导

假设我们有两个正态分布，$P$ 和 $Q$，其中：
- $P \sim \mathcal{N}(\mu_P, \sigma_P^2)$
- $Q \sim \mathcal{N}(\mu_Q, \sigma_Q^2)$

#### KL散度的定义

对于连续变量，KL散度的定义为：
$$
\text{KL}(P \parallel Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

#### 正态分布的概率密度函数

正态分布的概率密度函数为：

$$
p(x) = \frac{1}{\sqrt{2\pi \sigma_P^2}} e^{-\frac{(x-\mu_P)^2}{2\sigma_P^2}}
$$

$$
q(x) = \frac{1}{\sqrt{2\pi \sigma_Q^2}} e^{-\frac{(x-\mu_Q)^2}{2\sigma_Q^2}}
$$

#### 对数项展开

代入$p(x)$和$q(x)$到对数中，展开得到：

$$
\log \frac{p(x)}{q(x)} = \log \frac{\sigma_Q}{\sigma_P} + \left(\frac{(x-\mu_Q)^2}{2\sigma_Q^2} - \frac{(x-\mu_P)^2}{2\sigma_P^2}\right)
$$

#### 展开平方项

接下来，我们展开两个平方项：

$$
\frac{(x-\mu_Q)^2}{2\sigma_Q^2} - \frac{(x-\mu_P)^2}{2\sigma_P^2} = \frac{x^2 - 2x\mu_Q + \mu_Q^2}{2\sigma_Q^2} - \frac{x^2 - 2x\mu_P + \mu_P^2}{2\sigma_P^2}
$$

$$
= \left(\frac{1}{2\sigma_Q^2} - \frac{1}{2\sigma_P^2}\right)x^2 + \left(\frac{2\mu_Q}{2\sigma_Q^2} - \frac{2\mu_P}{2\sigma_P^2}\right)x + \left(\frac{\mu_Q^2}{2\sigma_Q^2} - \frac{\mu_P^2}{2\sigma_P^2}\right)
$$

#### 考虑期望和方差

代入期望和方差值：
- 期望 $E[x] = \mu_P$
- 期望 $E[x^2] = \sigma_P^2 + \mu_P^2$

用$E[x]$代替$x$，$E[x^2]$代替$x^2$并执行积分：
$$
\text{KL}(P \parallel Q) = \int p(x) \left[ \log \frac{\sigma_Q}{\sigma_P} + \frac{\sigma_P^2 + (\mu_P - \mu_Q)^2}{2\sigma_Q^2} - \frac{1}{2} \right] dx
$$

由于 $p(x)$ 是 $x$ 的概率密度函数，整个积分相当于公式中的系数求和：
$$
\text{KL}(P \parallel Q) = \log \frac{\sigma_Q}{\sigma_P} + \frac{\sigma_P^2 + (\mu_P - \mu_Q)^2}{2\sigma_Q^2} - \frac{1}{2}
$$