---
single: post
title:  "Diffusion model"
date:   2024-09-02 18:26:22 +1000
categories: 
    - deep-learning
author: Hui
---

## Diffusion model 

Diffusion model 是使用变分推理训练的参数化马尔可夫链，以在有限时间后生成与数据匹配的样本。

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) := \mathcal{N} \left( \mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I} \right)$$

我们不需要重复应用 $q$ 来从 $x_t \sim q(x_t|x_0)$ 中采样。
