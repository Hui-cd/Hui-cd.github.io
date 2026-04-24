---
single: post
title:  "贝叶斯分类和决策边界"
date:   2024-08-31 22:50:22 +1000
categories: 
    - machine-learning
author: Hui
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
## 贝叶斯分类器的后验概率公式推导

### 基本定义和公式

每个类别 $w_i$ 都有一个先验概率 $p(w_i)$，并且对于所有类别来说，先验概率之和为1：

$$ \sum_{k=1}^K p(w_k) = 1$$

对于两类分类问题，我们通常有：

$$ p(w_1) + p(w_2) = 1$$

### 贝叶斯规则应用
后验概率 $p(w_i|x)$ 是在给定观测数据 $x$ 后，某个特定类别 $w_i$ 的条件概率，根据贝叶斯规则，可以表示为：

$$p(w_i|x) = \frac{p(x|w_i)p(w_i)}{p(x)}$$

其中，$p(x)$ 是证据因子，表示所有类别生成数据 $x$ 的概率：

$$ p(x) = \sum_{k=1}^{K} p(x|w_k)p(w_k) $$

### 两类分类问题的具体推导
对于两类问题，后验概率 $p(w_1|x)$ 的计算可以简化如下：

$$p(w_1|x) = \frac{p(x|w_1)p(w_1)}{p(x|w_1)p(w_1) + p(x|w_2)p(w_2)}$$

进一步化简为：

$$p(w_1|x) = \frac{1}{1 + \frac{p(x|w_2)p(w_2)}{p(x|w_1)p(w_1)}}$$

设 \( x \) 的条件概率密度为正态分布，则：

$$p(x|w_i) = \frac{1}{\sqrt{2\pi \sigma_i^2}} \exp \left( -\frac{(x - m_i)^2}{2\sigma_i^2} \right)$$

代入上述正态分布得：

$$p(w_1|x) = \frac{1}{1 + \exp \left( -\left[ \frac{(x - m_1)^2}{2\sigma_1^2} - \frac{(x - m_2)^2}{2\sigma_2^2} \right] + \ln \left( \frac{\sigma_2 \cdot p(w_2)}{\sigma_1 \cdot p(w_1)} \right) \right)}$$

### 等方差和等先验概率的情况
如果两个类别的方差相同$\sigma_1 = \sigma_2$并且先验概率也相同$p(w_1) = p(w_2)$，该公式进一步简化，分类界限当 $p(w_1|x) = 0.5$ 时，解析表达式为：

$$x = \frac{m_1 + m_2}{2}$$

这种情况下，分类器退化为简单的距均值分类器（即最近均值分类器）。

### 多元情况下的分类边界
对于多元特征的情况，后验概率的推导涉及协方差矩阵，对于类别$w_1$ 和 $w_2$，假设它们有不同的协方差矩阵 $C_1$ 和 $C_2$，分类界限可表示为一个二次方程：

<div class="scrollable-math-formula">
$$x^T (C_1^{-1} - C_2^{-1}) x + 2x^T (C_1^{-1} m1 - C_2^{-1} m2) + m1^T C_1^{-1} m1 - m2^T C_2^{-1} m2 - \ln \left| \frac{C2}{C1} \right| - 2 \ln \left( \frac{p(w1)}{p(w2)} \right) = 0$$
</div>

这种情况下，决策边界是二次的，依赖于协方差矩阵的差异。

