# Logistic Regression

> 二分类问题的概率建模方法，通过 Sigmoid 函数将线性输出映射到概率空间。

## Overview

逻辑回归（Logistic Regression）是机器学习中经典的**概率分类器**。虽然名字里有"回归"，但它解决的是分类问题——通过 Sigmoid 函数将线性组合 $\theta^T x$ 映射到 $(0,1)$ 区间，表示样本属于正类的概率。

核心思想：用**对数似然函数**作为损失，通过梯度下降迭代优化参数。

## Key Facts / Claims

### 模型定义

- 预测概率：$p(y=1|x) = \sigma(\theta^T x)$，其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Sigmoid 导数：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$ —— 这一性质使梯度计算极度简化

### 似然函数与交叉熵损失

- 单个样本概率：$p(y_i|x_i) = \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}$
- 对数似然（即负交叉熵损失）：
  $$
  \log L(\theta) = \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  $$

### 梯度推导

- 损失函数对 $\theta$ 的梯度：
  $$
  \nabla L(\theta) = \sum_{n=1}^N (\hat{y}_n - y_n) x_n
  $$
- 形式简洁：预测误差 $(\hat{y} - y)$ 乘以特征 $x$，与线性回归的梯度形式一致

### 优化特性

- **无闭式解**：损失函数非线性，无法像线性回归那样解析求解
- **凸函数**：交叉熵损失关于 $\theta$ 是凸的，保证梯度下降收敛到全局最优
- **多分类扩展**：Softmax 回归是逻辑回归的多分类推广

## Related

- [[machine-learning-basics]] — 贝叶斯分类器、感知器、SVM 等其他分类方法
- [[transformer]] — 现代深度学习中 Softmax 在注意力机制中的应用
- [[llm-rl-algorithms]] — 策略梯度中的对数概率技巧与这里的对数似然思想相通

## Sources

- [Logistic Regression](https://hui-cd.github.io/2024/08/31/LogisticRegression/) — 辉少的博客原文
