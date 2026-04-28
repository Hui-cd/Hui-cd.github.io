# EM Algorithm

> 隐变量模型参数估计的迭代算法，通过 E 步和 M 步交替最大化似然下界。

## Overview

EM（Expectation-Maximization）算法是处理**含隐变量概率模型**参数估计的经典方法。当模型中存在无法直接观测的变量时，直接最大化似然函数变得困难（涉及对隐变量的求和/积分）。EM 算法通过迭代构造似然函数的下界并最大化该下界，间接实现似然最大化。

## Key Facts / Claims

### 核心思想

- 观测变量 $X$，隐变量 $Z$，模型参数 $\theta$
- 对数似然：$L(\theta) = \log p(X|\theta) = \log \sum_Z p(X,Z|\theta)$
- 直接优化困难：log 内部有求和

### E 步（Expectation）

- 基于当前参数 $\theta^{(t)}$，计算隐变量的后验分布 $p(Z|X,\theta^{(t)})$
- 构造 Q 函数：$Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$
- 物理意义：用当前参数"填补"缺失的隐变量信息

### M 步（Maximization）

- 最大化 Q 函数更新参数：$\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$
- 物理意义：在"完整数据"上做一次标准的极大似然估计

### 数学推导关键

- Jensen 不等式：$\log \sum_Z q(Z) \frac{p(X,Z|\theta)}{q(Z)} \geq \sum_Z q(Z) \log \frac{p(X,Z|\theta)}{q(Z)}$
- 当 $q(Z) = p(Z|X,\theta^{(t)})$ 时，下界在 $\theta = \theta^{(t)}$ 处与真实似然相切
- 因此每次迭代保证 $L(\theta^{(t+1)}) \geq L(\theta^{(t)})$

### 收敛性与局限

- **保证收敛**：每次迭代似然不减，收敛到局部极大值
- **局部最优**：对初始值敏感，可能陷入局部最优
- **收敛速度**：通常比梯度下降慢，但每步更稳定

### 典型应用

- 高斯混合模型（GMM）参数估计
- 隐马尔可夫模型（HMM）训练
- 概率 PCA、因子分析
- 变分推断中的坐标上升

## Related

- [[gaussian-distribution]] — GMM 中每个组件都是高斯分布
- [[machine-learning-basics]] — 贝叶斯框架下的参数估计思想
- [[vae]] — 变分推断可视为 EM 的随机/近似版本
- [[diffusion-model]] — 扩散模型的训练也可看作一种特殊的 EM

## Sources

- [EM算法详解与数学推导](https://hui-cd.github.io/em/) — 辉少的博客原文
