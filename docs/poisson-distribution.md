# Poisson Distribution

> 描述单位时间/空间内稀有事件发生次数的离散分布，二项分布的极限形式。

## Overview

Poisson 分布是概率论中重要的离散分布，用于建模固定时间或空间间隔内**随机事件发生次数**。典型场景：呼叫中心接到的电话数、某路口的交通事故数、放射性物质的衰变次数。

其核心特征是：事件发生的平均速率 $\lambda$ 恒定，且事件之间相互独立。

## Key Facts / Claims

### 概率质量函数

- $P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}$，其中 $k = 0, 1, 2, \dots$
- 均值：$\mathbb{E}[X] = \lambda$
- 方差：$\text{Var}[X] = \lambda$（均值等于方差是 Poisson 的标志性特征）

### 从二项分布推导

- 二项分布：$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$
- 设定 $p = \frac{\lambda}{n}$，保持 $np = \lambda$ 不变
- 当 $n \to \infty$ 时：
  - $\binom{n}{k} \approx \frac{n^k}{k!}$
  - $(1 - \frac{\lambda}{n})^{n-k} \approx e^{-\lambda}$
- 代入化简得：$P(X=k) \approx \frac{\lambda^k}{k!} e^{-\lambda}$

### 适用条件

- 事件在不相交区间独立发生
- 在极短区间内，事件发生概率很小
- 平均发生率 $\lambda$ 恒定
- 两个事件不会同时发生

## Related

- [[gaussian-distribution]] — 当 $\lambda$ 很大时，Poisson 可用高斯近似
- [[machine-learning-basics]] — 概率基础与统计推断
- [[probability-basics]] — 概率论基本概念

## Sources

- [Poisson 分布：概率质量函数的推导过程](https://hui-cd.github.io/2024/10/29/Poisson/) — 辉少的博客原文
