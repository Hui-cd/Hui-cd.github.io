# Probability Basics

> 先验概率、后验概率与贝叶斯定理的基础概念。

## Overview

概率论是机器学习的数学基石。理解先验概率、后验概率以及贝叶斯定理，是掌握贝叶斯分类器、概率图模型、变分推断等方法的前提。

## Key Facts / Claims

### 先验概率（Prior）

- 定义：在观测到任何新证据前，对事件发生概率的初始判断
- 来源：历史经验、领域知识或主观假设
- 示例：某社区 5% 的人患有某种遗传病，$P(\text{病}) = 0.05$

### 似然（Likelihood）

- 定义：在给定假设下，观测到当前证据的概率
- 示例：患病者中 95% 测试呈阳性（灵敏度），$P(\text{阳性}|\text{病}) = 0.95$

### 后验概率（Posterior）

- 定义：在观测到证据后，更新的事件发生概率
- 贝叶斯定理：$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- 示例：测试呈阳性者实际患病的概率

### 数值示例

- 先验：$P(\text{病}) = 0.05$
- 灵敏度：$P(\text{阳性}|\text{病}) = 0.95$
- 特异度：$P(\text{阴性}|\text{无病}) = 0.90$，即 $P(\text{阳性}|\text{无病}) = 0.10$
- 全概率：$P(\text{阳性}) = 0.95 \times 0.05 + 0.10 \times 0.95 = 0.1425$
- 后验：$P(\text{病}|\text{阳性}) = \frac{0.95 \times 0.05}{0.1425} \approx 0.333$

### 核心洞察

- 即使测试准确率很高，如果疾病本身罕见（先验低），阳性结果的实际患病概率可能远低于直觉
- 这就是**基础比率谬误**（Base Rate Fallacy）

## Related

- [[machine-learning-basics]] — 贝叶斯分类器直接应用这些概念
- [[gaussian-posterior]] — 高斯先验的后验更新是贝叶斯推断的解析实例
- [[em-algorithm]] — 隐变量模型的贝叶斯推断方法

## Sources

- [Probability Basics](https://hui-cd.github.io/Probability/) — 辉少的笔记原文
