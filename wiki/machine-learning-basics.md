# Machine Learning Basics

> 辉少博客中关于机器学习基础算法的推导合集：贝叶斯分类、Fisher LDA、感知器、SVM、逻辑回归

## Overview

这一系列文章覆盖了监督学习中最核心的线性/非线性分类方法，从概率视角（贝叶斯）到几何视角（SVM）再到迭代优化（感知器）。

## Key Facts / Claims

### 贝叶斯分类器
- 后验概率公式：$p(w_i|x) = \frac{p(x|w_i)p(w_i)}{p(x)}$
- 等方差等先验时，决策边界退化为最近均值分类器：$x = \frac{m_1 + m_2}{2}$
- 不等协方差时，决策边界为二次曲面

### Fisher 线性判别分析 (LDA)
- 目标：最大化类间散度 / 类内散度比率
- Fisher 准则：$J_F = \frac{(\mathbf{w}^T \mathbf{m}_1 - \mathbf{w}^T \mathbf{m}_2)^2}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$
- 最优投影：$\mathbf{w} = \gamma \mathbf{S_W^{-1}} (\mathbf{m}_1 - \mathbf{m}_2)$
- 转化为广义特征值问题：$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$

### 感知器 (Perceptron)
- 线性二分类器，决策函数：$f(x) = \text{sign}(w \cdot x + b)$
- 更新规则：$w_{k+1} = w_k + yx$（误分类时）
- 收敛性：有限步内收敛，更新次数上限 $M \leq \frac{\| w_0 - \alpha w^* \|^2}{\beta^2}$

### 支持向量机 (SVM)
- 最大化间隔：$y_k \left( \frac{\langle w, x_k \rangle}{\|w\|} - b \right) \geq \Delta$
- 对偶问题：$\max_{\alpha \geq 0} \left[ \sum \alpha_k - \frac{1}{2} \sum \alpha_k \alpha_l y_k y_l \langle \phi(x_k), \phi(x_l) \rangle \right]$
- 核技巧：通过特征映射 $\phi$ 隐式处理高维空间

## Related
- [[flash-attention]] — 现代深度学习中的注意力优化
- [[transformer]] — 基于注意力的大模型架构
- [[kl-divergence]] — 概率分布间的距离度量
- [[llm-rl-algorithms]] — 强化学习中的优化理论
- [[sft-vs-rlhf]] — SFT 的统计学本质

## Sources
- [贝叶斯分类和决策边界](../sources/2024-08-31-BayesianClassification.md) — 2024-08-31
- [Fisher 线性判别分析](../sources/2024-08-31-Fisher.md) — 2024-08-31
- [Perceptron(感知器)](../sources/2024-08-31-Perceptron.md) — 2024-08-31
- [支持向量机](../sources/2024-09-01-SVM.md) — 2024-09-01
