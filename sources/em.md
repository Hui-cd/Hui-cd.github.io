# EM算法详解与数学推导

EM（Expectation-Maximization）算法是一种迭代算法,用于在存在隐变量（latent variables）的情况下进行参数估计。本文将详细介绍EM算法的原理和完整的数学推导过程。

## 1. EM算法的基本思想

EM算法的核心思想是通过迭代来最大化似然函数的下界,从而间接地最大化似然函数。它包括两个主要步骤:

1. E步(Expectation): 计算似然函数的期望
2. M步(Maximization): 最大化这个期望

## 2. 数学符号和定义

- $X$: 观测变量
- $Z$: 隐变量
- $\theta$: 模型参数
- $L(\theta)=\log p(X|\theta)$: 对数似然函数
- $q(Z)$: 隐变量$Z$的概率分布

## 3. 数学推导

### 3.1 引入隐变量

我们首先引入隐变量$Z$,将对数似然函数表示为:

$$L(\theta) = \log p(X|\theta) = \log \sum_Z p(X,Z|\theta)$$

### 3.2 应用Jensen不等式

引入一个分布$q(Z)$,应用Jensen不等式:

$$\begin{align*}
L(\theta) &= \log \sum_Z p(X,Z|\theta) \\
&= \log \sum_Z q(Z) \frac{p(X,Z|\theta)}{q(Z)} \\
&\geq \sum_Z q(Z) \log \frac{p(X,Z|\theta)}{q(Z)} \\
&= \sum_Z q(Z) \log p(X,Z|\theta) - \sum_Z q(Z) \log q(Z) \\
&= E_q[\log p(X,Z|\theta)] + H(q)
\end{align*}$$

其中$H(q)$是$q(Z)$的熵。

### 3.3 定义辅助函数

定义辅助函数$Q(\theta, \theta^{(t)})$:

$$Q(\theta, \theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$$

这里$\theta^{(t)}$是第$t$次迭代的参数估计。

### 3.4 EM算法的迭代步骤

EM算法的迭代步骤如下:

1. E步: 计算$Q(\theta, \theta^{(t)})$
   $$Q(\theta, \theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$$

2. M步: 最大化$Q(\theta, \theta^{(t)})$以更新参数
   $$\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$$

### 3.5 算法收敛性证明

为了证明EM算法的收敛性,我们需要证明每次迭代都能增加似然函数的值。

设$q^{(t+1)}(Z) = p(Z|X,\theta^{(t)})$,则:

$$\begin{align*}
L(\theta^{(t+1)}) - L(\theta^{(t)}) &\geq Q(\theta^{(t+1)}, \theta^{(t)}) - Q(\theta^{(t)}, \theta^{(t)}) \\
&\quad + \sum_Z q^{(t+1)}(Z) \log \frac{q^{(t+1)}(Z)}{p(Z|X,\theta^{(t+1)})} \\
&\quad - \sum_Z q^{(t+1)}(Z) \log \frac{q^{(t+1)}(Z)}{p(Z|X,\theta^{(t)})} \\
&\geq 0
\end{align*}$$

这证明了EM算法的每次迭代都能增加似然函数的值,从而保证了算法的收敛性。

## 4. EM算法的一般步骤

1. 初始化参数$\theta^{(0)}$
2. 重复以下步骤直到收敛:
   - E步: 计算$Q(\theta, \theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$
   - M步: 求解$\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$
3. 输出最终估计的参数$\theta$

## 5. 总结

EM算法通过迭代的方式最大化似然函数的下界,从而间接地最大化似然函数。它在处理含有隐变量的问题时特别有效,如混合高斯模型、隐马尔可夫模型等。然而,EM算法也有一些局限性,如可能收敛到局部最优解,收敛速度可能较慢等。在实际应用中,可能需要结合其他技术来改进算法性能。