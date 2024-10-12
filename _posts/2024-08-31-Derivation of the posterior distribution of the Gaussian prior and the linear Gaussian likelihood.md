---
single: post
title:  "高斯先验和线性高斯似然的后验分布推导"
date:   2024-10-11 22:50:22 +1000
categories: 
    - machine-learning
author: Hui
---


我们有以下已知条件：

**先验分布**：

$$
p(x) = N(\mu_A, \sigma_A^2)
$$

**似然函数**：

$$
p(y \mid x) = N(ax, \sigma_B^2)
$$

我们的目标是推导后验分布 $p(x \mid y)$，证明它也是一个高斯分布，并找到其均值和方差。

## 步骤 1：写出先验和似然的概率密度函数

**先验分布**：

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma_A^2}} \exp\left(-\frac{(x - \mu_A)^2}{2\sigma_A^2}\right)
$$

**似然函数**：

$$
p(y \mid x) = \frac{1}{\sqrt{2\pi\sigma_B^2}} \exp\left(-\frac{(y - ax)^2}{2\sigma_B^2}\right)
$$

## 步骤 2：应用贝叶斯定理得到后验分布的非规范化形式

根据贝叶斯定理：

$$
p(x \mid y) \propto p(y \mid x) p(x)
$$

因此：

$$
p(x \mid y) \propto \exp\left(-\frac{(y - ax)^2}{2\sigma_B^2} - \frac{(x - \mu_A)^2}{2\sigma_A^2}\right)
$$

## 步骤 3：展开指数中的二次项

展开似然函数中的二次项：

$$
(y - ax)^2 = y^2 - 2axy + a^2x^2
$$

展开先验分布中的二次项：

$$
(x - \mu_A)^2 = x^2 - 2\mu_Ax + \mu_A^2
$$

将这些展开代入后验分布的指数中：

$$
-\frac{1}{2\sigma_B^2}(y^2 - 2axy + a^2x^2) - \frac{1}{2\sigma_A^2}(x^2 - 2\mu_Ax + \mu_A^2)
$$

## 步骤 4：整理关于 $x$ 的二次项、一阶项和常数项

关于 $x^2$ 的项：

$$
-\frac{a^2x^2}{2\sigma_B^2} - \frac{x^2}{2\sigma_A^2} = -\frac{(a^2/\sigma_B^2 + 1/\sigma_A^2)}{2}x^2
$$

关于 $x$ 的项：

$$
\frac{axy}{\sigma_B^2} + \frac{\mu_Ax}{\sigma_A^2} = \left(\frac{ay}{\sigma_B^2} + \frac{\mu_A}{\sigma_A^2}\right)x
$$

常数项（与 $x$ 无关，可以忽略）：

$$
-\frac{y^2}{2\sigma_B^2} - \frac{\mu_A^2}{2\sigma_A^2}
$$

## 步骤 5：将指数部分写成关于 $x$ 的完全平方形式

令：

$$
A = \frac{a^2}{\sigma_B^2} + \frac{1}{\sigma_A^2}
$$

$$
B = \frac{2ay}{\sigma_B^2} + \frac{2\mu_A}{\sigma_A^2}
$$

则指数部分可写为：

$$
-\frac{A}{2}x^2 + \frac{B}{2}x
$$

## 步骤 6：完成平方

考虑表达式：

$$
-\frac{A}{2}x^2 + \frac{B}{2}x = -\frac{A}{2}\left(x^2 - \frac{B}{A}x\right)
$$

完成平方：

$$
x^2 - \frac{B}{A}x = \left(x - \frac{B}{2A}\right)^2 - \left(\frac{B}{2A}\right)^2
$$

因此：

$$
-\frac{A}{2}\left(x^2 - \frac{B}{A}x\right) = -\frac{A}{2}\left(\left(x - \frac{B}{2A}\right)^2 - \left(\frac{B}{2A}\right)^2\right)
$$

$$
= -\frac{A}{2}\left(x - \frac{B}{2A}\right)^2 + \frac{A}{2}\left(\frac{B}{2A}\right)^2
$$

常数项仍可忽略。

## 步骤 7：写出后验分布

因此，后验分布的非规范化形式为：

$$
p(x \mid y) \propto \exp\left(-\frac{A}{2}\left(x - \frac{B}{2A}\right)^2\right)
$$

这表示后验分布是一个均值为 $\tilde{\mu} = \frac{B}{2A}$、方差为 $\tilde{\sigma}^2 = \frac{1}{A}$ 的高斯分布。

## 步骤 8：明确后验均值和方差的表达式

**后验方差**

$$
\tilde{\sigma}^2 = \frac{1}{A} = \frac{1}{\frac{a^2}{\sigma_B^2} + \frac{1}{\sigma_A^2}} = \left(\sigma_A^{-2} + a^2\sigma_B^{-2}\right)^{-1}
$$

因此：

$$
\tilde{\sigma}^{-2} = \sigma_A^{-2} + a^2\sigma_B^{-2}
$$

**后验均值**：

$$
\tilde{\mu} = \frac{B}{2A} = \tilde{\sigma}^2 \left(\frac{ay}{\sigma_B^2} + \frac{\mu_A}{\sigma_A^2}\right)
$$

因此：

$$
\tilde{\mu} = \tilde{\sigma}^2 \left(\sigma_A^{-2}\mu_A + a\sigma_B^{-2}y\right)
$$

## 步骤 9：总结

我们得到了后验分布的参数：

**后验分布**：
$$
p(x \mid y) = N(\tilde{\mu}, \tilde{\sigma}^2)
$$

**后验方差**：
$$
\tilde{\sigma}^{-2} = \sigma_A^{-2} + a^2\sigma_B^{-2}
$$

**后验均值**：
$$
\tilde{\mu} = \tilde{\sigma}^2 \left(\sigma_A^{-2}\mu_A + a\sigma_B^{-2}y\right)
$$
