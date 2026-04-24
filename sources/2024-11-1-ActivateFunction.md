---
single: post
title:  "常见激活函数详解"
date:   2024-11-01 17:50:22 +1000
categories: 
    - deep-learning
author: Hui
---

## 1. Sigmoid

公式：
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**优点**：
- 输出范围在 $(0, 1)$ 之间，适合处理概率相关的问题。
- 在早期的神经网络中被广泛使用。

**缺点**：
- 容易导致梯度消失问题，特别是在深层网络中。
- 输出非零中心，可能影响训练效率。

---

## 2. Tanh

公式：
$$
\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**优点**：
- 输出范围在 $(-1, 1)$ 之间，零均值，有助于加速收敛。
- 相比 Sigmoid，Tanh 的梯度消失问题有所缓解。

**缺点**：
- 在极端输入下，仍可能导致梯度消失。

---

## 3. ReLU

公式：
$$
\text{ReLU}(x) = \max(0, x)
$$

**优点**：
- 计算简单，高效。
- 缓解梯度消失问题，促进深层网络的训练。

**缺点**：
- 可能导致“神经元死亡”问题，即某些神经元在训练中永远不再激活。

---

## 4. Leaky ReLU

公式：
$$
\text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
$$

**优点**：
- 通过在负半轴引入一个小的斜率 $\alpha$，缓解了 ReLU 的“神经元死亡”问题。

**缺点**：
- 需要手动设置 $\alpha$ 值，可能需要调参。

---

## 5. PReLU

公式：
$$
\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
$$

**优点**：
- $\alpha$ 是可学习的参数，模型可以根据数据自动调整。

**缺点**：
- 增加了模型的参数量，可能导致过拟合。

---

## 6. ELU

公式：
$$
\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases}
$$

**优点**：
- 负半轴有非零输出，缓解了梯度消失问题。
- 输出均值接近零，有助于加速收敛。

**缺点**：
- 计算相对复杂，增加了计算成本。

---

## 7. SELU

公式：
$$
\text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases}
$$

**优点**：
- 具有自归一化属性，能使网络层的输出保持零均值和单位方差。

**缺点**：
- 对权重初始化和输入数据有特定要求，通常需要 **Lecun 正态初始化** 和标准化输入。

---

## 8. Swish

公式：
$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

**优点**：
- 平滑且非单调，表现优于 ReLU。

**缺点**：
- 计算复杂度高于 ReLU。在资源受限的硬件上可能不适合直接使用。

---

## 9. GELU

公式：
$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

其中，$\Phi(x)$ 是标准正态分布的累积分布函数。

**优点**：
- 在 Transformer 等模型中表现出色。

**缺点**：
- 计算复杂度较高。

---

## 10. Softplus

公式：
$$
\text{Softplus}(x) = \ln(1 + e^x)
$$

**优点**：
- 平滑的 ReLU 近似，避免了 ReLU 的“神经元死亡”问题。

**缺点**：
- 计算复杂度高于 ReLU。

---

## 11. Softmax

公式：
$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

**优点**：
- 将输入转换为概率分布，适用于多分类问题的输出层。

**缺点**：
- 计算复杂度高，容易导致数值不稳定。通常用在最后一层的输出。

---

## 12. Hard Sigmoid

公式：
$$
\text{Hard Sigmoid}(x) = \begin{cases} 0 & \text{if } x \leq -2.5 \\ 1 & \text{if } x \geq 2.5 \\ 0.2x + 0.5 & \text{otherwise} \end{cases}
$$

**优点**：
- 计算效率高，因为它是 Sigmoid 的近似形式，用于硬件加速或计算资源受限的设备上。

**缺点**：
- 相比标准 Sigmoid，它的输出较不平滑，可能导致梯度不稳定。

---

## 13. Maxout

Maxout 是一种不太常见的激活函数，其输入是多个线性变换的最大值。

公式：
$$
\text{Maxout}(x) = \max(w_1 \cdot x + b_1, w_2 \cdot x + b_2, \dots, w_k \cdot x + b_k)
$$

**优点**：
- 具有更高的表达能力，可以学习任意形状的激活模式。
- 可以缓解梯度消失问题。

**缺点**：
- 增加了参数量，计算复杂度较高，不适用于内存受限的设备。

---

## 14. Gaussian

公式：
$$
\text{Gaussian}(x) = e^{-x^2}
$$

**优点**：
- 平滑的激活函数，适合在高斯噪声模型中使用。

**缺点**：
- 随着输入增大或减小，梯度会迅速减小，容易导致梯度消失问题。通常要求输入数据经过标准化以限制其范围。

---

## 15. Tanh、Sigmoid 等

**要求**：
- 这些激活函数对输入范围较为敏感，通常适用于标准化、零均值的输入数据。
- 原因：在大输入值下，这些激活函数容易饱和，导致梯度消失问题。

---

## 16. SELU 和 Batch Normalization 不兼容

- **要求**：SELU 激活函数不建议与 Batch Normalization 一起使用，因为它本身具有自归一化属性，Batch Normalization 会干扰 SELU 的标准化效果。
