# Activation Functions

> 神经网络非线性变换的核心组件，从 Sigmoid 到 GELU 的演进史。

## Overview

激活函数为神经网络引入非线性，使其能够逼近任意复杂函数。从早期的 Sigmoid/Tanh 到现代的 ReLU 及其变体，再到 Transformer 时代的 GELU，激活函数的选择深刻影响着网络的训练动态和表达能力。

## Key Facts / Claims

### Sigmoid & Tanh（早期时代）

- **Sigmoid**：$\sigma(x) = \frac{1}{1 + e^{-x}}$，输出 $(0,1)$，适合概率输出
- **Tanh**：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$，输出 $(-1,1)$，零均值
- **共同问题**：两端饱和导致梯度消失；Sigmoid 输出非零中心

### ReLU 家族（深度学习时代）

- **ReLU**：$\max(0, x)$ —— 计算简单，缓解梯度消失，但存在"神经元死亡"问题
- **Leaky ReLU**：负半轴斜率 $\alpha$（如 0.01），缓解死亡问题
- **PReLU**：$\alpha$ 为可学习参数，自适应调整
- **ELU**：负半轴为 $\alpha(e^x - 1)$，输出均值接近零，平滑负半轴
- **SELU**：自带自归一化属性，配合 Lecun 初始化可使层输出保持零均值单位方差

### 现代激活函数（Transformer 时代）

- **Swish**：$x \cdot \sigma(x)$，Google 提出，平滑非单调，表现优于 ReLU
- **GELU**：$x \cdot \Phi(x)$，$\Phi$ 为标准正态 CDF —— **Transformer 标配**
- **Softplus**：$\ln(1 + e^x)$，ReLU 的平滑近似

### 特殊用途

- **Softmax**：多分类输出层，将 logits 转为概率分布
- **Hard Sigmoid**：计算高效的 Sigmoid 近似，用于移动端
- **Maxout**：$\max(w_1 x + b_1, w_2 x + b_2, ...)$，表达能力最强但参数量大

### 使用注意事项

- Sigmoid/Tanh 要求输入标准化（零均值、小方差），避免饱和
- SELU 不建议与 Batch Normalization 同时使用（自归一化被干扰）
- 现代大模型几乎统一使用 GELU 或 SwiGLU（Swish + Gating）

## Related

- [[transformer]] — GELU 在 Transformer FFN 中的应用
- [[machine-learning-basics]] — 感知器的阶跃激活函数是激活函数的雏形
- [[diffusion-model]] — Score 网络中的激活函数选择影响训练稳定性

## Sources

- [常见激活函数详解](https://hui-cd.github.io/2024/11/01/ActivateFunction/) — 辉少的博客原文
