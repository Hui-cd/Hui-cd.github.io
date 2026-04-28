# Flow Matching

> 通过速度场回归实现生成模型训练，SD3/FLUX 的核心技术

## Overview

Flow Matching（流匹配）是训练连续归一化流（CNF）的通用框架，已成为 Stable Diffusion 3、FLUX.1、Voicebox 等前沿模型的核心训练目标。与扩散模型「去噪」不同，Flow Matching 直接学习将简单分布（高斯噪声）通过速度场平滑地「流」向数据分布。

## Key Facts / Claims

### 核心思想：免模拟训练
1. 预定义从噪声 x_0 到数据 x_1 的概率路径
2. 计算路径在任意时刻的理想速度场
3. 训练神经网络 v_θ 拟合速度场

### 数学机制
- **ODE 定义**：$\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x))$
- **条件流匹配**：引入条件路径，学习条件速度场 $u_t(z|x_1)$
- **损失函数**：$\mathcal{L}_{CFM} = \mathbb{E}[||v_\theta(z_t, t) - u_t(z_t|x_1)||^2]$

### 最优传输路径（Linear Path）
- **条件路径**：$z_t = (1-t)x_0 + tx_1$
- **目标速度**：$u_t = x_1 - x_0$（恒定向量，指向目标）
- **几何解释**：拉直生成路径，可用大步长直接跨越

### 与扩散模型对比

| 特性 | Flow Matching (OT-CFM) | Diffusion (DDPM) |
|------|------------------------|------------------|
| 训练目标 | 回归速度场 v | 回归噪声 ε |
| 路径形状 | 直线 | 弧线 |
| 采样效率 | 高（10-50 steps） | 低（50-1000 steps）|
| 耦合方式 | 可自定义（最优传输） | 固定（高斯乘积） |

### 关键陷阱
- **轨迹交叉**：随机配对 x_0 和 x_1 会导致路径交叉，速度场冲突
- **解决**：使用最优传输（OT），将最近的噪声点和数据点配对

### 应用
- **图像生成**：SD3、FLUX.1（Rectified Flow）
- **语音合成**：Voicebox（非高斯先验）
- **蛋白质折叠**：FoldingDiff（黎曼流形扩展）

## Related
- [[diffusion-model]] — DDPM 与 Score Matching
- [[quantization]] — 模型部署优化
- [[vae]] — 另一大生成模型范式
- [[external-blogs]] — Yang Song 的 Score Matching 博客

## Counter-arguments & Data Gaps
- FM 在文本生成上的应用（主要是图像/语音）
- 与 Schrödinger Bridge 的关系
- 大步长下的数值稳定性

## Sources
- [Flow Matching (FM) 详解](../sources/flow-matching.md) — 面试笔记 Day6
