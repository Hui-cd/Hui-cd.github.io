# Quantization

> 大语言模型量化技术：AWQ、GPTQ、QLoRA 等方法的原理与对比

## Overview

量化是将大语言模型从高精度（FP16/FP32）转换为低精度（INT8/INT4）表示的技术，用于减少显存占用和提升推理速度。辉少的面试笔记详细解释了 AWQ（Activation-aware Weight Quantization）的核心原理。

## Key Facts / Claims

### AWQ (Activation-aware Weight Quantization)

**核心发现**：权重的重要性不取决于权重本身的数值大小，而是取决于它所处理的输入信号（激活值）的大小。

**数学机制**：
- 引入缩放系数 s（针对每个输入通道）：
  $$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$
- 激活侧缩小：X' = X / s
- 权重侧放大：W' = W · s
- 对放大后的权重进行量化：$Q(W') = \text{Int4}(W \cdot s)$

**技术优势**：
- 无梯度优化（不需要反向传播或 Hessian 矩阵）
- 硬件友好（缩放系数可融合到 LayerNorm 或线性操作中）
- 量化速度快（几分钟完成 7B 模型）

### 数值示例（FP16 → INT2）

| 方法 | 输入 | 权重 | 输出 | 误差 |
|------|------|------|------|------|
| 真实值 | [100, 1] | [0.3, 0.3] | 30.3 | - |
| 普通量化 | [100, 1] | [0, 0] | 0 | 30.3 |
| AWQ | [25, 1] | [1, 0] | 25 | 5.3 |

### 量化方法对比

| 方法 | 核心思想 | 是否需要梯度 | 速度 | 精度 |
|------|---------|------------|------|------|
| RTN | 四舍五入到最近值 | 否 | 最快 | 最低 |
| GPTQ | 利用 Hessian 矩阵逐层量化 | 否 | 中等 | 高 |
| AWQ | 激活感知，保护重要权重 | 否 | 快 | 高 |
| QLoRA | 量化 + LoRA 微调 | 是 | 慢 | 最高 |

## Related
- [[flash-attention]] — 另一项推理优化技术
- [[pagedattention]] — vLLM 的显存管理优化
- [[transformer]] — 需要量化的基础架构
- [[deepspeed]] — 训练时的显存优化
- [[qwen-series]] — Qwen3 的 MoE 量化部署

## Counter-arguments & Data Gaps
- AWQ vs GPTQ 在超大模型（70B+）上的对比
- INT4 量化对长上下文（>32k）的影响
- 量化与 MoE 架构的结合

## Sources
- [AWQ 量化详解](../sources/awq.md) — 面试笔记 Day5
