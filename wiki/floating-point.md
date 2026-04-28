# Floating Point Formats

> FP16、BF16 等低精度浮点格式在深度学习中的权衡。

## Overview

深度学习训练通常使用 32 位浮点数（FP32），但模型规模的增长推动了对低精度格式的需求。FP16 和 BF16 是两种主流的 16 位格式，在存储效率、计算速度和数值精度之间做出不同权衡。

## Key Facts / Claims

### FP16（IEEE 754 Half Precision）

- **结构**：1 位符号 + 5 位指数 + 10 位尾数
- **优点**：存储减半，在支持 Tensor Core 的 GPU 上计算速度提升 2-8 倍
- **缺点**：
  - 动态范围小（指数位少），容易上溢/下溢
  - 精度有限（尾数位少），梯度更新可能丢失信息

### BF16（Brain Floating Point）

- **结构**：1 位符号 + **8 位指数** + 7 位尾数
- **设计来源**：Google Brain 为 TPU 设计
- **优点**：
  - 指数位与 FP32 相同，动态范围与 FP32 一致，**不易溢出**
  - 无需特殊损失缩放（Loss Scaling）即可稳定训练
- **缺点**：尾数精度比 FP16 更低，但深度学习对精度不敏感

### 对比总结

| 特性 | FP16 | BF16 | FP32 |
|------|------|------|------|
| 符号位 | 1 | 1 | 1 |
| 指数位 | 5 | 8 | 8 |
| 尾数位 | 10 | 7 | 23 |
| 动态范围 | 小 | 与 FP32 相同 | 大 |
| 精度 | 较高 | 较低 | 最高 |
| 需 Loss Scaling | 是 | 否 | 否 |
| 硬件支持 | NVIDIA Pascal+ | NVIDIA Ampere+ / TPU | 通用 |

### 训练实践

- **混合精度训练**：FP16/BF16 前向+反向，FP32 主权重 + 损失缩放
- **自动混合精度（AMP）**：PyTorch/TensorFlow 自动选择合适精度
- 现代大模型（GPT、LLaMA）训练普遍采用 BF16 或 FP8

## Related

- [[deepspeed]] — ZeRO 优化中的 FP16/BF16 混合精度训练
- [[quantization]] — 推理阶段的 INT8/INT4 量化是更低精度的延伸
- [[flash-attention]] — 注意力计算中的数值稳定性与精度选择

## Sources

- [Float Point](https://hui-cd.github.io/Float_point/) — 辉少的笔记原文
