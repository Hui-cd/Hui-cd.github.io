# Transformer

> 基于自注意力机制的序列建模架构，现代大语言模型的基础

## Overview

Transformer 彻底改变了 NLP 和深度学习领域。辉少的面试笔记中详细描述了其计算流程，博客中也有相关文章。核心创新是完全抛弃循环结构，依赖自注意力机制并行捕捉长距离依赖。

## Key Facts / Claims

### 计算流程
1. **输入嵌入 + 位置编码**：$Input = Embedding(x) + PositionalEncoding(pos)$
2. **编码器**：多头自注意力 + 前馈网络，每层后有 Add & Norm
3. **解码器**：带掩码的自注意力 + 交叉注意力 + 前馈网络
4. **输出**：Linear + Softmax 生成概率分布

### 核心公式
- **自注意力**：$Attention(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- **多头**：$MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O$

### 训练 vs 推理
- **训练**：使用 Teacher Forcing + Causal Mask，一次性并行计算所有位置
- **推理**：严格自回归，逐 token 生成
- Causal Mask 确保预测 $x_t$ 时只能看到 $x_{<t}$

### 复杂度
- 时间：$O(n^2 \cdot d)$（序列长度的平方）
- 内存：$O(n^2)$（注意力矩阵）

## Related
- [[flash-attention]] — 注意力的 IO 优化实现
- [[infini-attention]] — 无限长序列的注意力方案
- [[diffusion-model]] — Transformer 作为扩散模型的骨干
- [[qwen-series]] — Qwen3 的 Hybrid Attention
- [[deepseek]] — DeepSeek 的 MLA 注意力优化
- [[external-blogs]] — 科学空间的 Transformer 分析

## Counter-arguments & Data Gaps
- 二次复杂度限制长序列处理（>8k token 时显存爆炸）
- 位置编码的外推性（训练长度外泛化）仍是开放问题
- 与 RNN、CNN 的混合架构（Mamba, RWKV）的竞争

## Sources
- [Transformer](../sources/Transformer.md) — 未标注日期
- [Transformer基本流程](../sources/Transformer基本流程.md) — 面试笔记
