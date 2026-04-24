# Infini-Attention

> Google 提出的无限长序列注意力机制，通过压缩记忆实现 $O(1)$ 内存的无限上下文

## Overview

Infini-Attention 解决了标准 Transformer 的 $O(N^2)$ 上下文记忆限制。核心思想是用**压缩记忆 (compressive memory)** 替代随序列长度增长的 KV Cache，通过关联绑定操作实现固定参数量的长期记忆存储和检索。

## Key Facts / Claims

### 压缩记忆检索
用 Query 从记忆库 $M_{s-1}$ 中检索内容：
$$A_{\text{mem}} = \frac{\sigma(Q) M_{s-1}}{\sigma(Q) z_{s-1}}$$
其中 $\sigma$ 是非线性激活，$z_{s-1}$ 是归一化项。

### 记忆更新（两种规则）
1. **线性更新**：$M_s \leftarrow M_{s-1} + \sigma(K)^T V$
2. **Delta 规则**：$M_s \leftarrow M_{s-1} + \sigma(K)^T (V - \frac{\sigma(K) M_{s-1}}{\sigma(K) z_{s-1}})$
   - 键值已存在时保持矩阵不变
   - 跟踪相同归一化项保证数值稳定性

### 长期上下文注入
通过门控标量 $\beta$ 聚合局部注意力和记忆内容：
$$A = \text{sigmoid}(\beta) \odot A_{\text{mem}} + (1 - \text{sigmoid}(\beta)) \odot A_{\text{dot}}$$

### 多头扩展
并行计算 $H$ 个上下文状态，串联投影：
$$O = [A_1; \ldots; A_H] W_O$$

### 实验结果
- 1B 模型解决 100 万 token 的密钥检索任务
- 8B 模型在 500K 书籍摘要任务达到 SOTA
- 内存压缩率 >100x
- 可视化显示专门处理局部/长期/混合的注意力头自然涌现

## Related
- [[flash-attention]] — 标准注意力的 IO 优化
- [[transformer]] — 基础架构
- [[deepseek]] — DeepSeek 的 Yarn 长上下文方案
- [[qwen-series]] — Qwen3 的 Hybrid Attention
- [[external-blogs]] — 科学空间的注意力分析

## Counter-arguments & Data Gaps
- 压缩记忆的容量上限理论分析不足
- 与 Ring Attention、Striped Attention 等长序列方案的直接对比
- 门控标量 $\beta$ 的训练动态和可解释性

## Sources
- [Infini-Attention](../sources/2024-08-31-Infini-attention.md) — 2024-08-31
