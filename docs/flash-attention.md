# Flash Attention

> GPU 上的注意力机制 IO 感知优化算法，通过分块和 online softmax 减少 HBM 读写

## Overview

Flash Attention 系列（v1/v2）解决了 Transformer 中注意力机制的内存瓶颈。标准注意力的 $O(N^2)$ 内存需求和二次时间复杂度限制了长序列处理。Flash Attention 通过**分块计算 (tiling)** 和 **online softmax** 技术，在不近似的情况下实现了 2-4 倍加速，并将内存从 $O(N^2)$ 降至 $O(N)$。

## Key Facts / Claims

### 标准注意力的瓶颈
- 时间复杂度：$O(n^2 \cdot d)$，内存：$O(n^2)$
- 需要物化 $S = QK^T$ 和 $P = \text{softmax}(S)$ 到 HBM
- 反向传播必须保存 $P \in \mathbb{R}^{N \times N}$

### Flash Attention 核心思想
1. **分块加载**：将 $Q, K, V$ 分块加载到 SRAM（片上高速缓存）
2. **Online Softmax**：逐块计算 softmax 统计量（max 和 sum），避免物化完整矩阵
3. **重计算**：反向传播时重新计算 $S$ 和 $P$，而非保存

### Online Softmax 数学
对于分块 $[S(1), S(2)]$：
1. 第一块：$m(1) = \text{rowmax}(S(1))$，$\ell(1) = \text{rowsum}(e^{S(1)-m(1)})$
2. 第二块：$m(2) = \max(m(1), \text{rowmax}(S(2)))$
3. 重新缩放：$\ell(2) = e^{m(1)-m(2)}\ell(1) + \text{rowsum}(e^{S(2)-m(2)})$
4. 输出累加：$O = \text{diag}(\ell)^{-1}(e^{S(1)-m}V(1) + e^{S(2)-m}V(2))$

### Flash Attention-2 改进
- **延迟重新缩放**：保持未缩放输出 $\tilde{O}$，循环结束后再统一缩放
- **减少非矩阵乘法 FLOPs**
- **Warp 划分优化**：对 $Q$ 划分而非 $K,V$，消除 warp 间同步
- **序列长度并行**：长序列时在序列维度增加并行度

### 性能数据
- 前向：理论 FLOPs 的 30-50%（A100）
- 反向：理论 FLOPs 的 25-35%
- 对比 GEMM 的 80-90% 仍有差距（内存带宽限制）

## Related
- [[transformer]] — 注意力机制的核心应用架构
- [[infini-attention]] — 无限长序列的注意力方案
- [[pagedattention]] — vLLM 的另一项推理优化
- [[deepseek]] — DeepSeek 使用 Flash Attention
- [[qwen-series]] — Qwen 系列使用 Flash Attention

## Counter-arguments & Data Gaps
- Flash Attention 对短序列（<1k）收益有限，分块开销可能超过收益
- 不同 GPU 架构（A100 vs H100）的最优块大小需要调优
- 与稀疏注意力、线性注意力的综合对比尚不完整

## Sources
- [Flash Attention2](../sources/2024-09-11-FlashAtten.md) — 2024-09-11
