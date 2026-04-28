# PagedAttention

> vLLM 的核心创新：将操作系统的分页机制引入 GPU 显存管理，解决 KV Cache 碎片化问题

## Overview

PagedAttention 是 vLLM 推理引擎的核心创新。它不是改变 Attention 的数学公式，而是一项系统工程创新：将 OS 的虚拟内存/分页机制引入 GPU 显存管理，使显存利用率从 20-40% 提升到接近 100%。

## Key Facts / Claims

### 核心痛点
传统 Transformer 推理中，KV Cache 需要**预先分配连续显存**：
- 假设最大长度 2048，即使只生成 50 个 token，也要预留 2048 位置
- 导致三种碎片：预留浪费、内部碎片、外部碎片
- 显存实际利用率仅 **20-40%**

### 核心思想：解耦逻辑与物理空间
- **逻辑块 (Logical Block)**：token 序列在逻辑上连续
- **物理块 (Physical Block)**：在 GPU 显存中可完全打散
- **块表 (Block Table)**：记录逻辑到物理的映射

### 算法实现
- Block Size 通常为 16 或 32
- 手写 CUDA Kernel 支持分页读取（非连续内存寻址）
- Attention 计算时通过查表聚合分散数据

### Copy-on-Write 显存共享
- Parallel Sampling 和 Beam Search 中，Prompt 部分共享物理块
- 只有生成不同内容时才分配新块
- **数值收益**：Prompt 1024 + 生成 128 × Beam 4，显存节省 **66%**

### 性能提升
- 显存浪费降至接近 0%
- 吞吐量提升 **2-4 倍**（vs HuggingFace TGI 旧版）
- 单次延迟基本不变（甚至微增），但通过更大 Batch Size 提升吞吐

## Related
- [[flash-attention]] — 另一项注意力优化（计算角度）
- [[transformer]] — 基础架构
- [[vllm]] — 实现 PagedAttention 的推理引擎
- [[quantization]] — 模型部署优化
- [[deepspeed]] — 训练时的显存优化

## Counter-arguments & Data Gaps
- PagedAttention 与 FlashAttention 的结合（vLLM 已实现）
- 与 Continuous Batching 的关系
- 在超长上下文（>100k）上的扩展性

## Sources
- [讲一下 pageattention 的思想和做法](../sources/pagedattention.md) — 面试笔记 Day4
