# GEMM

> 通用矩阵乘法（General Matrix Multiply），深度学习计算的核心原语。

## Overview

GEMM（General Matrix Multiply）即 $C = \alpha AB + \beta C$，是 BLAS（Basic Linear Algebra Subprograms）库中的三级运算。在深度学习中，全连接层、卷积层（im2col 后）、注意力计算都最终转化为 GEMM。

优化 GEMM 是提升深度学习训练/推理效率的核心课题。

## Key Facts / Claims

### 计算特性

- 运算强度（Arithmetic Intensity）：$\frac{2mnk}{mn+mk+nk}$ FLOPs/Byte
- 当矩阵足够大时，计算受限（Compute Bound）；小时，内存带宽受限（Memory Bound）
- 现代 GPU 上，优化后的 GEMM 可达到理论峰值 FLOPs 的 80-90%

### 优化技术

1. **Tiling/Blocking**
   - 将大矩阵分块加载到共享内存/SRAM
   - 隐藏内存延迟，提高数据复用

2. **循环重排（Loop Reordering）**
   - 调整 i,j,k 三重循环顺序以匹配硬件内存层次

3. **向量化（Vectorization）**
   - 使用 SIMD/Tensor Core 进行批量计算

4. **双缓冲（Double Buffering）**
   - 计算当前块的同时预取下一个块

### 深度学习中的 GEMM

- **全连接层**：$Y = XW^T + b$
- **卷积（im2col）**：将卷积展开为矩阵乘法
- **注意力**：$QK^T$ 和 $PV$ 都是 GEMM
- **Flash Attention**：通过分块将注意力也转化为片上 GEMM

### 关键库

- **cuBLAS**：NVIDIA GPU 上的标准 GEMM 实现
- **CUTLASS**：NVIDIA 开源的 C++ GEMM 模板库，可自定义分块策略
- **OpenBLAS/MKL**：CPU 端优化库

## Related

- [[flash-attention]] — 将注意力计算转化为分块 GEMM 的核心思想
- [[deepspeed]] — 大规模训练中的计算效率优化
- [[quantization]] — 低精度 GEMM（INT8/INT4）的硬件支持

## Sources

- [GEMM](https://hui-cd.github.io/GEMM/) — 辉少的笔记原文
