# DeepSpeed

> 微软的大规模模型训练优化库，通过 ZeRO 技术实现显存高效利用

## Overview

DeepSpeed 是训练大语言模型的核心基础设施。辉少的面试笔记详细解释了 ZeRO（Zero Redundancy Optimizer）的三个阶段，以及如何在有限显存下训练超大规模模型。

## Key Facts / Claims

### 显存消耗构成
混合精度训练（FP16/BF16）中：
- **模型参数**：FP16 格式权重
- **梯度**：FP16 格式
- **优化器状态**：FP32 动量、方差、主权重（占模型参数 3-4 倍）

### ZeRO 三阶段

| 阶段 | 切分内容 | 显存节省 | 通信开销 | 适用场景 |
|------|---------|---------|---------|---------|
| ZeRO-1 | 优化器状态 | 4x | 低 | 大多数 LLM 微调首选 |
| ZeRO-2 | 优化器+梯度 | 8x | 低 | 显存稍紧缺时 |
| ZeRO-3 | 优化器+梯度+参数 | 线性 N 倍 | 高（~50%） | 超大模型预训练（70B+） |

### ZeRO-Offload
- 将优化器状态和部分参数下放到 CPU RAM
- 利用 CPU 进行权重更新
- PCIe 带宽成为瓶颈，速度变慢

### 工程实践
- **ZeRO-1 最常用**：性价比最高，通信开销几乎无增加
- **ZeRO-3 仅在必要时**：70B+ 参数模型，用时间换空间

## Related
- [[llm-rl-algorithms]] — 大模型训练中的 RL 阶段
- [[grpo-global]] — 多 GPU 训练设置
- [[transformer]] — 需要优化的基础架构
- [[quantization]] — 另一项显存优化技术
- [[flash-attention]] — 注意力计算优化

## Counter-arguments & Data Gaps
- DeepSpeed vs FSDP（PyTorch 原生）的对比
- ZeRO-Infinity（NVMe 卸载）的适用场景
- 通信优化：All-Reduce vs Reduce-Scatter + All-Gather

## Sources
- [DeepSpeed 是什么 三个阶段又是什么](../sources/deepspeed.md) — 面试笔记 Day3
