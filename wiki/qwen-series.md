# Qwen Series

> 通义千问系列模型的演进：从 Qwen1 到 Qwen3 的架构创新

## Overview

Qwen（通义千问）是阿里巴巴开源的大语言模型系列。辉少的面试笔记详细追踪了从 Qwen1 到 Qwen3 的演进，特别是 Qwen3 引入的高稀疏 MoE、双模式推理、多模态 Agent 能力等重大改进。

## Key Facts / Claims

### Qwen3 核心改进

**1. 高稀疏 MoE 架构**
- Qwen3-Next-80B-A3B：总参数 80B，激活仅 3B
- 性能对标 Qwen2.5-72B，推理成本仅 10%
- 原生支持多 Token 预测（Multi-token Prediction）

**2. 双模式推理（Thinking Mode）**
- **Instruct Mode**：快速响应，类似 Qwen2
- **Thinking Mode**：深度思考，类似 o1/R1，自动展开 CoT
- 预训练阶段引入大规模 RL 验证

**3. Qwen3-VL：从感知到行动**
- **DeepStack ViT**：融合多层特征，解决小物体检测和密集文字识别
- **Interleaved-MRoPE**：支持超长视频（小时级）时间戳定位
- **Visual Action**：直接操作 GUI（点击、滑动）
- **Visual Coding**：将 UI 设计图转为 HTML/CSS/JS

**4. Hybrid Attention**
- 结合 Global Attention + Sliding Window Attention
- 处理 1M+ 上下文时 KV Cache 大幅降低
- 解决超长文本后段"注意力迷失"问题

### 性能对标

| Qwen3 模型 | 约等于 Qwen2.5 性能 | 核心优势 |
|-----------|-------------------|---------|
| Qwen3-32B (Dense) | > Qwen2.5-72B | 单卡显存获旗舰效果 |
| Qwen3-30B-A3B (MoE) | ≈ Qwen2.5-14B/32B | 极高吞吐量，边缘部署 |
| Qwen3-Next-80B-A3B | > Qwen2.5-72B | 激活仅 3B，大规模并发 |

## Related
- [[deepseek]] — 另一大国产开源模型系列
- [[llm-rl-algorithms]] — Qwen3 使用的 RL 训练
- [[transformer]] — 基础架构
- [[quantization]] — Qwen3 的量化部署
- [[flash-attention]] — 注意力优化
- [[external-blogs]] — 科学空间的 Transformer 分析

## Counter-arguments & Data Gaps
- Qwen3 的 MoE 路由机制细节
- Thinking Mode 的 RL 训练流程（类似 DeepSeek-R1？）
- 与 LLaMA、Mistral 系列的直接对比

## Sources
- [Qwen 系列笔记](../sources/qwen-series/) — 面试笔记
