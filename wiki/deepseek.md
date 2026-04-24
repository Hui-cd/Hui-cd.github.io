# DeepSeek

> DeepSeek 系列模型的核心技术：MLA、DeepSeekMoE、Decoupled RoPE、Yarn、R1 训练流水线

## Overview

DeepSeek 是当前最具影响力的开源大模型系列之一。辉少的面试笔记详细拆解了其核心技术创新：MLA（低秩压缩注意力）、MoE 架构、解耦 RoPE 位置编码、以及 R1 的纯 RL 训练流程。

## Key Facts / Claims

### MLA (Multi-Head Latent Attention)
- **核心**：低秩压缩 KV Cache，将 Key-Value 投影到低维潜向量 $C_{KV}$
- **矩阵吸收**：推理时将 Query 投影矩阵与 $W_{UK}$ 预先合并，直接用低维潜向量计算注意力分数
- **问题**：与标准 RoPE 不兼容（矩阵乘法不满足交换律）
- **解决**：Decoupled RoPE（解耦 RoPE）

### Decoupled RoPE
- **内容通道**：主体向量（如 128 维），无位置信息，可用 MLA 压缩
- **位置通道**：额外小维度向量（如 64 维），显式应用 RoPE
- **最终分数**：$Score = (q_{content} \cdot k_{content}^T) + (q_{rope}^{rot} \cdot k_{rope}^{rot T})$

### Yarn (Yet Another RoPE Extension)
- **目标**：支持 128k 超长上下文
- **分频段处理**：
  - 高频（前部）：不插值，保持局部分辨率
  - 低频（后部）：线性插值，$\theta' = \theta_{old}/s$，拉长周期
- **长度缩放**：$Score_{final} = Score_{raw} \cdot (0.1 \ln(s) + 1)$

### DeepSeekMoE
- **257 个专家**：256 路由专家 + 1 共享专家
- **每 Token 激活**：8 个路由专家
- **优势**：相比 Dense 模型大幅减少激活参数量；共享专家缓解路由坍缩

### DeepSeek-R1 训练流水线（四阶段）
1. **冷启动**：数千条高质量长思维链 SFT，教会模型 `<think>` 格式
2. **推理导向 RL**：GRPO + 规则奖励（准确性 + 格式），涌现自我反思能力
3. **拒绝采样 + 大规模 SFT**：用阶段 2 模型生成 60 万条数据，筛选后重训练
4. **全场景对齐**：混合奖励模型（规则 + 人类偏好），平衡推理与通用能力

## Related
- [[llm-rl-algorithms]] — GRPO 算法详解
- [[flash-attention]] — 注意力优化
- [[transformer]] — 基础架构
- [[qwen-series]] — 另一大国产开源模型系列
- [[grpo-global]] — 辉少对 GRPO 的改进项目

## Counter-arguments & Data Gaps
- MLA 的压缩率与精度损失的定量分析
- Yarn 在极端长度（>200k）上的效果
- R1 的纯 RL 涌现是否可复现于其他模型规模

## Sources
- [DeepSeek 系列笔记](../sources/deepseek-series/) — 面试笔记
