# GRPO-Global

> 辉少的项目：改进 GRPO 的 Batch-Level 归一化，解决代码修复任务中的信用分配问题

## Overview

GRPO-Global 是辉少针对软件工程（SWE）任务对原生 GRPO 的改进。核心问题是：原生 GRPO 在组内归一化时，遇到全对或全错的极端情况会导致梯度消失。GRPO-Global 将归一化范围扩大到整个 Batch，并设计了八维复合奖励信号解决稀疏奖励问题。

## Key Facts / Claims

### 核心改进：Batch-Level Advantage Normalization
- **原生 GRPO 问题**：$A_i = \frac{r_i - \text{mean}(r_{group})}{\text{std}(r_{group})}$
  - 全对（全 1）或全错（全 0）时，std = 0，梯度消失
- **GRPO-Global 方案**：$A_{i,j} = \frac{r_{i,j} - \text{mean}(R_{batch})}{\text{std}(R_{batch})}$
  - 跨问题对比，即使组内一致，只要与 Batch 平均有差异就有梯度

### SWE-LIM 数据筛选
- **目标**：从海量 GitHub Issue 中筛选高质量 Bug-Patch 对
- **评分**：
  - Patch 复杂度：$Score_{patch} = \max(0, 1.0 - \frac{Length_{patch}}{10000} \times 0.5)$
  - 问题清晰度：$Score_{stmt} = \min(1.0, \frac{Length_{stmt}}{500})$

### 八维复合奖励信号
1. Patch Validation (0.30) — 代码能否应用并通过测试
2. Format Match (0.10) — diff 格式标记
3. Code Format (0.10) — `<think>` 标签
4. Diff Accuracy (0.10) — 与标准答案文本相似度
5. Reasoning Step (0.10) — 思维链标记
6. Repetition Penalty (0.05) — n-gram 重复惩罚
7. Length Constraints (0.25) — 惩罚冗余代码

### 实验结果
- SWE-LIM 在 Django 仓库解决率 84.2%，但其他仓库 0%（过拟合）
- GRPO-G 在 GSM8K 上比原生 GRPO 收敛后 Reward 高 0.15，方差更小
- SWE-Bench Lite 最终 Pass Rate：0.33%（1/300）
  - 原因：8k Context 窗口太小；单轮生成缺乏编译器反馈

## Related
- [[llm-rl-algorithms]] — PPO/DPO/GRPO 基础
- [[deepseek]] — DeepSeek-R1 使用 GRPO
- [[sft-vs-rlhf]] — RLHF 训练流程
- [[deepspeed]] — 多 GPU 训练优化
- [[projects-overview]] — 辉少的项目索引

## Counter-arguments & Data Gaps
- SWE-LIM 的泛化能力严重不足
- 8k Context 是主要瓶颈，需要扩展到 32k/128k
- 需要多轮 Agent 交互（编译器反馈 → 自我修正）

## Sources
- [GRPO-Global 技术解构](../sources/grpo-global.md) — 项目笔记
