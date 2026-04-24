# SFT vs RLHF

> 为什么有 SFT 还需要 RLHF：从行为克隆到全局价值寻优的本质差异

## Overview

SFT（监督微调）和 RLHF（基于人类反馈的强化学习）是大语言模型对齐的两个阶段。辉少的面试笔记从统计学本质、控制论视角、系统动力学三个维度深入剖析了二者的差异。

## Key Facts / Claims

### SFT 的本质：行为克隆
- **目标**：最小化模型分布与经验分布的 KL 散度
- **数学**：$\mathcal{L}_{SFT} = -\mathbb{E}[\sum_{t=1}^T \log \pi_\theta(y_t|x, y_{<t})]$
- **机制**：Token-level 局部贪婪匹配，被动执行「照猫画虎」
- **局限**：解空间受限于训练数据支撑集，缺乏自主探索能力

### RLHF 的本质：全局价值寻优
- **目标**：最大化轨迹预期累积回报
- **数学**：$\max_\theta \mathbb{E}[r_\phi(x,y) - \beta D_{KL}(\pi_\theta || \pi_{SFT})]$
- **机制**：Sequence-level 价值评估，允许在线探索
- **优势**：通过 GAE 实现稀疏动态学分分配

### 核心差异

| 维度 | SFT | RLHF |
|------|-----|------|
| 优化目标 | 局部概率最大化 | 全局回报最大化 |
| 探索能力 | 插值泛化（数据内） | 外推涌现（超越人类） |
| 暴露偏差 | 有（教师强制） | 无（在线采样） |
| 学分分配 | 均匀（所有 token 等权） | 动态（优势函数） |
| 主要风险 | 分布外失效 | 奖励劫持 |

### 为什么需要 RLHF
1. **消除暴露偏差**：训练时基于真实前缀 vs 推理时基于自身生成前缀
2. **稀疏学分分配**：关键 token（连词、算子）应获得更高权重
3. **超越人类演示**：探索数据集中不存在的高阶解答路径
4. **长文本连贯性**：在线采样使长序列生成更鲁棒

## Related
- [[llm-rl-algorithms]] — PPO/DPO/GRPO 具体实现
- [[deepseek]] — DeepSeek-R1 的 RL 训练流程
- [[grpo-global]] — 辉少对 GRPO 的改进
- [[transformer]] — 基础架构（Teacher Forcing）
- [[external-blogs]] — Lilian Weng 的 Reward Hacking 文章

## Counter-arguments & Data Gaps
- RLHF 的奖励模型本身可能有偏差
- Goodhart's Law：过度优化奖励导致退化
- SFT + RLHF 是否是最佳组合？Direct Preference Optimization 提供了替代路径

## Sources
- [有 SFT 为什么还需要 RLHF](../sources/sft-vs-rlhf.md) — 面试笔记 Day3
