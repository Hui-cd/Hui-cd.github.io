# LLM RL Algorithms

> PPO、DPO、GRPO 的对比与核心机制，来自辉少的面试准备笔记

## Overview

大语言模型的对齐阶段主要使用三种强化学习算法：PPO（需要 Critic）、DPO（跳过 RL 直接偏好优化）、GRPO（用组内采样替代 Critic）。辉少的笔记详细对比了它们的优化目标、KL 散度处理方式、以及工程实践中的取舍。

## Key Facts / Claims

### PPO (Proximal Policy Optimization)
- **架构**：Actor-Critic，需要 4 个模型（Actor、Critic、Reference、Reward Model）
- **优化目标**：最大化预期奖励 $\mathbb{E}[r(x,y) - \beta \log \frac{\pi_\theta}{\pi_{ref}}]$
- **优势计算**：GAE（Generalized Advantage Estimation），$\hat{A}_t = \sum (\gamma\lambda)^k \delta_{t+k}$
- **KL 处理**：融入 Reward，参与 GAE 计算
- **核心 Loss**：$L^{CLIP} + c_1 L^{VF} + c_2 L^S$

### DPO (Direct Preference Optimization)
- **架构**：只需 2 个模型（Policy + Reference）
- **优化目标**：最大化偏好似然差（good vs bad 的相对概率）
- **核心洞察**：利用最优策略与奖励函数的解析解关系，将 RL 问题转化为二元分类
- **Loss**：$-\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})]$
- **KL 处理**：隐式包含在 Loss 推导公式中

### GRPO (Group Relative Policy Optimization)
- **架构**：只需 Actor + Reference，无 Critic
- **优化目标**：最大化组内相对优势
- **优势计算**：$A_i = \frac{r_i - \text{mean}(r_{group})}{\text{std}(r_{group})}$
- **KL 处理**：**显式作为独立 Loss 项**，独立于 Advantage 计算
  - 原因 1：防止组内标准化消解 KL 的物理尺度
  - 原因 2：弥补 Critic 缺失后的稳定性

### 三者对比

| 特性 | PPO | DPO | GRPO |
|------|-----|-----|------|
| 模型数量 | 4 | 2 | 2 |
| 显存占用 | 高 | 低 | 低 |
| 优化驱动力 | Critic 预测 | 数据对比 | 组内竞争 |
| KL 处理方式 | 融入 Reward | 隐式 | 显式独立 |
| 适用场景 | 通用 RL | 偏好对齐 | LLM 推理训练 |

## Related
- [[grpo-global]] — 辉少的 GRPO-Global 项目改进
- [[deepseek]] — DeepSeek-R1 使用 GRPO 训练
- [[sft-vs-rlhf]] — SFT 与 RLHF 的关系
- [[machine-learning-basics]] — 基础优化理论
- [[external-blogs]] — Lilian Weng 的 Reward Hacking 文章

## Counter-arguments & Data Gaps
- GRPO 组内全同奖励时的梯度消失问题（辉少的 GRPO-Global 项目已改进）
- DPO 在非限定性生成任务中可能不如 PPO
- 三种算法在长文本生成中的对比实验不足

## Sources
- [PPO](../sources/PPO.md) — 面试笔记 Day2
- [DPO vs PPO vs GRPO](../sources/dpo-ppo-grpo.md) — 面试笔记 Day2
- [GRPO vs PPO 核心区别](../sources/grpo-ppo-comparison.md) — 面试笔记 Day2
