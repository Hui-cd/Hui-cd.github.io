# External Blogs

> 权威 AI 研究者的博客和网站索引

## Overview

除了辉少自己的笔记，以下外部博客是 AI 领域的重要知识来源。这些博客由顶尖研究者和从业者维护，涵盖了大语言模型、扩散模型、强化学习等前沿方向。

## Lilian Weng

**URL**: https://lilianweng.github.io/

**背景**: OpenAI 前安全系统负责人，现 Anthropic。博客以深度技术解析著称。

**核心文章**:
- **"LLM Powered Autonomous Agents"** (2023) — Agent 架构基石文章
  - 涵盖 Planning、Memory、Tool use
  - ReAct、Reflexion、Chain-of-Thought
  - Multi-agent 系统
- **"Reward Hacking in Reinforcement Learning"** (2024-11) — Agent 安全与对齐
  - Specification gaming 和 reward hacking
  - 对 LLM agent 对齐的高度相关

**特点**:  posting 频率 2024 年后降低（转职原因），但每篇都是精品。

## 科学空间 (苏剑林)

**URL**: https://kexue.fm/

**背景**: 中文世界最深入的 Transformer/注意力机制分析者之一。

**核心贡献**:
- **RoPE (Rotary Position Embedding)** — 旋转位置编码
  - 被 LLaMA、PaLM 等大模型广泛采用
  - 公式: $f(q, m) = q e^{i m \theta}$
- **GAU (Gated Attention Unit)** — 融合注意力与 FFN
- **Transformer 升级之路系列** — 系统性改进研究

**核心洞察**:
> "注意力机制的核心不是'关注哪里'，而是**通过内积实现相似度度量后的加权平均**。"
> "Transformer 的成功在于**并行计算 + 全局交互 + 可扩展性**的三位一体。"

## Yang Song

**URL**: https://yang-song.net/blog/

**背景**: Score Matching 和扩散模型统一框架的提出者（Score SDE）。

**核心工作**:
- **Score-Based Generative Modeling through SDEs** (ICLR 2021)
  - 统一了 Denoising Score Matching、Langevin dynamics、DDPM
  - 核心洞察：**Diffusion models = Score matching + SDEs**
- **Denoising Diffusion Implicit Models (DDIM)** — 快速采样
- **Noise Conditional Score Networks (NCSN)** 系列

## Yi Su

**URL**: https://ysymyth.github.io/

**背景**: 强化学习与大模型结合的研究者。

**关注方向**:
- RLHF 与对齐
- Agent 学习与决策
- 大模型的推理能力

## 使用建议

**阅读优先级**:
1. **入门**: Lilian Weng 的 Agent 文章 + 科学空间的 Transformer 解读
2. **进阶**: Yang Song 的 Score Matching + 辉少的扩散模型笔记
3. **前沿**: 各博客的最新更新 + 结合辉少的面试笔记

**与辉少笔记的关系**:
- 辉少的扩散模型笔记（DDPM + Score Matching + SDE）直接对应 Yang Song 的工作
- 辉少的 Transformer 笔记与科学空间的分析互补
- 辉少的 RL 笔记（PPO/DPO/GRPO）可与 Lilian Weng 的 Reward Hacking 结合

## Sources
- [Lilian Weng](https://lilianweng.github.io/)
- [科学空间](https://kexue.fm/)
- [Yang Song](https://yang-song.net/blog/)
- [Yi Su](https://ysymyth.github.io/)
