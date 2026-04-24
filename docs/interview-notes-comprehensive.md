# Interview Notes Comprehensive

> 辉少 LLM Interview Day1-Day14 的综合索引

## Overview

辉少的面试笔记覆盖了大语言模型的完整技术栈，从基础架构到训练优化，从量化部署到多模态应用。以下按主题分类整理。

## Day1: Transformer 基础
- Transformer 基本流程
- Transformer vs CNN/RNN
- 为什么训练不是自回归的（Teacher Forcing）
- Adaptive Softmax
- 自注意力 vs 注意力
- LLM 上下文存储

## Day2: 训练与优化
- PPO 详解
- DPO vs PPO vs GRPO
- GRPO vs PPO 核心区别
- LoRA 与全量微调
- MoE 路由坍缩
- AdamW vs Adam vs SGD
- DeepSeek MLA
- 为什么都是 Decoder-only

## Day3: 数据与训练
- SFT 为什么还需要 RLHF
- DeepSpeed ZeRO 三阶段
- Prefix Tuning vs P-Tuning
- Prompt Tuning vs Instruction Tuning
- 多模态预训练原理
- Qwen-VL 详解

## Day4: 注意力优化
- Flash Attention 思想
- PagedAttention 思想

## Day5: 量化与多模态
- AWQ 量化详解
- Weight-only 量化
- BLIP 详解
- CLIP 详解

## Day6: 生成模型
- Flow Matching 详解
- FLUX 详解
- Stable Diffusion 3 架构
- CogVideoX 模型结构
- BLIP2
- DPO 训练中 reward 下降问题
- PPO 超参数调优

## Day7-Day8: 进阶主题
- DDPM
- Score-based Models
- Transformer 核心原理
- 大模型训练优化策略
- Tokenizer 工作机制
- LoRA/QLoRA/Adapter 微调方法

## Day10-Day11: 多模态与训练
- Cross-Attention Adapter
- 多模态 LLM 预训练
- 稀疏 Attention 与视频 Attention
- ChatGPT 训练步骤
- LLM 损失函数
- QLoRA 详解
- 位置编码（RoPE）

## Day12-Day14: RL 与工程
- GRPO 熵坍缩、长文本优化、梯度消失
- RL 奖励函数 vs 调参
- MoE 工程实现
- 为什么 LLM 后训练偏向 Policy-Based RL
- RLHF 流程
- 奖励模型训练
- Cold Start 详解
- GRPO 详解
- SFT Cross Entropy 推导
- Transformer FFN 必要性
- 微调全流程（预训练→SFT→Cold Start→RL）
- GRPO 生成长回答的原理

## DeepSeek 系列
- DeepSeek V1
- DeepSeek V2
- DeepSeek MoE

## Qwen 系列
- Qwen1.0
- Qwen1.5
- Qwen2.0（Dense + MoE）
- Qwen3（MoE + Thinking Mode）
- Qwen-VL / Qwen-VL2
- Gated Attention

## 实战项目
- MCP Agent（车载语音助手）
- Emotion Model
- GRPO-Global
- MRI-PET
- Next-Next Token
- Tesla 问答系统
- 澳洲电网数据生成

## Sources
- [LLM Interview 笔记](../sources/llm-interview/) — 104 个 markdown 文件
