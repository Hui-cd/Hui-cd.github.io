# LLM Questions 2025

> 2025 年最新大模型面试题汇总，涵盖 RAG、Transformer、训练、微调、深度学习等

## Overview

辉少整理的 2025 年大模型面试题，覆盖了大语言模型技术的完整栈。从 Transformer 基础原理到 RLHF 训练，从 RAG 架构到多模态应用，是一份系统化的面试准备资料。

## Key Facts / Claims

### 一、RAG (检索增强生成)
- 内容缺失、排名问题、上下文整合限制
- RAG-Fusion 技术：多查询生成 + 重新排序
- 索引优化、数据优化策略
- 未来发展方向

### 二、Transformer 专题

**基础原理 (21 题)**
- 多头注意力：多子空间表征集成
- Q/K 不同权重矩阵：非对称性建模
- 点积 vs 加法注意力：硬件效率
- Scaled Attention：梯度稳定（除以 $\sqrt{d_k}$）
- Padding Mask：逻辑阻断
- 多头降维：计算量守恒
- Encoder/Decoder 结构
- 位置编码：$\sqrt{d_{model}}$ 缩放、正弦编码
- 残差结构：梯度高速公路
- LayerNorm vs BatchNorm：序列长度无关性
- FFN：Key-Value 记忆网络
- Cross-Attention：Encoder-Decoder 交互
- 并行化：Encoder 完全并行，Decoder 训练可并行
- Tokenization：BPE、WordPiece、BBPE、SentencePiece
- 学习率：Warmup + Inverse Square Root Decay

**训练与优化 (19 题)**
- QKV 来源：随机初始化的可学习参数
- FFN 训练内容：模式匹配 + 内容输出
- 复杂度分析：Embedding $O(1)$, Attention $O(n^2d)$, FFN $O(nd^2)$
- 位置编码的相对位置：线性变换属性
- LayerNorm 假设：层内特征同分布
- 降低 Embedding 参数：ALBERT 分解、Input-Output Tying
- 降低 FFN 参数：Weight Sharing、MoE
- 深度过深：梯度消失、秩崩溃、过度平滑

**应用与创新 (29 题)**
- Zero-shot Learning：NLI 转换、Prompting、CLIP 对齐
- Embedding 相似度比较：CKA、Procrustes Analysis
- 知识蒸馏：LSTM 模仿 BERT
- BERT 泛化限制：位置外推性、领域漂移、伪相关性
- GPT 缺陷：单向注意力，无法利用下文
- MLM 缺陷：预训练-微调差异、独立性假设
- 解决方案：ELECTRA、SpanBERT、XLNet

### 三、大模型 (LLMs) 基础与进阶
- 主流开源模型：LLaMA、Qwen、ChatGLM、DeepSeek
- Decoder-only 原因：生成能力、效率、Scaling Laws
- 涌现能力：规模效应
- 复读机问题：Unlikelihood Training、Repetition Penalty、Contrastive Search
- 长文本处理：位置编码外推、稀疏注意力

### 四、大模型微调
- 全参数微调显存估算
- SFT 数据构建、Continue Pretrain
- 灾难性遗忘缓解
- Chat vs Base 选择
- 词表扩增必要性
- Loss 突刺原因和解决
- Batch Size 设置

### 五、深度学习基础
- LN vs BN 原理和区别
- 交叉熵推导、代码手写
- Sigmoid、Softmax、ReLU
- 多头注意力手写
- Adam 优化器原理
- AUC 计算
- KL 散度
- 梯度消失/爆炸缓解
- L1/L2 正则区别
- Dropout 原理

### 六、多模态 / NLP 算法
- DPO 算法原理
- GPT vs BERT 结构和参数量
- Flash Attention 原理
- BERT 预训练任务
- FP16 量化策略
- QFormer 原理
- 位置编码：RoPE、ALiBi
- CLIP 原理、InfoNCE Loss
- BLIP2 架构
- LLaVA 和 LLaMA 区别
- 大模型幻觉
- 混合精度训练
- RMSNorm 手写
- DeepSpeed 原理
- PEFT 微调
- RAG 流程

### 七、GQA (Grouped-Query Attention) 深度分析
- **问题**：长文本推理时 KV Cache 的 Memory Bound
- **方案**：Query Head 分组共享 KV Head
- **数学**：$H_q=32, H_{kv}=8, G=4$，通过 Broadcasting 复制 KV
- **效果**：KV Cache 读取降低 4 倍，吞吐量提升近 4 倍
- **几何解释**：低秩约束，Query 间语义冗余使信息损失极小
- **工程**：PagedAttention 中减少 Page Table 开销，FlashAttention 中 SRAM 复用

## Related
- [[transformer]] — 基础架构详解
- [[flash-attention]] — 注意力优化
- [[llm-rl-algorithms]] — PPO/DPO/GRPO
- [[deepseek]] — 开源模型系列
- [[qwen-series]] — 通义千问系列
- [[deepspeed]] — 训练优化
- [[quantization]] — 模型量化
- [[interview-notes-comprehensive]] — 面试笔记综合索引

## Sources
- [LLM Questions 2025](../sources/llm-questions-2025.md) — /mnt/c/Users/gyh14/WorkSpace/Personal-Notes/Notes/LLM Questions.md
