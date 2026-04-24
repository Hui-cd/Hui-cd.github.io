# MCP Agent Project

> 车载语音助手项目：大小模型协同 + MCP 协议调度

## Overview

辉少的实战项目，构建车载语音助手系统。核心架构是「大小模型协同」：用 Bert-Tiny 做高并发过滤，Bert-Base 做意图召回，DeepSeek 大模型做复杂语义理解和参数抽取。通过 MCP 协议解耦决策与执行。

## Key Facts / Claims

### 三阶段架构

**第一阶段：离线训练**
- **拒识数据**：50万条，DeepSeek 生成对抗样本 + 线上日志
- **意图数据**：40万条，400+ 技能泛化为数十万条指令
- **模型 A**：Bert-Tiny（3层，312维）— 二分类拒识
- **模型 B**：Bert-Base（12层，768维）— 400+ 意图多分类

**第二阶段：在线推理**
1. **信号拦截**：Bert-Tiny 过滤 90%+ 无效流量
2. **上下文理解**：DeepSeek 微调版做 Query 改写
3. **意图粗排**：Bert-Base Top-K 截取 5 个候选意图
4. **精排决策**：DeepSeek 辨析 + Function Calling 参数抽取

**第三阶段：执行反馈**
- MCP 协议调度工具调用
- 高德 API、QQ 音乐 API、车载 CAN Bus
- 大模型生成自然语言回复 + TTS

### 核心技术壁垒
1. **大小模型协同**：Tiny 高并发（QPS 500+）+ Base 准确率 + 大模型复杂理解
2. **数据飞轮**：大模型生成数据反哺小模型
3. **MCP 标准化**：无限扩展新技能，无需重构核心代码

## Related
- [[deepseek]] — 项目中使用的大模型
- [[llm-rl-algorithms]] — 模型训练方法
- [[qwen-series]] — 另一大模型系列

## Sources
- [MCP Agent 项目](../sources/mcp-agent-project.md) — 实战笔记
