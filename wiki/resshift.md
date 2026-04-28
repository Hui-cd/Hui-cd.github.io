# ResShift

> 基于扩散模型的图像超分辨率方法，通过残差移位实现高效重建。

## Overview

ResShift 是一种将扩散模型应用于图像超分辨率（Super-Resolution）的方法。其核心思想是通过**残差移位**（Residual Shifting）策略，在低分辨率（LR）图像和高分辨率（HR）图像之间建立高效的映射关系，避免传统扩散模型在超分辨率任务中的高计算成本。

## Key Facts / Claims

### 核心思想

- 假设 LR 图像 $y_0$ 和 HR 图像 $x_0$ 具有相同的空间分辨率（LR 通过最近邻插值上采样）
- 学习从 $y_0$ 到 $x_0$ 的残差映射，而非直接从噪声生成 HR
- 残差移位：在扩散过程中逐步将 LR 特征移向 HR 特征

### 与标准扩散模型的区别

- **标准 DDPM**：从纯噪声 $\mathcal{N}(0,I)$ 开始，逐步去噪生成图像
- **ResShift**：以 LR 图像为条件，学习残差分布 $p(x_0 - y_0)$
- 优势：条件信息强，收敛更快，生成质量更高

### 技术细节

- 条件编码器：提取 LR 图像的多尺度特征
- 残差预测：噪声预测网络预测的是 HR 与 LR 之间的残差噪声
- 加速采样：由于条件强，可用更少的扩散步数（如 15 步 vs 1000 步）

## Related

- [[diffusion-model]] — 基础扩散模型原理
- [[flow-matching]] — 另一种生成模型范式，也可用于超分辨率
- [[unet]] — ResShift 的骨干网络通常为 U-Net 架构

## Sources

- [ResShift](https://hui-cd.github.io/ResShift/) — 辉少的笔记原文
