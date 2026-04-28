# Orthogonal Basis

> 向量空间的正交基与标准正交基，格拉姆-施密特正交化过程。

## Overview

正交基是线性代数中的核心概念。一组基向量如果两两正交（内积为零），则称为正交基；如果每个基向量还是单位长度，则称为标准正交基（Orthonormal Basis）。正交基大大简化了坐标计算和投影操作。

## Key Facts / Claims

### 定义

- **正交性**：$\mathbf{u} \cdot \mathbf{v} = 0$
- **单位长度**：$\|\mathbf{u}\| = 1$，即 $\mathbf{u} \cdot \mathbf{u} = 1$
- **标准正交基**：同时满足正交性和单位长度的基

### 格拉姆-施密特正交化

- 从任意线性无关向量组 $\{\mathbf{v}_1, \dots, \mathbf{v}_n\}$ 构造正交基 $\{\mathbf{u}_1, \dots, \mathbf{u}_n\}$
- 递推公式：
  $$
  \mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \frac{\mathbf{u}_j \cdot \mathbf{v}_k}{\mathbf{u}_j \cdot \mathbf{u}_j} \mathbf{u}_j
  $$
- 每一步将当前向量投影到已构建的正交子空间上，减去投影分量

### 标准例子

- 二维：$(1,0)$ 和 $(0,1)$
- 三维：$(1,0,0)$, $(0,1,0)$, $(0,0,1)$

### 在机器学习中的应用

- **PCA**：特征向量构成正交基，实现数据降维
- **QR 分解**：将矩阵分解为正交矩阵 Q 和上三角矩阵 R
- **傅里叶变换**：正弦/余弦函数构成函数空间的正交基

## Related

- [[machine-learning-basics]] — Fisher LDA 中的投影方向正交性
- [[gaussian-distribution]] — 多维高斯的协方差矩阵特征分解与正交基
- [[transformer]] — 注意力机制中的 Q/K/V 投影可理解为坐标变换

## Sources

- [正交基](https://hui-cd.github.io/正交基/) — 辉少的笔记原文
