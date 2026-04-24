---
single: post
title:  "Transformer"
date:   2024-09-02 18:26:22 +1000
categories: 
    - deep-learning
author: Hui
---

## Transformer 结构

Transformer 是一个经典的encoder-decoder的模型

### Transformer的工作流程
首先第一步，获取输入句子的每一个单词的向量表示$X$, $X$是每个单词的word embedding和这个单词的位置相加得到的，通常一个词的embedding是在大量文本数据上训练Word2Vec或GloVe模型，最终将文本表示转换为向量的表示



Self-attention机制，有时称为intra-attention机制，是一种将单个序列的不同位置联系起来以计算序列表示的注意力机制

