---
title: 词嵌入word2vec
chapter: 第14章 NLP预训练
section: 14.1
prev: "[[风格迁移]]"
next: "[[近似训练与GloVe]]"
up: "[[第14章-NLP预训练]]"
source: https://zh.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html
tags:
  - d2l
  - pytorch
  - fam/NLP
---
# 14.1 词嵌入 word2vec

## 一句话总结

**大白话**：独热编码里每个词都是一根孤零零的「坐标轴」，彼此毫无关系。[[词嵌入]]改成给每个词一串小数（稠密向量），让意思相近的词在空间里离得近。word2vec 靠「猜上下文」自学这些向量：skip-gram 用中心词猜周围词，CBOW 反过来。

**严谨说法**：[[词嵌入]]把词映射为稠密低维向量，使语义相近的词在向量空间靠近；word2vec 用「上下文预测」自监督学习这些向量，含跳元 (skip-gram) 与连续词袋 (CBOW) 两种模型。

## 本节解决的问题

- 独热编码的缺陷（高维、无语义关系）？
- skip-gram 与 CBOW 的预测方向区别？
- 如何用上下文共现自监督学到词向量？

## 核心概念 / 公式 / API

| 模型 | 任务 |
| --- | --- |
| Skip-gram | 用中心词预测上下文词 |
| CBOW | 用上下文词预测中心词 |

- skip-gram 条件概率（softmax over 全词表）：

$$P(w_o\mid w_c)=\frac{\exp(\mathbf{u}_o^\top\mathbf{v}_c)}{\sum_{w}\exp(\mathbf{u}_w^\top\mathbf{v}_c)}$$

- 每个词有两套向量：作中心词 $\mathbf{v}$、作上下文 $\mathbf{u}$。
- 分母对整个词表求和 → 计算昂贵 → 需近似训练（见下篇）。

## 最小可运行代码

```python
import torch
from torch import nn

# 词嵌入层：把词索引映射为稠密向量
embed = nn.Embedding(num_embeddings=10000, embedding_dim=100)
center = torch.tensor([1, 5, 8])
print(embed(center).shape)        # (3, 100)
```

## 易错点

- 独热向量正交且高维，无法表达语义相似度。
- skip-gram 的 softmax 分母遍历全词表，朴素实现极慢。
- 中心词向量与上下文向量是两套，别混用。

## 和前后章节 / 真实项目的连接

- 加速训练：负采样/层序 softmax → [[近似训练与GloVe]]。
- 词向量作为下游 NLP 模型输入 → [[情感分析]]。

## 复习卡片

- Q: 词嵌入相对独热的优势？
  A: 低维稠密且能表达语义相似度。
- Q: skip-gram 与 CBOW 的区别？
  A: skip-gram 用中心词预上下文，CBOW 反之。
- Q: word2vec 训练的计算瓶颈？
  A: softmax 分母遍历整个词表。

## 术语

- [[词嵌入]]、[[词表]]、[[自监督学习]]

## 标签

#d2l #pytorch #nlp #embedding #ch14
