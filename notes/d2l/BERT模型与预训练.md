---
title: BERT模型与预训练
chapter: 第14章 NLP预训练
section: 14.8-14.10
prev: "[[子词嵌入与相似性]]"
next: "[[情感分析]]"
up: "[[第14章-NLP预训练]]"
source: https://zh.d2l.ai/chapter_natural-language-processing-pretraining/bert.html
tags:
  - d2l
  - pytorch
  - fam/NLP
---
# 14.8-14.10 BERT 模型与预训练

## 一句话总结

**大白话**：word2vec 给每个词一个固定向量，可「bank」是河岸还是银行它分不清。BERT 用 [[Transformer]] 编码器双向读全句，让同一个词的表示随上下文变化。它的自学方式是「完形填空」（[[掩码语言模型|MLM]]：盖住 15% 的词让它猜）+「判断两句是否相邻」（NSP）。预训练一次，[[微调]]到各种下游任务。

**严谨说法**：BERT 是基于 Transformer 编码器的双向预训练语言模型，用**[[掩码语言模型]] (MLM)** 和**下一句预测 (NSP)** 两个[[自监督学习|自监督]]任务在大规模语料上预训练，得到上下文相关的词表示，再微调到各种下游任务。

## 本节解决的问题

- word2vec 的词向量为何「静态」（不随上下文变）？
- BERT 如何得到上下文相关表示？
- MLM 与 NSP 两个预训练任务怎么设计？

## 核心概念 / 公式 / API

### 输入表示
- 输入 = **词元嵌入 + 段嵌入 + 位置嵌入** 之和。
- 特殊符号：`<cls>`（句首，聚合句级表示）、`<sep>`（分隔句子对）。

### 两个预训练任务
| 任务 | 做法 |
| --- | --- |
| 掩码语言模型 (MLM) | 随机遮盖 15% 词元，让模型预测被遮词（实现双向） |
| 下一句预测 (NSP) | 判断句子 B 是否紧接句子 A |

- 编码器是双向自注意力（看得到左右全文），故表示**上下文相关**。
- 下游微调：在 `<cls>` 表示或词元表示上加一个任务头即可。

## 最小可运行代码

```python
import torch
from torch import nn

# BERT 编码器层（即 Transformer 编码器层）堆叠
layer = nn.TransformerEncoderLayer(d_model=768, nhead=12,
                                   dim_feedforward=3072, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=12)

# 输入嵌入 = 词元 + 段 + 位置
tok = nn.Embedding(30000, 768)
seg = nn.Embedding(2, 768)
pos = nn.Parameter(torch.randn(1, 512, 768))
```

## 易错点

- 把 BERT 当生成模型——它是编码器（理解），生成用 GPT 类解码器。
- 微调时忘记用预训练的分词器/词表与归一化。
- MLM 仅对被掩码位置算损失。

## 和前后章节 / 真实项目的连接

- 架构基于 Transformer 编码器 [[Transformer]]，双向思想呼应 [[深度与双向RNN]]。
- 下游微调应用 → [[微调BERT下游任务]]；预训练-微调范式同 [[微调]]。

- 预训练目标 (MLM) 源于语言模型思想：[[文本预处理与语言模型]]。
- 编码器由多头自注意力堆叠：[[多头注意力]] · [[Transformer]]。

## 复习卡片

- Q: BERT 基于 Transformer 的哪一部分？
  A: 编码器（双向自注意力）。
- Q: 两个预训练任务？
  A: 掩码语言模型 (MLM) 与下一句预测 (NSP)。
- Q: BERT 词表示与 word2vec 的本质区别？
  A: BERT 上下文相关（动态），word2vec 静态。

## 术语

- [[掩码语言模型]]、[[自监督学习]]、[[词嵌入]]、[[迁移学习]]

## 标签

#d2l #pytorch #nlp #bert #transformer #ch14
