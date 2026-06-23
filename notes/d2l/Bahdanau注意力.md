---
title: Bahdanau注意力
chapter: 第10章 注意力机制
section: 10.4
prev: "[[注意力评分函数]]"
next: "[[多头注意力]]"
up: "[[第10章-注意力机制]]"
source: https://zh.d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html
tags:
  - d2l
  - pytorch
  - fam/注意力
---
# 10.4 Bahdanau 注意力

## 一句话总结

**大白话**：原来的 seq2seq 把整句源文压成一个固定向量，长句根本塞不下（瓶颈）。Bahdanau 注意力让解码器每写一个词，都回头「重新看一遍原文、挑当前最相关的几个词重点看」——翻译到哪、注意力就对齐到源句哪，瓶颈打破。

**严谨说法**：Bahdanau 注意力把固定的上下文向量换成「随解码步动态变化的[[注意力机制|注意力]]汇聚」：解码每一步用当前隐状态作 query，对编码器所有时间步输出做注意力，从而对齐源句相关词，突破 seq2seq 的信息瓶颈。

## 本节解决的问题

- 普通 seq2seq 的固定上下文向量瓶颈是什么？
- 如何让解码器在每步关注源序列的不同部分？
- query/key/value 在翻译中分别是谁？

## 核心概念 / 公式 / API

- 瓶颈：编码器把整句压成单个向量 $\mathbf{c}$，长句信息损失严重。
- Bahdanau：解码第 t 步的上下文 $\mathbf{c}_t$ 由注意力动态生成：
  - **query** = 解码器上一步隐状态 $\mathbf{s}_{t-1}$
  - **key = value** = 编码器各时间步输出 $\mathbf{h}_1,\dots,\mathbf{h}_T$

$$\mathbf{c}_t=\sum_i \alpha(\mathbf{s}_{t-1},\mathbf{h}_i)\,\mathbf{h}_i$$

- 用加性注意力评分（q、k 来源不同）。

## 最小可运行代码

```python
import torch
from torch import nn

class AttnDecoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.attention = AdditiveAttention(hidden, hidden, hidden, 0.1)
        self.embedding = nn.Embedding(vocab, embed)
        self.rnn = nn.GRU(embed + hidden, hidden)
    def forward(self, X, state, enc_outputs):
        # 每个解码步: query=上一步隐状态, key=value=编码器输出
        query = state[-1].unsqueeze(1)
        context = self.attention(query, enc_outputs, enc_outputs)  # (B,1,h)
        x = torch.cat([self.embedding(X), context], dim=-1)
        return self.rnn(x.transpose(0, 1), state)
```

## 易错点

- 把 key 与 value 设成不同张量——这里二者都是编码器输出。
- 解码每步都要重新计算注意力，不能复用固定上下文。
- 注意力权重可视化能验证对齐是否合理（调试利器）。

## 和前后章节 / 真实项目的连接

- 解决 [[机器翻译与seq2seq]] 的固定上下文瓶颈。
- 把 RNN 完全替换为注意力即得 Transformer → [[Transformer]]。

## 复习卡片

- Q: Bahdanau 注意力解决什么瓶颈？
  A: seq2seq 固定上下文向量对长句信息损失大。
- Q: 翻译中 query/key/value 分别是谁？
  A: query=解码器隐状态，key=value=编码器各步输出。
- Q: 用哪种评分函数？
  A: 加性注意力（q、k 来源不同）。

## 术语

- [[注意力机制]]、[[查询键值]]、[[编码器-解码器]]

## 标签

#d2l #pytorch #attention #nlp #ch10
