---
title: 长短期记忆网络LSTM
chapter: 第9章 现代RNN
section: 9.2
prev: "[[门控循环单元GRU]]"
next: "[[深度与双向RNN]]"
up: "[[第9章-现代RNN]]"
source: https://zh.d2l.ai/chapter_recurrent-modern/lstm.html
tags:
  - d2l
  - pytorch
  - fam/RNN
---
# 9.2 长短期记忆网络 (LSTM)

## 一句话总结

**大白话**：LSTM 比 GRU 多准备了一条专门的「记忆传送带」（记忆细胞 $C_t$），还配三个[[门控机制|阀门]]：遗忘门决定传送带上旧东西扔不扔、输入门决定新东西放不放、输出门决定从传送带上拿多少出来用。关键在于这条传送带是「加法更新」的，[[梯度]]能几乎不衰减地一路传回去，所以能记很长。

**严谨说法**：LSTM 引入独立的**记忆细胞** $C_t$ 和三个门（输入/遗忘/输出），通过门控的加法式更新让梯度沿细胞状态近乎无衰减地流动，是最经典的长程依赖序列模型。

## 本节解决的问题

- 记忆细胞与隐状态的分工？
- 输入门、遗忘门、输出门分别控制什么？
- LSTM 为何比普通 RNN 更能记长程？

## 核心概念 / 公式 / API

$$\mathbf{I}_t=\sigma(\cdots)\ \text{输入门},\quad \mathbf{F}_t=\sigma(\cdots)\ \text{遗忘门},\quad \mathbf{O}_t=\sigma(\cdots)\ \text{输出门}$$

$$\tilde{\mathbf{C}}_t=\tanh(\cdots)\ \text{候选记忆}$$

$$\mathbf{C}_t=\mathbf{F}_t\odot\mathbf{C}_{t-1}+\mathbf{I}_t\odot\tilde{\mathbf{C}}_t$$

$$\mathbf{H}_t=\mathbf{O}_t\odot\tanh(\mathbf{C}_t)$$

- 遗忘门保留旧记忆、输入门写入新信息、输出门决定暴露多少记忆给隐状态。
- 细胞状态 $C_t$ 的**加法式更新**是梯度稳定的关键。

| API | 说明 |
| --- | --- |
| `nn.LSTM(input_size, hidden_size)` | 状态是 `(h, c)` 元组 |

## 输入输出形状

- 状态为元组 `(h, c)`，各 `(num_layers, B, hidden)`。

## 最小可运行代码

```python
import torch
from torch import nn

lstm = nn.LSTM(28, 256)
X = torch.randn(35, 32, 28)
h = torch.zeros(1, 32, 256); c = torch.zeros(1, 32, 256)
Y, (h, c) = lstm(X, (h, c))
print(Y.shape, h.shape, c.shape)     # (35,32,256) (1,32,256) (1,32,256)
```

## 易错点

- LSTM 状态是 `(h, c)` 元组，截断/初始化时两者都要处理。
- 把记忆细胞 C 与隐状态 H 混为一谈：H 才是对外输出。
- 参数量约为同规模 GRU 的 4/3，注意算力。

## 和前后章节 / 真实项目的连接

- 与 [[门控循环单元GRU]] 同为门控方案，解决 [[通过时间反向传播BPTT]] 的梯度问题。
- 作 seq2seq 编码器/解码器骨干 → [[机器翻译与seq2seq]]。

## 复习卡片

- Q: LSTM 的三个门？
  A: 输入门、遗忘门、输出门。
- Q: 梯度稳定的关键结构？
  A: 记忆细胞 $C_t$ 的加法式更新。
- Q: LSTM 对外输出的是哪个量？
  A: 隐状态 $H_t=O_t\odot\tanh(C_t)$。

## 术语

- [[门控机制]]、[[隐状态]]、[[梯度消失与爆炸]]

## 标签

#d2l #pytorch #rnn #modern-rnn #ch09
