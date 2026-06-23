---
title: 循环神经网络RNN
chapter: 第8章 循环神经网络
section: 8.4-8.6
prev: "[[文本预处理与语言模型]]"
next: "[[通过时间反向传播BPTT]]"
up: "[[第8章-循环神经网络]]"
source: https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html
tags:
  - d2l
  - pytorch
  - fam/RNN
---
# 8.4-8.6 循环神经网络 RNN（从零 + 简洁）

## 一句话总结

**大白话**：RNN 像一个一边读一边记笔记的人——每读一个词，就用「老笔记 + 新词」更新成「新笔记」（[[隐状态]]），这份笔记一路带着走。读写笔记用的是同一套规则（参数时间步共享）。好坏用[[困惑度]]衡量（越小说明它对下一个词越笃定），训练时要[[梯度裁剪]]防止梯度爆炸。

**严谨说法**：RNN 用一个随时间更新的**[[隐状态]]** $h_t$ 概括历史信息，参数在所有时间步共享；可做字符级语言模型，用[[困惑度]] (perplexity) 评估，并需[[梯度裁剪]]稳定训练。

## 本节解决的问题

- 隐状态如何在时间步间传递信息？
- RNN 参数为何在时间步共享？
- 如何用 RNN 生成文本、如何评估？

## 核心概念 / 公式 / API

### 隐状态更新

$$\mathbf{h}_t=\phi(\mathbf{W}_{xh}\mathbf{x}_t+\mathbf{W}_{hh}\mathbf{h}_{t-1}+\mathbf{b}_h),\quad \mathbf{o}_t=\mathbf{W}_{hq}\mathbf{h}_t+\mathbf{b}_q$$

- $\mathbf{W}_{hh}$ 携带「记忆」；所有时间步共享同一组权重。

### 困惑度 (perplexity)

$$\exp\Big(-\frac{1}{T}\sum_t \log P(x_t\mid x_{<t})\Big)$$

- 越小越好；等于「平均每步在多少个等可能选项中纠结」。

### 梯度裁剪
- 防梯度爆炸：$\mathbf{g}\leftarrow\min(1,\theta/\lVert\mathbf{g}\rVert)\mathbf{g}$。

| API | 说明 |
| --- | --- |
| `nn.RNN(input_size, hidden_size)` | 内置 RNN 层 |
| 输入 one-hot 或 `nn.Embedding` | token → 向量 |

## 输入输出形状

- `nn.RNN` 输入 `(T, B, input_size)`，输出 `(T, B, hidden)` 与最终隐状态 `(num_layers, B, hidden)`。

## 最小可运行代码

简洁实现：
```python
import torch
from torch import nn

vocab_size, hidden = 28, 256
rnn = nn.RNN(vocab_size, hidden)
X = torch.randn(35, 32, vocab_size)        # (T, B, V) one-hot
state = torch.zeros(1, 32, hidden)
Y, state = rnn(X, state)
print(Y.shape, state.shape)                # (35,32,256) (1,32,256)

def grad_clipping(net, theta):             # 梯度裁剪
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for p in params: p.grad[:] *= theta / norm
```

## 易错点

- 不裁剪梯度 → RNN 极易梯度爆炸训练发散。
- 时间步间传递隐状态时忘记 `detach`（顺序分区），导致计算图无限增长。
- 混淆 one-hot 与 embedding 输入维度。

## 和前后章节 / 真实项目的连接

- 训练原理（沿时间反向）见 [[通过时间反向传播BPTT]]。
- 梯度消失的根治方案：门控 RNN → [[门控循环单元GRU]]、[[长短期记忆网络LSTM]]。

## 复习卡片

- Q: RNN 的记忆体现在哪个权重？
  A: 隐到隐权重 $W_{hh}$。
- Q: 困惑度的直观含义？
  A: 平均每步在多少等可能选项中选择，越小越好。
- Q: 为什么 RNN 需要梯度裁剪？
  A: 时间步连乘易梯度爆炸。

## 术语

- [[隐状态]]、[[困惑度]]、[[梯度裁剪]]、[[梯度消失与爆炸]]

## 标签

#d2l #pytorch #rnn #ch08
