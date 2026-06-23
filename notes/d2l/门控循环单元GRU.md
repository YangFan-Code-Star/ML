---
title: 门控循环单元GRU
chapter: 第9章 现代RNN
section: 9.1
prev: "[[通过时间反向传播BPTT]]"
next: "[[长短期记忆网络LSTM]]"
up: "[[第9章-现代RNN]]"
source: https://zh.d2l.ai/chapter_recurrent-modern/gru.html
tags:
  - d2l
  - pytorch
  - fam/RNN
---
# 9.1 门控循环单元 (GRU)

## 一句话总结

**大白话**：普通 RNN 的笔记每步都被全部改写，旧信息很快丢光。GRU 给笔记加了两个「阀门」（[[门控机制|门]]）：更新门管「这次旧笔记保留多少」，重置门管「写新内容时要不要先忘掉旧的」。这样它能选择性地长期记住重要信息，缓解[[梯度消失与爆炸|梯度消失]]，而且比 LSTM 简单。

**严谨说法**：GRU 用**重置门**与**更新门**控制信息流：更新门决定保留多少旧状态、重置门决定遗忘多少历史，从而让网络学会长程依赖、缓解梯度消失，结构比 LSTM 更简洁。

## 本节解决的问题

- 如何让 RNN 选择性记忆/遗忘？
- 更新门、重置门各自的作用？
- GRU 相比普通 RNN 好在哪？

## 核心概念 / 公式 / API

$$\mathbf{R}_t=\sigma(\mathbf{X}_t\mathbf{W}_{xr}+\mathbf{H}_{t-1}\mathbf{W}_{hr}+\mathbf{b}_r) \quad\text{重置门}$$

$$\mathbf{Z}_t=\sigma(\mathbf{X}_t\mathbf{W}_{xz}+\mathbf{H}_{t-1}\mathbf{W}_{hz}+\mathbf{b}_z) \quad\text{更新门}$$

$$\tilde{\mathbf{H}}_t=\tanh(\mathbf{X}_t\mathbf{W}_{xh}+(\mathbf{R}_t\odot\mathbf{H}_{t-1})\mathbf{W}_{hh}+\mathbf{b}_h)$$

$$\mathbf{H}_t=\mathbf{Z}_t\odot\mathbf{H}_{t-1}+(1-\mathbf{Z}_t)\odot\tilde{\mathbf{H}}_t$$

- 更新门 $Z_t\to 1$：几乎完全保留旧状态（跳过当前输入），实现长记忆。
- 重置门 $R_t\to 0$：忽略历史，相当于重新开始。

| API | 说明 |
| --- | --- |
| `nn.GRU(input_size, hidden_size)` | 内置 GRU |

## 最小可运行代码

```python
import torch
from torch import nn

gru = nn.GRU(28, 256)
X = torch.randn(35, 32, 28)
state = torch.zeros(1, 32, 256)
Y, state = gru(X, state)
print(Y.shape)        # (35, 32, 256)
```

## 易错点

- 混淆两个门的作用：更新门管「记多少旧」，重置门管「忘多少旧」。
- GRU 只有单一隐状态 H（无独立记忆细胞），与 LSTM 区别。

## 和前后章节 / 真实项目的连接

- 解决 [[通过时间反向传播BPTT]] 的梯度消失/长程依赖问题。
- 与更复杂的 [[长短期记忆网络LSTM]] 对比；二者常作为 seq2seq 编码器 → [[机器翻译与seq2seq]]。

## 复习卡片

- Q: GRU 的两个门？
  A: 重置门（忘多少历史）与更新门（保留多少旧状态）。
- Q: 更新门趋近 1 意味着什么？
  A: 几乎完全保留旧隐状态，实现长程记忆。
- Q: GRU 与 LSTM 结构差异？
  A: GRU 无独立记忆细胞，门更少更简洁。

## 术语

- [[门控机制]]、[[隐状态]]、[[梯度消失与爆炸]]

## 标签

#d2l #pytorch #rnn #modern-rnn #ch09
