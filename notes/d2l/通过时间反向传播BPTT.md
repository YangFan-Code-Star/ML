---
title: 通过时间反向传播BPTT
chapter: 第8章 循环神经网络
section: 8.7
prev: "[[循环神经网络RNN]]"
next: "[[门控循环单元GRU]]"
up: "[[第8章-循环神经网络]]"
source: https://zh.d2l.ai/chapter_recurrent-neural-networks/bptt.html
tags:
  - d2l
  - pytorch
  - fam/RNN
---
# 8.7 通过时间反向传播 (BPTT)

## 一句话总结

**大白话**：把 RNN 按时间「拉直」，它其实就是一个很深的网络——有多少时间步就有多少层。反向传播时同一个权重 $W_{hh}$ 被连乘很多次：大于 1 越乘越炸（[[梯度消失与爆炸|爆炸]]），小于 1 越乘越没（消失）。这就是为什么要[[梯度裁剪]]、为什么后面要发明门控 RNN。

**严谨说法**：BPTT 是把 RNN 沿时间展开成深网络后做反向传播；由于隐到隐权重 $W_{hh}$ 反复连乘，梯度随时间跨度指数增长（爆炸）或衰减（消失），这正是门控 RNN 的动机。

## 本节解决的问题

- RNN 的梯度如何沿时间反传？
- 梯度爆炸/消失的数学根源？
- 截断 BPTT 为何能控制计算与稳定性？

## 核心概念 / 公式 / API

### 时间展开与连乘
- 损失对早期隐状态的梯度含 $\prod W_{hh}^\top$ 项；谱半径 >1 → 爆炸；<1 → 消失。

### 三种处理
| 策略 | 做法 |
| --- | --- |
| 完整 BPTT | 沿整条序列反传，计算/显存大、易不稳 |
| 截断 BPTT | 每隔若干步切断梯度（`detach` 隐状态），常用 |
| 随机截断 | 随机决定截断点（理论分析用） |

- 工程上配合**梯度裁剪**抑制爆炸。

## 最小可运行代码

```python
# 截断 BPTT：在时间步之间分离隐状态，阻断梯度无限回传
state = None
for X, y in train_iter:
    if state is None:
        state = init_state(batch_size)
    else:
        if isinstance(state, tuple):       # LSTM 的 (h, c)
            state = tuple(s.detach() for s in state)
        else:
            state = state.detach()         # 截断
    y_hat, state = net(X, state)
    # loss.backward(); grad_clipping(net, 1); opt.step()
```

## 易错点

- 顺序分区训练时不 `detach` 隐状态 → 计算图横跨所有 batch，显存暴涨。
- 只裁剪梯度不解决消失问题（裁剪只防爆炸）。
- 把梯度消失误判为学习率问题。

## 和前后章节 / 真实项目的连接

- 直接动机：门控单元缓解长程依赖的梯度消失 → [[门控循环单元GRU]]、[[长短期记忆网络LSTM]]。
- 自动微分基础见 [[微积分与自动微分]]。

- 其本质仍是链式法则与自动微分：[[微积分与自动微分]]。

## 复习卡片

- Q: 梯度爆炸/消失的根源？
  A: 隐到隐权重 $W_{hh}$ 沿时间连乘。
- Q: 截断 BPTT 怎么实现？
  A: 每隔若干步对隐状态 `detach`，切断梯度回传。
- Q: 梯度裁剪能解决消失吗？
  A: 不能，它只抑制爆炸。

## 术语

- [[梯度消失与爆炸]]、[[梯度裁剪]]、[[隐状态]]

## 标签

#d2l #pytorch #rnn #ch08
