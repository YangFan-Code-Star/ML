---
title: softmax回归
chapter: 第3章 线性神经网络
section: 3.4-3.7
prev: "[[线性回归]]"
next: "[[图像分类数据集]]"
up: "[[第3章-线性神经网络]]"
source: https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html
tags:
  - d2l
  - pytorch
  - fam/基础
---
# 3.4 / 3.6-3.7 softmax 回归（从零 + 简洁）

## 一句话总结

**大白话**：线性模型每个类别先给一个「分数」（logit），softmax 把这些分数挤成「加起来等于 1 的概率」（分越高占比越大）。用[[交叉熵]]衡量「预测概率离正确答案有多远」。工程上别自己先 softmax 再取 log（会溢出），交给 `nn.CrossEntropyLoss` 一步到位。

**严谨说法**：softmax 回归把线性输出转成概率分布做多分类，用[[交叉熵]]损失度量；关键工程点是 softmax 与对数的**数值稳定**合并实现（`nn.CrossEntropyLoss` 内部直接吃 logits）。

## 本节解决的问题

- 如何把线性模型扩展到多分类输出概率？
- 交叉熵损失为什么适合分类？
- 为什么不能先 softmax 再取 log（数值溢出）？

## 核心概念 / 公式 / API

| 概念 | 公式 | 说明 |
| --- | --- | --- |
| softmax | $\hat{y}_j=\dfrac{\exp(o_j)}{\sum_k \exp(o_k)}$ | 输出归一化为概率 |
| 交叉熵 | $-\sum_j y_j\log\hat{y}_j$ | 真实分布与预测的距离 |
| 预测 | `argmax` | 取概率最大类别 |

- 数值稳定：先减最大值 $o_j-\max_k o_k$ 再做指数，避免上溢；log-softmax 与交叉熵合并避免下溢。
- `nn.CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`，**输入是未归一化的 logits**。

## 输入输出形状

- 输入 `X(n, d)` → logits `(n, q)`（q 为类别数）→ softmax `(n, q)`。
- 标签为整数索引 `(n,)`，交叉熵内部自动取对应分量。

## 最小可运行代码

仓库脚本：`d2l/code/3.6SoftMax回归从零开始实现.py`、`d2l/code/3.7SoftMax回归简洁实现.py`

从零（数值稳定 softmax）：
```python
import torch

def softmax(X):
    X = X - X.max(dim=1, keepdim=True).values  # 数值稳定
    e = torch.exp(X)
    return e / e.sum(dim=1, keepdim=True)

def cross_entropy(yhat, y):
    return -torch.log(yhat[range(len(yhat)), y])
```

简洁（直接吃 logits，省去手写 softmax）：
```python
from torch import nn

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
loss = nn.CrossEntropyLoss()          # 内部含 log-softmax
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

## 易错点

- 从零版自己 softmax 后又传给 `CrossEntropyLoss` → 等于做了两次，错误。
- 直接 `exp` 大 logits 会溢出 → 必须先减最大值。
- 标签应为整数类别索引，不是独热向量（`CrossEntropyLoss` 要求）。

## 和前后章节 / 真实项目的连接

- 数据集见 [[图像分类数据集]]；概率/极大似然基础见 [[概率与查阅文档]]。
- 分类头思想贯穿 CNN/Transformer 的输出层。

## 复习卡片

- Q: `nn.CrossEntropyLoss` 的输入应是什么？
  A: 未归一化的 logits 与整数标签，不要先做 softmax。
- Q: softmax 如何避免上溢？
  A: 每行先减去最大 logit 再取指数。
- Q: 交叉熵衡量什么？
  A: 预测概率分布与真实分布的差异（负对数似然）。

## 术语

- [[交叉熵]]、[[极大似然估计]]、[[条件概率]]

## 标签

#d2l #pytorch #linear-networks #ch03
