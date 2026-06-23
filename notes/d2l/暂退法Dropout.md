---
title: 暂退法Dropout
chapter: 第4章 多层感知机
section: 4.6
prev: "[[权重衰退]]"
next: "[[数值稳定性与初始化]]"
up: "[[第4章-多层感知机]]"
source: https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html
tags:
  - d2l
  - pytorch
  - fam/基础
---
# 4.6 暂退法 (Dropout / 丢弃法)

## 一句话总结

**大白话**：训练时随机「关掉」一部分神经元（让它们临时罢工），逼着网络别死靠某几个神经元、要学会多条后路——这是一种[[正则化]]。为保持总量不变，存活的神经元要按比例放大。预测时全员上岗、不再随机关。

**严谨说法**：暂退法在训练时以概率 p 随机将隐藏单元置零并对存活单元放大 $1/(1-p)$，强迫网络不依赖特定神经元，从而[[正则化]]缓解[[过拟合]]；推理时关闭。

## 本节解决的问题

- 如何通过「注入噪声」提升泛化？
- 为什么要对存活单元做 $1/(1-p)$ 缩放？
- 训练与推理为什么行为不同？

## 核心概念 / 公式 / API

### Dropout 操作
- 训练时每个元素以概率 p 置零，存活元素除以 $1-p$（保持期望不变）：

$$h' = \begin{cases} 0 & \text{概率 } p \\ h/(1-p) & \text{概率 } 1-p \end{cases}$$

- $E[h']=h$，故推理时直接用原值、关闭 dropout。
- 直觉：相当于训练一个集成的「子网络」，减少神经元间共适应。

| API | 说明 |
| --- | --- |
| `nn.Dropout(p)` | 层；`net.train()` 时生效，`net.eval()` 时关闭 |

## 最小可运行代码

从零：
```python
import torch

def dropout_layer(X, p):
    assert 0 <= p <= 1
    if p == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > p).float()
    return mask * X / (1.0 - p)        # 缩放保持期望
```

简洁：
```python
from torch import nn

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 10),
)
# 训练: net.train(); 推理: net.eval()  -> 自动启用/关闭 dropout
```

## 易错点

- 推理时忘记 `net.eval()` → dropout 仍随机置零，预测不稳定。
- 把 dropout 放在输出层之后；通常用于隐藏层激活之后。
- 同时叠加过强的 weight decay + 高 dropout 可能欠拟合。

## 和前后章节 / 真实项目的连接

- 与 [[权重衰退]] 互补：dropout 注入噪声，weight decay 约束幅度。
- `train()/eval()` 模式切换也影响 [[批量规范化]]。

- 对比 L2 正则的另一条正则化路线：[[权重衰退]]。

## 复习卡片

- Q: 存活单元为何除以 $1-p$？
  A: 保持输出期望不变，推理时无需修改。
- Q: 推理时 dropout 行为？
  A: 关闭（`eval()`），直接前向。
- Q: dropout 的正则直觉？
  A: 随机失活减少神经元共适应，近似子网络集成。

## 术语

- [[正则化]]、[[过拟合]]

## 标签

#d2l #pytorch #mlp #regularization #ch04
