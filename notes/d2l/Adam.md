---
title: Adam
chapter: 第11章 优化算法
section: 11.10
prev: "[[AdaGrad与RMSProp]]"
next: "[[学习率调度器]]"
up: "[[第11章-优化算法]]"
source: https://zh.d2l.ai/chapter_optimization/adam.html
tags:
  - d2l
  - pytorch
  - fam/优化
---
# 11.10 Adam

## 一句话总结

**大白话**：Adam = 动量（记住「往哪走」的惯性）+ RMSProp（记住「每个参数该迈多大」的自适应步长）+ 一个开局校正。好处是几乎不用怎么调就能用，是最常用的优化器。它的改良版 AdamW 把[[权重衰退]]单独拎出来做，正则更干净，是 Transformer 训练标配。

**严谨说法**：Adam = 动量（一阶矩 EWMA）+ RMSProp（二阶矩 EWMA）+ 偏差修正，对[[学习率]]较鲁棒、几乎开箱即用，是深度学习最常用的优化器；AdamW 将权重衰退正确解耦。

## 本节解决的问题

- 如何同时利用动量与自适应学习率？
- 为什么需要偏差修正？
- AdamW 与 Adam 的区别？

## 核心概念 / 公式 / API

$$\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t \quad(\text{一阶矩})$$

$$\mathbf{v}_t=\beta_2\mathbf{v}_{t-1}+(1-\beta_2)\mathbf{g}_t^2 \quad(\text{二阶矩})$$

$$\hat{\mathbf{m}}_t=\frac{\mathbf{m}_t}{1-\beta_1^t},\quad \hat{\mathbf{v}}_t=\frac{\mathbf{v}_t}{1-\beta_2^t}\quad(\text{偏差修正})$$

$$\mathbf{w}_t=\mathbf{w}_{t-1}-\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}\hat{\mathbf{m}}_t$$

- 默认 $\beta_1=0.9,\ \beta_2=0.999$。
- 偏差修正纠正初期 m、v 被零初始化带来的偏小。
- **AdamW**：把权重衰退从梯度里解耦，正则更正确（Transformer 训练标配）。

| API | 说明 |
| --- | --- |
| `torch.optim.Adam(params, lr=1e-3)` | 标准 Adam |
| `torch.optim.AdamW(params, lr, weight_decay)` | 解耦权重衰退 |

## 最小可运行代码

```python
import torch

opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
```

## 易错点

- Adam 的 `weight_decay` 实为 L2 加到梯度，正则不纯；要正确权重衰退用 AdamW。
- 初期不做偏差修正会导致更新过小。
- Adam 收敛快但有时泛化略逊精调的 SGD+动量。

## 和前后章节 / 真实项目的连接

- 融合 [[动量法]] 与 [[AdaGrad与RMSProp]]。
- 与学习率调度配合见 [[学习率调度器]]；权重衰退见 [[权重衰退]]。

## 复习卡片

- Q: Adam 结合了哪两种思想？
  A: 动量（一阶矩）与 RMSProp（二阶矩）。
- Q: 偏差修正解决什么？
  A: 矩估计零初始化导致的初期偏小。
- Q: AdamW 相对 Adam 的改进？
  A: 把权重衰退与梯度更新解耦，正则更正确。

## 术语

- [[学习率]]、[[梯度]]、[[正则化]]

## 标签

#d2l #pytorch #optimization #ch11
