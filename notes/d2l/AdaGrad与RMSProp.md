---
title: AdaGrad与RMSProp
chapter: 第11章 优化算法
section: 11.7-11.9
prev: "[[动量法]]"
next: "[[Adam]]"
up: "[[第11章-优化算法]]"
source: https://zh.d2l.ai/chapter_optimization/adagrad.html
tags:
  - d2l
  - pytorch
  - fam/优化
---
# 11.7-11.9 AdaGrad / RMSProp / Adadelta

## 一句话总结

**大白话**：与其所有参数用同一个步长，不如「谁动得多就让它步子小一点、谁动得少就步子大一点」（自适应[[学习率]]）。AdaGrad 把历史梯度平方一直累加当分母——结果越加越大、后期几乎不动了；RMSProp 改成「只记最近一段」（指数加权平均），避免过早停步；Adadelta 更进一步连全局学习率都省了。

**严谨说法**：这一族「自适应学习率」算法按每个参数的历史[[梯度]]幅度调整步长：AdaGrad 累积全部平方梯度（后期步长过小），RMSProp 改用指数加权平均解决衰减过快，Adadelta 进一步免去全局学习率。

## 本节解决的问题

- 为什么不同参数应有不同学习率？
- AdaGrad 的「学习率单调衰减」缺陷？
- RMSProp/Adadelta 如何改进？

## 核心概念 / 公式 / API

### AdaGrad

$$\mathbf{s}_t=\mathbf{s}_{t-1}+\mathbf{g}_t^2,\quad \mathbf{w}_t=\mathbf{w}_{t-1}-\frac{\eta}{\sqrt{\mathbf{s}_t+\epsilon}}\odot\mathbf{g}_t$$

- 稀疏/不常更新的特征获得更大步长；但 $s_t$ 单调增 → 后期步长趋零。

### RMSProp

$$\mathbf{s}_t=\gamma\mathbf{s}_{t-1}+(1-\gamma)\mathbf{g}_t^2$$

- 用 EWMA 代替累加，避免学习率过早衰减，适合非凸深度学习。

### Adadelta
- 用更新量的 EWMA 取代全局学习率 η，**无需手设学习率**。

| API | 说明 |
| --- | --- |
| `torch.optim.Adagrad` / `RMSprop` / `Adadelta` | 三者内置 |

## 最小可运行代码

```python
import torch

opt = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.9)
# alpha 即 gamma（EWMA 衰减系数）
```

## 易错点

- AdaGrad 在长训练中步长衰减到几乎不更新，深度学习首选 RMSProp/Adam。
- ε 防除零，不可省略。
- RMSProp 的 `alpha` 即衰减系数 γ，别和动量混淆。

## 和前后章节 / 真实项目的连接

- RMSProp + 动量 = Adam → [[Adam]]。
- EWMA 思想同 [[动量法]]。

## 复习卡片

- Q: AdaGrad 的主要缺陷？
  A: 平方梯度累加单调增，后期学习率趋零。
- Q: RMSProp 如何改进？
  A: 用指数加权平均替代累加，避免步长过早衰减。
- Q: Adadelta 的特点？
  A: 无需手动设置全局学习率。

## 术语

- [[学习率]]、[[梯度]]

## 标签

#d2l #pytorch #optimization #ch11
