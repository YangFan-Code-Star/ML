---
title: ResNet与DenseNet
chapter: 第7章 现代CNN
section: 7.6-7.7
prev: "[[批量规范化]]"
next: "[[序列模型]]"
up: "[[第7章-现代CNN]]"
source: https://zh.d2l.ai/chapter_convolutional-modern/resnet.html
tags:
  - d2l
  - pytorch
  - fam/CNN
---
# 7.6-7.7 残差网络 ResNet 与稠密连接 DenseNet

## 一句话总结

**大白话**：网络太深时，反而越训越差（退化）。ResNet 的妙招是给每个块加一条「直通车」——让它去学「在输入基础上要改动多少（残差）」而不是从零学整个映射，这样实在学不到东西时退回成「原样传过去」最省事，[[梯度]]也能顺着直通车无损回传。DenseNet 更激进：把前面每一层的输出全拼起来给后面用，特征复用到极致。

**严谨说法**：ResNet 用**残差连接** $f(x)=x+g(x)$ 让网络更容易学习恒等映射，从根本上解决「越深越难训」的退化问题；DenseNet 进一步把前面所有层在通道维拼接，最大化特征复用。

## 本节解决的问题

- 为什么加深网络反而性能下降（退化）？
- 残差连接如何缓解梯度消失/退化？
- DenseNet 与 ResNet 的连接方式差异（相加 vs 拼接）？

## 核心概念 / 公式 / API

### 残差块

$$y=\mathrm{ReLU}\big(x+\mathcal{F}(x)\big)$$

- 学习「残差」 $\mathcal{F}(x)=y-x$ 比直接学 $y$ 更易优化；恒等通路让梯度无衰减地回传。
- 通道/尺寸变化时旁路用 1×1 卷积调整。

### DenseNet
- 稠密块：每层输入是之前所有层输出的**通道拼接**；过渡层用 1×1 卷积 + 池化压缩通道与尺寸。

| 网络 | 跨层连接 |
| --- | --- |
| ResNet | 相加 `x + f(x)` |
| DenseNet | 拼接 `cat([x, f(x)])` |

## 最小可运行代码

```python
import torch
from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_c, out_c, use_1x1=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn1, self.bn2 = nn.BatchNorm2d(out_c), nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(in_c, out_c, 1, stride=stride) if use_1x1 else None
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3: X = self.conv3(X)     # 旁路对齐维度
        return F.relu(Y + X)                 # 残差相加
```

## 易错点

- 残差相加要求主路与旁路形状一致，变维处务必加 1×1 卷积。
- 误以为残差是简单加深；关键在恒等通路改善梯度流。
- DenseNet 通道随拼接快速增长，需过渡层压缩防显存爆炸。

## 和前后章节 / 真实项目的连接

- 依赖 BN [[批量规范化]]；残差思想被 Transformer 沿用（残差 + LayerNorm）→ [[Transformer]]。
- 真实项目：ResNet/类似骨干是检测/分割/DINOv3 的主干特征提取器。

- 残差连接被 [[Transformer]] 沿用：每个子层后做 Add & Norm。

## 复习卡片

- Q: 残差块学习的是什么？
  A: 残差 $f(x)=y-x$，并保留恒等通路。
- Q: ResNet 解决了什么问题？
  A: 深层网络的退化（越深训练误差反升）。
- Q: ResNet 与 DenseNet 跨层连接的区别？
  A: ResNet 相加，DenseNet 通道拼接。

## 术语

- [[残差连接]]、[[梯度消失与爆炸]]、[[卷积]]

## 标签

#d2l #pytorch #cnn #modern-cnn #ch07
