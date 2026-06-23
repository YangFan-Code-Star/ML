---
title: GoogLeNet
chapter: 第7章 现代CNN
section: 7.4
prev: "[[AlexNet-VGG-NiN]]"
next: "[[批量规范化]]"
up: "[[第7章-现代CNN]]"
source: https://zh.d2l.ai/chapter_convolutional-modern/googlenet.html
tags:
  - d2l
  - pytorch
  - fam/CNN
---
# 7.4 含并行连结的网络 (GoogLeNet / Inception)

## 一句话总结

**大白话**：与其纠结这一层该用大窗口还是小窗口，不如「全都要」——Inception 块同时用 1×1、3×3、5×5 几条路并行看图，再把结果在通道方向拼起来，让网络自己挑哪种尺度有用。为了不让计算量爆炸，先用 1×1 [[卷积]]把通道压一压再算。

**严谨说法**：GoogLeNet 的 Inception 块用**多条不同尺度的并行路径**（1×1、3×3、5×5 卷积与池化）并在通道维拼接，让网络自动组合多尺度特征，并用 1×1 卷积降维控制计算量。

## 本节解决的问题

- 如何在一层同时捕获多种尺度的特征？
- 如何避免多分支带来的计算爆炸？
- 通道维拼接需要满足什么条件？

## 核心概念 / 公式 / API

### Inception 块四条并行路径
1. 1×1 卷积
2. 1×1 卷积 → 3×3 卷积
3. 1×1 卷积 → 5×5 卷积
4. 3×3 最大池化 → 1×1 卷积

- 四路输出在**通道维 (dim=1) 拼接**，故各路空间尺寸必须相同（用 padding 对齐）。
- 1×1 卷积先降维，显著减少 3×3/5×5 的计算成本。

## 输入输出形状

- 各分支输出 `(B, c_i, H, W)`（H,W 一致），拼接后 `(B, Σc_i, H, W)`。

## 最小可运行代码

```python
import torch
from torch import nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super().__init__()
        self.p1 = nn.Conv2d(in_c, c1, 1)
        self.p2_1 = nn.Conv2d(in_c, c2[0], 1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], 3, padding=1)
        self.p3_1 = nn.Conv2d(in_c, c3[0], 1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], 5, padding=2)
        self.p4_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, 1)
    def forward(self, x):
        p1 = F.relu(self.p1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat([p1, p2, p3, p4], dim=1)   # 通道维拼接
```

## 易错点

- 各分支 padding 不一致导致空间尺寸不同，无法在通道维拼接。
- 忘记 1×1 降维 → 5×5 分支计算量巨大。
- 拼接维度写错（应为 `dim=1` 通道维）。

## 和前后章节 / 真实项目的连接

- 1×1 卷积降维基础见 [[卷积填充与通道]]。
- 多分支/拼接思想与 ResNet 的相加形成对比 → [[ResNet与DenseNet]]。

## 复习卡片

- Q: Inception 块的核心思想？
  A: 多尺度并行卷积+池化，通道维拼接融合。
- Q: 1×1 卷积在 Inception 中的作用？
  A: 降维以控制 3×3/5×5 的计算量。
- Q: 各分支拼接的前提条件？
  A: 空间尺寸 H×W 必须相同。

## 术语

- [[卷积]]、[[感受野]]

## 标签

#d2l #pytorch #cnn #modern-cnn #ch07
