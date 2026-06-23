---
title: AlexNet-VGG-NiN
chapter: 第7章 现代CNN
section: 7.1-7.3
prev: "[[LeNet]]"
next: "[[GoogLeNet]]"
up: "[[第7章-现代CNN]]"
source: https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html
tags:
  - d2l
  - pytorch
  - fam/CNN
---
# 7.1-7.3 AlexNet / VGG / NiN

## 一句话总结

**大白话**：这三张网就是 CNN 的「变强三连」。AlexNet：把 LeNet 加深加宽、换上 ReLU 和 [[暂退法Dropout|Dropout]]，一举引爆深度学习；VGG：别东拼西凑了，用「3×3 小[[卷积]]块」搭积木一样规整地堆；NiN：用 1×1 卷积 + 全局平均池化干掉占参数大头的全连接层。趋势就是越来越「模块化」。

**严谨说法**：从 AlexNet（更深 + ReLU + dropout + 数据增广，引爆深度学习）到 VGG（用「重复的小卷积块」标准化设计），再到 NiN（用 1×1 卷积 + 全局平均池化替代全连接），CNN 设计走向模块化。

## 本节解决的问题

- AlexNet 相比 LeNet 关键改进是什么？
- VGG 的「块」思想为何重要？
- NiN 如何去掉占参数大头的全连接层？

## 核心概念 / 公式 / API

| 网络 | 关键创新 |
| --- | --- |
| AlexNet | 更深更宽、ReLU、Dropout、最大池化、数据增广、GPU 训练 |
| VGG | 用多个 3×3 卷积堆叠的**重复块**，结构规整、易加深 |
| NiN | NiN 块 = 卷积 + 两个 1×1 卷积；末端**全局平均池化**代替全连接 |

- VGG 块：若干 `Conv3x3+ReLU` 后接一个 `MaxPool2`。
- NiN 用全局平均池化输出类别数通道 → 参数大减、抗过拟合。

## 最小可运行代码

```python
import torch
from torch import nn

def vgg_block(num_convs, in_c, out_c):
    layers = []
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU()]
        in_c = out_c
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

def nin_block(in_c, out_c, k, s, p):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p), nn.ReLU(),
        nn.Conv2d(out_c, out_c, 1), nn.ReLU(),   # 1x1 卷积
        nn.Conv2d(out_c, out_c, 1), nn.ReLU(),
    )
```

## 易错点

- VGG 全连接层参数极多，显存吃紧；可换全局平均池化。
- 直接堆很深的网络而不加 BN/残差，训练困难（见后续）。
- 输入分辨率与首层卷积/池化要匹配（AlexNet 用 224×224）。

## 和前后章节 / 真实项目的连接

- 1×1 卷积概念见 [[卷积填充与通道]]。
- 并行分支思想延伸到 [[GoogLeNet]]；加深的关键解法见 [[ResNet与DenseNet]]。

## 复习卡片

- Q: AlexNet 的标志性改进？
  A: 更深、ReLU、Dropout、数据增广、GPU 训练。
- Q: VGG 的核心设计思想？
  A: 用重复的 3×3 卷积块标准化堆叠加深网络。
- Q: NiN 如何减少参数？
  A: 用 1×1 卷积 + 全局平均池化替代全连接。

## 术语

- [[卷积]]、[[正则化]]、[[激活函数]]

## 标签

#d2l #pytorch #cnn #modern-cnn #ch07
