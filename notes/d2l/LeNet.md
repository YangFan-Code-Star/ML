---
title: LeNet
chapter: 第6章 卷积神经网络
section: 6.6
prev: "[[汇聚层]]"
next: "[[AlexNet-VGG-NiN]]"
up: "[[第6章-卷积神经网络]]"
source: https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html
tags:
  - d2l
  - pytorch
  - fam/CNN
---
# 6.6 卷积神经网络 (LeNet)

## 一句话总结

**大白话**：LeNet 是 CNN 的「鼻祖」，套路很简单：用「[[卷积]]→激活→池化」这套组合拳一层层把图缩小、把通道（特征种类）变多，最后摊平交给全连接层分类。它把第 6 章学的零件第一次拼成了能在 Fashion-MNIST 上跑通的完整模型。

**严谨说法**：LeNet 是最早的卷积网络范式：**卷积+激活+池化堆叠提取空间特征 → 展平 → 全连接分类**；它把第 6 章的卷积/池化组件组装成一个可在 Fashion-MNIST 上训练的完整模型。

## 本节解决的问题

- 如何把卷积、池化、全连接组装成完整分类网络？
- 卷积块如何逐步降空间、升通道？
- 在 GPU 上训练 CNN 的标准循环？

## 核心概念 / 公式 / API

### LeNet 结构
```
Conv(1->6,5x5) -> Sigmoid -> AvgPool2
Conv(6->16,5x5) -> Sigmoid -> AvgPool2
Flatten -> FC(16*5*5->120) -> FC(120->84) -> FC(84->10)
```
- 规律：空间尺寸逐层减小，通道数逐层增大，最后展平交给全连接分类。

## 输入输出形状

- 输入 `(B,1,28,28)`（书中 padding=2 → 32×32）→ … → 输出 `(B,10)`。

## 最小可运行代码

```python
import torch
from torch import nn

net = nn.Sequential(
    nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2, 2),
    nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10),
)

# GPU 训练循环骨架
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.9)
# for X, y in train_iter: X,y=X.to(device),y.to(device); ...
```

## 易错点

- 展平前的特征图尺寸算错 → `nn.Linear` 输入维度不匹配。
- 卷积网络对学习率较敏感，sigmoid 版需较大 lr。
- 忘记把数据与模型放同一 device → 报错（见 [[GPU与设备管理]]）。

## 和前后章节 / 真实项目的连接

- 用到 [[卷积填充与通道]] 与 [[汇聚层]] 的全部组件。
- 现代 CNN（AlexNet/VGG/ResNet）是其深化 → [[AlexNet-VGG-NiN]]。

## 复习卡片

- Q: LeNet 的总体范式？
  A: 卷积+激活+池化堆叠 → 展平 → 全连接分类。
- Q: 卷积主干中空间与通道的变化趋势？
  A: 空间变小、通道变多。
- Q: 全连接首层输入维度由什么决定？
  A: 最后一个卷积/池化输出特征图展平后的元素数。

## 术语

- [[卷积]]、[[感受野]]、[[激活函数]]

## 标签

#d2l #pytorch #cnn #ch06
