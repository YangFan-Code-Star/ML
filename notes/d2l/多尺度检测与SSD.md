---
title: 多尺度检测与SSD
chapter: 第13章 计算机视觉
section: 13.5-13.7
prev: "[[目标检测与锚框]]"
next: "[[R-CNN系列]]"
up: "[[第13章-计算机视觉]]"
source: https://zh.d2l.ai/chapter_computer-vision/multiscale-object-detection.html
tags:
  - d2l
  - pytorch
  - fam/CV
---
# 13.5-13.7 多尺度检测、数据集与 SSD

## 一句话总结

**大白话**：物体有大有小，一个尺度看不全。SSD 在网络的不同深度「分层设岗」——浅层图细、[[感受野]]小，专抓小目标；深层图粗、感受野大，专抓大目标。一次前向就在所有尺度上同时报出「是什么 + 框在哪」，所以快（单阶段）。

**严谨说法**：SSD 是单阶段检测器：在 CNN 不同深度的特征图上设置多尺度[[锚框]]，浅层大特征图检测小目标、深层小特征图检测大目标，一次前向同时预测类别与边界框偏移。

## 本节解决的问题

- 为什么要在多个尺度的特征图上检测？
- 单阶段检测器如何同时输出类别与框？
- 检测损失如何组合分类与回归？

## 核心概念 / 公式 / API

### 多尺度
- 浅层特征图分辨率高、感受野小 → 检测小物体。
- 深层特征图分辨率低、感受野大 → 检测大物体。

### SSD 结构
- 骨干 + 多个下采样块；每个尺度接两个预测头：
  - 类别预测（每锚框 $c+1$ 类）
  - 边界框偏移预测（每锚框 4 个值）
- **损失 = 分类交叉熵 + 边界框 L1/Smooth-L1 回归**。

## 输入输出形状

- 每尺度类别预测 `(B, a*(c+1), H, W)`、框预测 `(B, a*4, H, W)`，展平拼接所有尺度。

## 最小可运行代码

```python
import torch
from torch import nn

def cls_predictor(in_c, num_anchors, num_classes):
    return nn.Conv2d(in_c, num_anchors * (num_classes + 1), 3, padding=1)

def bbox_predictor(in_c, num_anchors):
    return nn.Conv2d(in_c, num_anchors * 4, 3, padding=1)

# 检测损失 = 分类 + 回归
cls_loss = nn.CrossEntropyLoss(reduction="none")
bbox_loss = nn.L1Loss(reduction="none")
```

## 易错点

- 把所有尺度预测拼接前要统一展平顺序，否则与锚框错位。
- 单阶段速度快但对小目标/密集场景精度可能不如两阶段。
- 正负样本不平衡需难例挖掘（hard negative mining）。

## 和前后章节 / 真实项目的连接

- 锚框/IoU/NMS 基础见 [[目标检测与锚框]]。
- 两阶段对比 R-CNN → [[R-CNN系列]]。

## 复习卡片

- Q: 为什么多尺度特征图检测？
  A: 浅层检小目标、深层检大目标，覆盖不同尺寸。
- Q: SSD 每个尺度输出什么？
  A: 锚框的类别预测与边界框偏移预测。
- Q: 检测损失由哪两部分组成？
  A: 分类交叉熵 + 边界框回归（L1/Smooth-L1）。

## 术语

- [[锚框]]、[[交并比IoU]]、[[感受野]]

## 标签

#d2l #pytorch #cv #detection #ch13
