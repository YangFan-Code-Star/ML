---
title: R-CNN系列
chapter: 第13章 计算机视觉
section: 13.8
prev: "[[多尺度检测与SSD]]"
next: "[[语义分割与FCN]]"
up: "[[第13章-计算机视觉]]"
source: https://zh.d2l.ai/chapter_computer-vision/rcnn.html
tags:
  - d2l
  - pytorch
  - fam/CV
---
# 13.8 区域卷积神经网络 (R-CNN 系列)

## 一句话总结

**大白话**：两阶段检测就是「先圈出可能有东西的区域，再逐个细看是什么、框得准不准」。这一系列在不断「去重复、提速」：R-CNN 每个候选都单独跑一遍 CNN（极慢）→ Fast R-CNN 整图只跑一次、用 RoI 池化抠特征 → Faster R-CNN 连「圈区域」都交给网络自己学（RPN），全程端到端。

**严谨说法**：R-CNN 系列是两阶段检测器：先提取候选区域再分类回归；从 R-CNN → Fast R-CNN（共享特征 + RoI 池化）→ Faster R-CNN（用 RPN 学习候选区域）逐步端到端化、提速并提精度。

## 本节解决的问题

- 两阶段检测的「区域提议 + 分类」流程？
- RoI 池化解决什么问题？
- Faster R-CNN 的 RPN 如何替代手工候选？

## 核心概念 / 公式 / API

| 模型 | 关键改进 |
| --- | --- |
| R-CNN | 选择性搜索取候选 → 各自过 CNN 分类（慢，重复计算） |
| Fast R-CNN | 整图过一次 CNN，**RoI 池化**抠取候选特征，共享计算 |
| Faster R-CNN | 用**区域提议网络 RPN**端到端学习候选框 |
| Mask R-CNN | 加 RoI Align + 掩码分支，做实例分割 |

- **RoI 池化/Align**：把不同大小候选区域映射成固定尺寸特征，便于送入全连接。

## 最小可运行代码

```python
import torch
import torchvision

# 直接使用 torchvision 的 Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
x = [torch.rand(3, 600, 800)]
preds = model(x)            # 含 boxes / labels / scores
print(preds[0].keys())
```

## 易错点

- R-CNN 对每个候选独立跑 CNN，重复计算极慢——Fast R-CNN 才共享特征。
- RoI 池化的量化误差影响定位精度，Mask R-CNN 用 RoI Align 改进。
- 两阶段精度高但速度通常慢于单阶段 SSD/YOLO。

## 和前后章节 / 真实项目的连接

- 与单阶段 SSD 对比 [[多尺度检测与SSD]]；锚框基础 [[目标检测与锚框]]。
- 实例分割延伸到语义分割 → [[语义分割与FCN]]。
- 真实检测项目该选哪个架构、用不用预训练：见 [[建模方法论]]。

## 复习卡片

- Q: 两阶段检测的两个阶段？
  A: 区域提议 + 对候选区域分类与边界框回归。
- Q: RoI 池化的作用？
  A: 把不同大小候选映射为固定尺寸特征。
- Q: Faster R-CNN 的核心创新？
  A: 用 RPN 端到端学习候选区域。

## 术语

- [[锚框]]、[[交并比IoU]]、[[迁移学习]]

## 标签

#d2l #pytorch #cv #detection #ch13
