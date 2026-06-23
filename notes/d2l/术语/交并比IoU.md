---
title: 交并比IoU
aliases:
  - IoU
  - Intersection over Union
  - 交并比
tags:
  - d2l
  - 术语
---
# 交并比 (IoU, Intersection over Union)

> **大白话**：衡量两个框「重叠得有多好」。把「重叠面积」除以「合起来的总面积」——完全重合是 1，毫不相干是 0。

## 定义
$$\mathrm{IoU}(A,B)=\frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}.$$

## 用途
- 给[[锚框]]分配正/负样本（IoU 高→正样本）；
- NMS 中判断两个预测框是否冗余；
- 检测/分割的评估指标（如 mAP 基于不同 IoU 阈值）。

## 出现章节
[[目标检测与锚框]]、[[多尺度检测与SSD]]、[[R-CNN系列]]。

## 关联术语
- [[锚框]]
- 回到 [[术语索引]]
