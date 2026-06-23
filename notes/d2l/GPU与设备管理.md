---
title: GPU与设备管理
chapter: 第5章 深度学习计算
section: 5.6
prev: "[[读写文件]]"
next: "[[从全连接到卷积]]"
up: "[[第5章-深度学习计算]]"
source: https://zh.d2l.ai/chapter_deep-learning-computation/use-gpu.html
tags:
  - d2l
  - pytorch
  - fam/计算
---
# 5.6 GPU 与设备管理

## 一句话总结

**大白话**：GPU 是算得飞快的「专用车间」。要加速，得先把数据和模型都 `.to(device)` 搬进同一个车间——而且一起参与运算的东西必须在同一个车间，否则报错。来回搬运很费时间，能不搬就别搬。

**严谨说法**：把[[张量]]与模型 `.to(device)` 搬到 GPU 才能加速；参与同一运算的张量必须在同一设备上，数据加载与计算尽量减少 CPU↔GPU 拷贝。

## 本节解决的问题

- 如何查询并指定计算设备？
- 如何把张量/模型放到 GPU？
- 跨设备运算为什么报错？

## 核心概念 / 公式 / API

| 操作 | API |
| --- | --- |
| 检测 | `torch.cuda.is_available()`, `torch.cuda.device_count()` |
| 设备对象 | `torch.device("cuda:0")` / `"cpu"` |
| 搬运 | `x.to(device)`, `net.to(device)` |
| 查看所在设备 | `x.device` |

```python
import torch
def try_gpu(i=0):
    if torch.cuda.device_count() > i:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")
```

- 模型参数与输入数据必须在同一设备，否则报 `Expected all tensors to be on the same device`。

## 最小可运行代码

```python
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = nn.Sequential(nn.Linear(3, 1)).to(device)   # 模型上 GPU
X = torch.rand(2, 3, device=device)               # 数据上 GPU
print(net(X).device)                              # cuda:0
```

## 易错点

- 模型在 GPU、数据在 CPU（或反之）→ 设备不一致报错。
- 频繁 `.cpu()/.cuda()` 拷贝拖慢训练；尽量在 GPU 上连续计算。
- `.item()` / `print` 会触发同步，循环内频繁调用影响性能。

## 和前后章节 / 真实项目的连接

- 训练循环中需把每个 batch `.to(device)`，贯穿 CNN/RNN 训练。
- 多 GPU 与性能优化见 [[多GPU训练]]。

- 多卡扩展（数据并行）：[[多GPU训练]]。

## 复习卡片

- Q: 跨设备运算报什么错？如何解决？
  A: tensors not on same device；把两者 `.to(同一 device)`。
- Q: 如何把模型搬到 GPU？
  A: `net.to(device)`。
- Q: 怎样减少性能损耗？
  A: 减少 CPU↔GPU 拷贝与频繁同步（如 `.item()`）。

## 标签

#d2l #pytorch #computation #gpu #ch05
