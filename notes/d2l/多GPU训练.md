---
title: 多GPU训练
chapter: 第12章 计算性能
section: 12.5-12.7
prev: "[[自动并行与硬件]]"
next: "[[图像增广]]"
up: "[[第12章-计算性能]]"
source: https://zh.d2l.ai/chapter_computational-performance/multiple-gpus.html
tags:
  - d2l
  - pytorch
  - fam/计算
---
# 12.5-12.7 多 GPU 训练与参数服务器

## 一句话总结

**大白话**：人多力量大——把一批数据拆成几份分给几张 GPU，每张各自算自己那份的[[梯度]]，然后大家「凑一起求平均」（all-reduce）再统一更新，保证几张卡始终学的是同一个模型。PyTorch 推荐用 `DistributedDataParallel`（每卡一个进程，效率高）。

**严谨说法**：数据并行把一个 batch 切分到多张 GPU，各自算梯度后**全归约 (all-reduce)** 同步求平均再更新；PyTorch 用 `DataParallel`（简单）或推荐的 `DistributedDataParallel` 实现，参数服务器是另一种聚合范式。

## 本节解决的问题

- 数据并行 vs 模型并行的区别？
- 多卡梯度如何同步（all-reduce / 参数服务器）？
- 为什么推荐 DDP 而非 DataParallel？

## 核心概念 / 公式 / API

### 并行范式
| 范式 | 切分对象 |
| --- | --- |
| 数据并行 | 切分 batch，模型复制到每卡（最常用） |
| 模型并行 | 切分模型层/参数到不同卡（模型太大时） |

### 梯度同步
- **All-Reduce**：各卡梯度求和/平均后广播回所有卡（去中心化，DDP 用）。
- **参数服务器**：worker 上传梯度、server 聚合更新再下发（中心化）。

| API | 说明 |
| --- | --- |
| `nn.DataParallel` | 单进程多线程，受 GIL 限制，不推荐 |
| `nn.parallel.DistributedDataParallel` | 多进程，每卡一进程，高效，推荐 |

- 多卡时有效 batch 变大，学习率通常需按卡数线性放大（配 warmup）。

## 最小可运行代码

```python
import torch
from torch import nn

# 简单数据并行（演示，多用于单机快速试验）
net = nn.Sequential(nn.Linear(20, 10)).cuda()
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
# 生产环境推荐 DistributedDataParallel + torchrun 多进程启动
```

## 易错点

- `DataParallel` 受 Python GIL 与主卡显存瓶颈限制，扩展性差。
- 多卡未按比例调学习率/warmup，导致大 batch 训练不收敛。
- DDP 需正确初始化进程组并保证各进程数据分片不重叠。

## 和前后章节 / 真实项目的连接

- 依赖异步/自动并行机制 [[编译器与异步计算]]、[[自动并行与硬件]]。
- 大学习率需配学习率调度 [[学习率调度器]]。

- 单卡设备管理基础：[[GPU与设备管理]]。

## 复习卡片

- Q: 数据并行如何同步梯度？
  A: All-Reduce 求平均后广播回各卡。
- Q: 为什么推荐 DDP 而非 DataParallel？
  A: DDP 多进程绕过 GIL、扩展性与效率更好。
- Q: 多卡训练学习率怎么调？
  A: 按卡数大致线性放大并配 warmup。

## 标签

#d2l #pytorch #performance #ch12
