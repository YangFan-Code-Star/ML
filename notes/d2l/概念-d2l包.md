---
title: 概念-d2l包
type: 导航
tags:
  - d2l
  - moc
  - fam/导航
---
# d2l 包、#@save 机制与 PyTorch 标准导入

> 返回 [[学习地图MOC]] | 来源 [[前言精读]]

## 一句话总结

**大白话**：书里反复要用的代码被打包成一个工具箱 `d2l`，凡是「以后还要拿来用」的代码块就贴个 `#@save` 标签收进工具箱。本笔记只用 PyTorch 这一套。

**严谨说法**：为避免重复，书中反复使用的函数/类被封装进轻量的 `d2l` 包；凡是要保存进包的代码块都用 `#@save` 标记。本笔记体系**只采用 PyTorch 版本**的导入与实现。

## 核心概念

### `#@save` 机制
- 在某个函数、类或一组导入上方写注释 `#@save`，表示「把这段代码保存到 `d2l` 包」。
- 这样后续章节可以直接 `from d2l import torch as d2l` 复用，而不必重复定义。
- 它本质是 D2L 工具链的一个约定标记，不是 Python 语法。

```python
#@save
def some_reusable_function(x):
    """被 #@save 标记的函数会被收录进 d2l 包，供后续章节复用。"""
    return x
```

### `d2l` 包的依赖
- `d2l` 软件包是**轻量级**的，核心只依赖标准库 + 少量第三方库（`numpy`, `pandas`, `matplotlib`, `requests` 等）。
- 它支持 mxnet / pytorch / tensorflow / paddle 多个后端；本笔记**只用 pytorch 后端**。

### 安装（PyTorch 环境）

```bash
pip install d2l
# PyTorch 本体单独安装（按本机 CUDA 版本选择），例如：
pip install torch torchvision
```

## PyTorch 标准导入约定

前言给出的 PyTorch 版统一导入块（本笔记体系沿用）：

```python
#@save
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
```

`d2l` 通用工具导入块（与框架无关的部分）：

```python
#@save
import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
```

> 注意：上面通用块里包含 `IPython.display` 与 `matplotlib_inline`，它们主要服务于 Jupyter 内联绘图。
> 由于本笔记体系**只跑纯 Python 脚本（不使用 Jupyter）**，在脚本中绘图可改用 `plt.show()` 或 `plt.savefig(...)`，无需 inline 后端。

## 各导入项作用速查

| 导入                           | 作用                                         |
| ---------------------------- | ------------------------------------------ |
| `torch`                      | 张量与自动微分核心                                  |
| `torch.nn` (`nn`)            | 神经网络层与模型容器（`nn.Linear`, `nn.Sequential` 等） |
| `torch.nn.functional` (`F`)  | 无状态函数式算子（`F.relu`, `F.cross_entropy` 等）    |
| `torch.utils.data` (`data`)  | `Dataset` / `DataLoader` 数据管道              |
| `torchvision` / `transforms` | 视觉数据集与图像预处理                                |
| `PIL.Image`                  | 图像读写                                       |

## 与其他笔记的连接

- 「从零 vs 简洁」两个版本都基于这套导入约定 → [[概念-在实践中学习]]。
- 「工程方法」要素（设备、数值、性能）从这里的环境配置起步 → [[概念-四要素]]。

## 易错点

- 误把 `#@save` 当成 Python 装饰器或语法——它只是 D2L 的注释约定。
- 在纯脚本环境照搬 Jupyter 的 inline 绘图代码，导致图不显示——改用 `plt.show()` / `plt.savefig()`。
- `d2l` 包版本与 PyTorch 版本不匹配时可能报错，优先核对官方安装页。

## 复习卡片

- Q: `#@save` 的作用？
  A: 标记代码块保存进 `d2l` 包，供后续章节复用。
- Q: 本笔记体系用哪个后端？
  A: 仅 PyTorch，纯 Python 脚本，不使用 Jupyter。
- Q: `F` 和 `nn` 的区别？
  A: `nn` 是有状态的层/模块；`F` 是无状态的函数式算子。

## 标签

#d2l #concept #pytorch #tooling
