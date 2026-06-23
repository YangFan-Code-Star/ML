---
title: Transformer
chapter: 第10章 注意力机制
section: 10.7
prev: "[[自注意力与位置编码]]"
next: "[[优化与凸性]]"
up: "[[第10章-注意力机制]]"
source: https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html
tags:
  - d2l
  - pytorch
  - fam/注意力
---
# 10.7 Transformer

## 一句话总结

**大白话**：Transformer 把 RNN 的「一个个顺序读」彻底换成「[[自注意力|一眼看全局]]」，于是能大规模并行、训得又快又好。它的积木很统一：每层都是「多头自注意力 → [[残差连接|加一下原输入]]再[[归一化]] → 一个小前馈网络 → 再加再归一」。堆几层就成，是 BERT/GPT 等大模型的共同地基。

**严谨说法**：Transformer 完全用注意力替代循环：编码器/解码器由「多头自注意力 + 逐位前馈网络 + [[残差连接|残差]] + 层规范化」堆叠而成，可大规模并行训练，是 BERT/GPT 等现代大模型的基础架构。

## 本节解决的问题

- Transformer 编码器层/解码器层的组成？
- 残差 + LayerNorm、逐位前馈网络的作用？
- 解码器的掩码自注意力与编码器-解码器注意力区别？

## 核心概念 / 公式 / API

### 编码器层（重复 N 次）
1. 多头**自注意力**（Q=K=V=输入）
2. **Add & Norm**：残差 + LayerNorm
3. **逐位前馈网络 (FFN)**：两层 MLP，对每个位置独立作用
4. 再一次 Add & Norm

### 解码器层
1. **掩码**多头自注意力（防止看到未来词）
2. **编码器-解码器注意力**（query 来自解码器，key/value 来自编码器输出）
3. 逐位前馈 + Add & Norm

| 组件 | 作用 |
| --- | --- |
| 残差连接 | 缓解深层梯度问题（见 [[ResNet与DenseNet]]） |
| LayerNorm | 对每个样本特征维归一化，适合变长序列 |
| FFN | 逐位置非线性变换，扩展表达 |
| 掩码 | 解码自回归不泄露未来 |

## 最小可运行代码

```python
import torch
from torch import nn

# 用 PyTorch 内置 Transformer 编码器层
layer = nn.TransformerEncoderLayer(d_model=128, nhead=8,
                                   dim_feedforward=512, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=6)
x = torch.randn(2, 10, 128)        # (B, seq, d)
print(encoder(x).shape)            # (2, 10, 128)
```

## 易错点

- 解码器自注意力忘记加因果掩码 → 训练泄露未来词。
- 位置编码不可少（自注意力无序）→ 见 [[自注意力与位置编码]]。
- LayerNorm（非 BatchNorm）才适合变长序列与小 batch。

## 和前后章节 / 真实项目的连接

- 综合 [[多头注意力]]、[[自注意力与位置编码]]、残差与 LayerNorm。
- 直接通往预训练大模型：编码器→BERT [[BERT模型与预训练]]，解码器→GPT 类生成。
- 真实项目：视觉 Transformer (ViT)、DINOv3 等多模态骨干。

- 归一化对比：本节用 LayerNorm 而非 [[批量规范化]] 的 BatchNorm。

## 复习卡片

- Q: Transformer 编码器层四件套？
  A: 多头自注意力 + Add&Norm + 逐位前馈 + Add&Norm。
- Q: 解码器为何要掩码自注意力？
  A: 自回归生成时防止看到未来位置。
- Q: 为何用 LayerNorm 而非 BatchNorm？
  A: 适合变长序列、对 batch 大小不敏感。

## 术语

- [[注意力机制]]、[[自注意力]]、[[位置编码]]、[[残差连接]]、[[归一化]]

## 标签

#d2l #pytorch #attention #transformer #ch10
