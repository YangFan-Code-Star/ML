---
title: 微调BERT下游任务
chapter: 第15章 NLP应用
section: 15.6-15.7
prev: "[[自然语言推断]]"
up: "[[第15章-NLP应用]]"
source: https://zh.d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html
tags:
  - d2l
  - pytorch
  - fam/NLP
---
# 15.6-15.7 针对下游任务微调 BERT

## 一句话总结

**大白话**：一个预训练好的 BERT 能干各种活——只要在它头上接一个小小的「任务帽子」再[[微调|微调]]一下就行。整句任务（分类）用 `<cls>` 这个「全句摘要」位置；逐词任务（标注、问答）用每个词自己的表示。这是当今 NLP 落地的统一套路。

**严谨说法**：下游任务只需在预训练 BERT 上加一个轻量任务头并端到端[[微调]]：句级任务用 `<cls>` 表示接全连接，词级任务（如序列标注/问答）用每个词元的表示，是 NLP 现代落地的统一范式。

## 本节解决的问题

- 不同类型 NLP 任务如何复用同一个 BERT？
- 句级 vs 词元级任务的输出取法？
- 微调的工程注意点？

## 核心概念 / 公式 / API

| 任务类型 | 用哪个表示 | 例子 |
| --- | --- | --- |
| 单句分类 | `<cls>` 向量 → 全连接 | 情感分析 |
| 句子对分类 | `<cls>` 向量（输入 `句A <sep> 句B`） | 自然语言推断 |
| 序列标注 | 每个词元表示 → 全连接 | 命名实体识别 |
| 问答 | 词元表示预测答案起止位置 | SQuAD |

- 微调：加载预训练权重 + 新任务头（随机初始化），用小学习率端到端训练。
- 与从零训练相比，数据需求小、收敛快、精度高。

## 最小可运行代码

```python
import torch
from torch import nn

class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden, num_classes):
        super().__init__()
        self.bert = bert                       # 预训练编码器
        self.head = nn.Linear(hidden, num_classes)
    def forward(self, tokens, segments):
        encoded = self.bert(tokens, segments)  # (B, L, hidden)
        cls = encoded[:, 0, :]                 # 取 <cls> 表示
        return self.head(cls)

opt = torch.optim.AdamW(net.parameters(), lr=2e-5)   # 微调常用小 lr
```

## 易错点

- 微调学习率过大（如 1e-3）会破坏预训练知识，常用 2e-5~5e-5。
- 句子对任务忘记加段嵌入/`<sep>` 分隔。
- 词级任务误用 `<cls>`，应取对应词元表示。

## 和前后章节 / 真实项目的连接

- 复用 [[BERT模型与预训练]] 的预训练模型；微调思想同 [[微调]]。
- 优化器 AdamW 见 [[Adam]]。
- 真实项目：基于 BERT/大模型微调是当前 NLP 工程主流，对应 LLaMA-Factory 等微调框架。

- 迁移学习/微调思想最早见于视觉：[[微调]]。

## 复习卡片

- Q: 句级任务用 BERT 的哪个表示？
  A: `<cls>` 词元的输出向量。
- Q: 词级任务（如标注）取什么？
  A: 每个词元各自的输出表示。
- Q: 微调常用学习率量级？
  A: 很小，约 2e-5~5e-5（配 AdamW）。

## 术语

- [[迁移学习]]、[[掩码语言模型]]、[[自监督学习]]

## 标签

#d2l #pytorch #nlp #bert #ch15
