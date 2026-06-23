---
title: 机器翻译与seq2seq
chapter: 第9章 现代RNN
section: 9.5-9.8
prev: "[[深度与双向RNN]]"
next: "[[注意力提示与汇聚]]"
up: "[[第9章-现代RNN]]"
source: https://zh.d2l.ai/chapter_recurrent-modern/seq2seq.html
tags:
  - d2l
  - pytorch
  - fam/RNN
---
# 9.5-9.8 机器翻译、编码器-解码器、seq2seq 与束搜索

## 一句话总结

**大白话**：翻译要把一句变长的中文变成一句变长的英文。做法分两半（[[编码器-解码器]]）：编码器先把整句中文「读懂」浓缩成一个向量，解码器再据此一个词一个词「写出」英文，写到 `<eos>` 收工。训练时给解码器喂正确答案（[[教师强制]]）训得快；预测时用[[束搜索]]每步留几个候选，比只取最优（贪心）质量更好。

**严谨说法**：编码器-解码器架构把变长输入序列编码为上下文向量，再自回归解码出变长输出，是机器翻译的标准范式；训练用教师强制 (teacher forcing)，预测用束搜索 (beam search) 在效率与质量间折中。

## 本节解决的问题

- 变长序列到变长序列如何建模？
- 训练时如何处理填充与教师强制？
- 贪心、穷举、束搜索的取舍？

## 核心概念 / 公式 / API

### 编码器-解码器
- **编码器**（如 LSTM）读完输入，末隐状态作为上下文 $\mathbf{c}$。
- **解码器**以 $\mathbf{c}$ 初始化，自回归生成目标词，直到 `<eos>`。

### 训练技巧
| 技巧 | 说明 |
| --- | --- |
| 教师强制 | 解码器输入用真实目标词而非上一步预测 |
| 填充 + 掩码 | 批内对齐长度，损失对 `<pad>` 掩码 |
| `<bos>/<eos>` | 标记句子开始/结束 |

### 评估与搜索
- **BLEU**：衡量预测与参考译文的 n-gram 重合度。
- 搜索策略：
  - 贪心：每步取最大概率，快但非全局最优。
  - 穷举：全局最优但指数级不可行。
  - **束搜索**：每步保留 top-k 候选，平衡质量与成本。

## 最小可运行代码

```python
import torch
from torch import nn

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab, embed, hidden, layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed)
        self.rnn = nn.GRU(embed, hidden, layers)
    def forward(self, X):
        X = self.embedding(X.t())          # (T, B, embed)
        output, state = self.rnn(X)
        return output, state               # state 作为解码器初始上下文

# 带掩码的交叉熵：对 <pad> 不计损失
def masked_loss(pred, label, valid_len):
    weights = (torch.arange(label.shape[1])[None, :] < valid_len[:, None]).float()
    loss = nn.functional.cross_entropy(pred.permute(0, 2, 1), label, reduction="none")
    return (loss * weights).mean()
```

## 易错点

- 损失未对填充 token 掩码 → 模型把精力浪费在 `<pad>` 上。
- 束宽 k 越大质量越好但越慢；k=1 即贪心。
- 训练用教师强制、预测无真实标签，存在曝光偏差。

## 和前后章节 / 真实项目的连接

- 上下文向量是固定长度瓶颈 → 注意力机制来缓解 → [[Bahdanau注意力]]。
- 解码器自回归思想贯穿 Transformer/大模型生成 → [[Transformer]]。

## 复习卡片

- Q: 编码器-解码器如何处理变长 I/O？
  A: 编码器压成上下文向量，解码器自回归生成到 `<eos>`。
- Q: 教师强制是什么？
  A: 训练时解码器输入用真实目标词而非自身预测。
- Q: 束搜索相对贪心的优势？
  A: 保留 top-k 候选，质量更高且可控成本。

## 术语

- [[编码器-解码器]]、[[束搜索]]、[[教师强制]]、[[困惑度]]

## 标签

#d2l #pytorch #rnn #seq2seq #nlp #ch09
