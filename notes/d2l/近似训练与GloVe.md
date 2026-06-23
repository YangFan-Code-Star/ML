---
title: 近似训练与GloVe
chapter: 第14章 NLP预训练
section: 14.2-14.5
prev: "[[词嵌入word2vec]]"
next: "[[子词嵌入与相似性]]"
up: "[[第14章-NLP预训练]]"
source: https://zh.d2l.ai/chapter_natural-language-processing-pretraining/approx-training.html
tags:
  - d2l
  - pytorch
  - fam/NLP
---
# 14.2 / 14.5 近似训练与 GloVe

## 一句话总结

**大白话**：word2vec 每猜一个词都要在「整本词典」里做 softmax，太慢。负采样把它简化成「这对词是不是真搭档？再随便抓几个假词当反例」的小判断；层序 softmax 用一棵树把代价从「全词表」降到「树高」。GloVe 换个思路——直接统计「谁和谁经常一起出现」，对这张共现表做拟合。

**严谨说法**：为避免 word2vec 全词表 softmax 的高成本，用**负采样**（变成若干二分类）或**层序 softmax**（哈夫曼树）近似；GloVe 则直接对全局共现矩阵做加权最小二乘，融合全局统计与局部上下文。

## 本节解决的问题

- 如何把昂贵的多分类 softmax 变便宜？
- 负采样的目标函数是什么？
- GloVe 与 word2vec 的本质差异？

## 核心概念 / 公式 / API

### 负采样 (Negative Sampling)
- 把「预测正确上下文」变成：真实词对判正 + 若干随机采样的噪声词判负的二分类。

$$\log\sigma(\mathbf{u}_o^\top\mathbf{v}_c)+\sum_{k}\log\sigma(-\mathbf{u}_{k}^\top\mathbf{v}_c)$$

- 噪声分布常用词频的 3/4 次方。

### 层序 softmax
- 用哈夫曼二叉树，预测复杂度从 $O(V)$ 降到 $O(\log V)$。

### GloVe
- 对全局共现计数 $x_{ij}$ 做加权最小二乘，目标使 $\mathbf{u}_i^\top\mathbf{v}_j$ 拟合 $\log x_{ij}$。
- 结合全局统计（共现矩阵）与局部上下文，训练高效。

## 易错点

- 负采样数 k 太小估计差、太大变慢，需折中。
- 噪声分布直接用原始词频会过度采样高频词，故取 3/4 次幂。
- GloVe 需先统计全局共现矩阵，内存随词表平方增长。

## 和前后章节 / 真实项目的连接

- 解决 [[词嵌入word2vec]] 的 softmax 成本问题。
- 子词与多义词改进 → [[子词嵌入与相似性]]。

## 复习卡片

- Q: 负采样把任务变成什么？
  A: 正样本对 + 若干噪声词的二分类。
- Q: 层序 softmax 的复杂度？
  A: 从 O(V) 降到 O(log V)。
- Q: GloVe 拟合的目标量？
  A: 词向量内积拟合共现计数的对数。

## 术语

- [[词嵌入]]、[[词表]]

## 标签

#d2l #pytorch #nlp #embedding #ch14
