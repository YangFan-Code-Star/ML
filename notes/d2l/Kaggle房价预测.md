---
title: Kaggle房价预测
chapter: 第4章 多层感知机
section: 4.10
prev: "[[分布偏移]]"
next: "[[层和块]]"
up: "[[第4章-多层感知机]]"
source: https://zh.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html
tags:
  - d2l
  - pytorch
  - fam/基础
---
# 4.10 实战 Kaggle：房价预测

> **怎么读这篇**：上半部分是 d2l 教材流程；下半部分「为新问题构建合适模型的方法论」是结合本仓库实战 `d2l/code/4.10实战kaggle比赛预测房价.py` 总结的可迁移套路——面对一个陌生数据集时，照着走就行。

## 一句话总结

**大白话**：拿真实房价数据从头跑一遍完整流程——把脏数据洗干净（数值拉到同一量纲、类别变独热、缺失值补上）→ 建模 → 用「对数误差」打分（房价有贵有便宜，关心的是「差几个百分比」而不是「差几块钱」）→ K 折轮流验证选超参 → 导出提交文件。这是第一部分知识的总演练。

**严谨说法**：用真实 Kaggle 房价数据走完端到端流程：特征预处理（标准化数值、独热类别、填补缺失）→ 模型 → 用**相对误差（对数 RMSE）**评估 → K 折交叉验证选超参 → 生成提交，是第一部分知识的综合实战。

## 本节解决的问题

- 如何把含混合类型/缺失值的真实表格数据变成训练张量？
- 房价跨度大时为什么用对数均方根误差？
- 如何用 K 折交叉验证选学习率/正则等超参？
- **面对一个新问题，怎样一步步搭出「合适」的模型，而不是上来就堆大网络？**

## 核心概念 / 公式 / API

### 预处理流程
1. 数值特征**标准化**为零均值单位方差，再填 0 补缺失（标准化后 0 即均值，是合理默认）。
2. 类别特征 `pd.get_dummies(dummy_na=True)` 独热（缺失值单独成一类）。
3. 训练/测试**拼在一起**做预处理，天然保证两边列集合一致，再拆回、转张量。

### 评估指标：对数 RMSE
> **大白话**：贵房子差 1 万和便宜房子差 1 万，严重程度不同；取对数后比的是「差百分之几」。

$$\sqrt{\frac{1}{n}\sum_i\big(\log y_i-\log\hat{y}_i\big)^2}$$

实现技巧：把标签先 `log` 变换，训练直接用 MSE，其平方根即对数 RMSE；预测时用 `exp` 还原量纲。

### K 折交叉验证
- 数据分 K 份，轮流留一份验证、其余训练，取**平均验证误差**——这是判断模型好坏、选超参的可信依据（单次划分太靠运气）。

---

## 为新问题构建合适模型的方法论（实战提炼）

### 建模六步套路（换任何数据集都按这六步想）

| 步骤   | 做什么                            | 本例对应                                           |
| ---- | ------------------------------ | ---------------------------------------------- |
| 1 数据 | 读入→洗缺失→数值标准化→类别独热→目标变换         | `load_data()`，目标 `log` 变换                      |
| 2 模型 | 配置驱动 + 可复用 Block 搭网络           | `MLPBlock` / `HousePriceNet` / `build_model()` |
| 3 损失 | 选与业务/比赛指标一致的损失                 | log 空间 MSE = 对数 RMSE                           |
| 4 优化 | Adam + `weight_decay`(L2) 控过拟合 | `torch.optim.Adam`                             |
| 5 评估 | K 折交叉验证得可信泛化误差                 | `k_fold()`                                     |
| 6 预测 | 全量重训→预测→还原量纲→写提交               | `train_and_pred()`                             |

### 工程化搭模型（而非一行 `nn.Sequential`）

学到的是工业界标准写法，可移植到任何任务：

- **可复用基础块** `MLPBlock(nn.Module)`：把 `Linear→(BN)→激活→(Dropout)` 打包成一个语义单元——这几层几乎总成套出现，封装后堆叠不易写错、想换激活/正则只改一处。
- **配置驱动主模型** `HousePriceNet(nn.Module)`：用 `nn.ModuleList`/循环按 `hidden_dims` 列表**程序化堆叠** Block，改结构只改配置不改代码（`hidden_dims=[]` 自动退化为线性回归，方便对比）。
- **权重初始化** `_init_weights()`：`Linear` 用 Kaiming(适配 ReLU)，`BatchNorm` 的 weight=1/bias=0——好初始化显著影响收敛，是工程必备习惯。详见 [[参数初始化]]。
- **工厂函数** `build_model(input_dim, cfg)`：统一「配置→模型实例」入口，K 折与全量训练复用同一份，保证一致。
- **Config 集中超参**：换数据/调参只动一处，是可复现实验的基础。

### 关键实证：模型不是越大越好（本次 K 折扫参结论）

同一套数据、同一流程，只改模型/正则，5 折平均验证对数 RMSE：

| 配置                                    | valid log-RMSE | 现象                   |
| ------------------------------------- | -------------- | -------------------- |
| `[256,64]` + BN + dropout0.2 + wd1e-3 | 0.32           | 正则太强，**欠拟合**         |
| `[256,64]` + BN + wd1e-4              | 0.45           | **严重过拟合**，预测爆到 196 万 |
| **`[64]` 无BN + wd5e-3** ✅             | **0.15**       | 简单稳健，最终采用            |
| `[128,32]` + dropout0.2               | 0.36           | 偏大且不稳                |

> **核心教训**：在「样本少（1460）+ 高维稀疏独热（346 维）」的表格数据上，**更深更宽 + BatchNorm 反而有害**；简单的单隐藏层 + 较强 L2 正则才最稳。模型容量要匹配数据规模，详见 [[模型选择与过拟合]]、[[权重衰退]]。

### 面对新问题的选型决策

1. **先建最简基线**：线性/单隐藏层，跑通端到端 + K 折，拿到一个「分数下限」。
2. **看 train vs valid 差距判断方向**：
   - 两者都高 → 欠拟合 → 加容量 / 减正则 / 多训。
   - train 低、valid 高 → 过拟合 → 加正则（[[权重衰退]]、[[暂退法Dropout]]）/ 减容量 / 加数据。
3. **逐步加复杂度，每步用 K 折验证是否真的变好**，不被单次划分误导。
4. **正则与归一化要看数据规模**：小数据慎用 BN（统计量不稳），优先 weight_decay；大数据再考虑 BN（[[批量规范化]]）。
5. **特征工程常比堆网络更划算**：偏态数值列再 `log`、构造交互特征往往比加层更有效。

## 输入输出形状

- `X(B, 346) → 隐藏层(B, 64) → 输出(B, 1)`（本例独热后 346 维）。

## 最小可运行代码

仓库脚本：`d2l/code/4.10实战kaggle比赛预测房价.py`（含完整工程化实现）

```python
import torch
from torch import nn

# --- 工程化模型：可复用 Block + 配置驱动 ---
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, use_bn=False):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class HousePriceNet(nn.Module):
    def __init__(self, in_features, hidden_dims=(64,)):
        super().__init__()
        blocks, prev = [], in_features
        for h in hidden_dims:            # 按配置程序化堆叠，改结构不改代码
            blocks.append(MLPBlock(prev, h))
            prev = h
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, 1)
        self.apply(self._init)           # 权重初始化是工程习惯
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.head(self.backbone(x))

# 标签先 log，训练用 MSE，sqrt 即对数 RMSE；预测时 exp 还原
def log_rmse(net, X, y):
    return torch.sqrt(nn.functional.mse_loss(net(X), y)).item()
```

## 易错点

- 标准化用训练集统计量，避免数据泄露（书中用全量近似，工程上应只用训练集）。
- 直接对原始房价算 RMSE，被高价样本主导；改用对数 RMSE。
- 独热后特征维度暴增（346 维），必须配权重衰退防过拟合。
- **小数据上盲目堆深网络 + BatchNorm 会更差**；先建简单基线再逐步加复杂度。
- 调参只看一次划分不可信，用 K 折平均。

## 和前后章节 / 真实项目的连接

- 综合运用 [[数据操作与预处理]]、[[模型选择与过拟合]]、[[权重衰退]]、[[暂退法Dropout]]、[[批量规范化]]、[[参数初始化]]。
- 工程化搭模型（Block / 配置驱动 / 工厂函数）衔接下一章 [[层和块]]、[[参数管理]]、[[自定义层]]。
- 端到端 + 六步套路是日后所有 Kaggle/真实项目的模板。

## 复习卡片

- Q: 为什么用对数 RMSE？
  A: 房价跨度大，关注相对误差，对数化后误差更均衡。
- Q: 建模六步套路是什么？
  A: 数据 → 模型 → 损失 → 优化 → 评估(K折) → 预测。
- Q: train 低但 valid 高说明什么？怎么办？
  A: 过拟合；加正则（weight_decay/dropout）、减容量或加数据。
- Q: 小数据集为什么慎用 BatchNorm？
  A: 批统计量不稳定，容易更差；优先用 weight_decay 正则。
- Q: 怎样选超参更稳健？
  A: K 折交叉验证取平均验证误差，逐步加复杂度并验证。

## 术语

- [[过拟合]]、[[正则化]]、[[张量]]、[[学习率]]、[[参数初始化]]、[[归一化]]

## 标签

#d2l #pytorch #mlp #kaggle #ch04
