"""
实战 Kaggle 比赛：预测房价（House Prices - Ames 数据集）
================================================================

本脚本的目标不是"跑出一个结果"，而是让你学到一套**可迁移到任何表格数据集的建模套路**。
读完一遍，你应该能把同样的结构套用到别的回归/分类任务上。

------------------------------------------------------------
构建一个模型的「六步套路」（换数据集时按这六步思考即可）
------------------------------------------------------------
  1. 数据 (Data)     ：读入 -> 清洗缺失 -> 数值标准化 -> 类别 one-hot -> 目标变换
  2. 模型 (Model)    ：用「配置驱动 + 可复用 Block」的工程化方式搭网络（本文件重点）
  3. 损失 (Loss)     ：选与业务/比赛指标一致的损失，这里是 log 空间的 RMSE
  4. 优化 (Optimizer)：Adam + weight_decay(L2 正则)，控制过拟合
  5. 评估 (Eval)     ：K 折交叉验证，得到「可信」的泛化误差，用于调参
  6. 预测 (Predict)  ：用全部训练数据重训 -> 预测测试集 -> 还原量纲 -> 写提交文件

工程化的关键思想贯穿全文：
  - 所有超参集中在 Config，换数据/调参只改一处；
  - 模型由「配置 -> 自动搭建」，改结构不改代码；
  - 数据预处理的统计量(均值/方差/列集合)训练测试共享，杜绝数据泄漏与列错位。
"""

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. 配置区 (Config)
# ------------------------------------------------------------
# 工程实践：把"会变的东西"全部抽到一个配置对象里。
# 换数据集、调参、做实验，原则上只动这里，其余代码保持不变。
# ============================================================
@dataclass
class Config:
    # --- 路径 ---
    data_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "HousePrice")
    )

    # --- 模型结构（配置驱动的核心）---
    # hidden_dims 是隐藏层宽度列表：
    #   [256, 64]  -> 两个隐藏层的 MLP
    #   []         -> 没有隐藏层，自动退化为「线性回归」，方便对比学习
    # 经 K 折实验确定：单隐藏层 [64] 在本数据集上最稳（更深更宽反而过拟合）。
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    dropout: float = 0.0          # Dropout 概率；本数据集主要靠 weight_decay 正则即可
    use_batchnorm: bool = False   # 本数据量小、特征是高维稀疏 one-hot，BN 实测反而更差，故关闭
    activation: str = "relu"      # 激活函数名，集中管理便于切换

    # --- 训练超参 ---
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 5e-3    # L2 正则强度，过强会欠拟合，过弱会过拟合，需折中
    k_folds: int = 5              # K 折交叉验证的折数

    seed: int = 42

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 激活函数注册表：用字符串选择，避免在代码里到处 if-else
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


# ============================================================
# 1. 数据 (Data)
# ------------------------------------------------------------
# 关键原则：训练集和测试集必须经过「完全相同」的预处理，
# 且数值标准化用的均值/方差、one-hot 后的列集合，都要以训练集为准，
# 否则会发生「数据泄漏」或「列对不齐」，导致测试结果不可信。
#
# 这里采用 d2l 经典做法：把 train/test 的特征**拼在一起**统一处理，
# 这样能天然保证两边列集合一致；标准化用的是合并后的统计量（对本任务足够）。
# ============================================================
def load_data(cfg: Config):
    """读取并预处理房价数据。

    Returns:
        train_features: Tensor (n_train, d)  训练特征
        train_labels:   Tensor (n_train, 1)  log 变换后的标签
        test_features:  Tensor (n_test,  d)  测试特征
        test_ids:       np.ndarray           测试集的 Id 列（写提交文件用）
    """
    train_df = pd.read_csv(os.path.join(cfg.data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(cfg.data_dir, "test.csv"))

    n_train = train_df.shape[0]
    test_ids = test_df["Id"].values

    # 目标变量：房价右偏严重，取 log 后更接近正态，训练更稳定，
    # 也正好契合比赛指标(对 log 价格算 RMSE)。
    train_labels = np.log(train_df["SalePrice"].values).astype(np.float32)

    # 去掉 Id（无信息）和 SalePrice（标签），再把训练/测试特征拼起来统一处理。
    all_features = pd.concat(
        [train_df.drop(["Id", "SalePrice"], axis=1), test_df.drop(["Id"], axis=1)],
        axis=0,
    )

    # MSSubClass 是用数字编码的「类别」，要当类别处理而非数值。
    if "MSSubClass" in all_features.columns:
        all_features["MSSubClass"] = all_features["MSSubClass"].astype(str)

    numeric_cols = all_features.select_dtypes(include=[np.number]).columns
    categorical_cols = all_features.select_dtypes(exclude=[np.number]).columns

    # 数值列：标准化到均值 0、方差 1（让各特征量纲一致，利于优化），
    # 标准化后用 0 填充缺失（此时 0 即均值，是合理的默认值）。
    all_features[numeric_cols] = all_features[numeric_cols].apply(
        lambda col: (col - col.mean()) / (col.std() if col.std() > 0 else 1.0)
    )
    all_features[numeric_cols] = all_features[numeric_cols].fillna(0)

    # 类别列：缺失值单独成一类(dummy_na=True)，再做 one-hot。
    all_features = pd.get_dummies(
        all_features, columns=list(categorical_cols), dummy_na=True
    )
    # get_dummies 可能产生 bool 列，统一转成 float。
    all_features = all_features.astype(np.float32)

    # 拆回训练/测试，并转成 Tensor。
    feats = torch.tensor(all_features.values, dtype=torch.float32)
    train_features = feats[:n_train]
    test_features = feats[n_train:]
    train_labels = torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1)

    return train_features, train_labels, test_features, test_ids


# ============================================================
# 2. 模型 (Model) —— 工程化、配置驱动的搭建方式（本文件重点）
# ------------------------------------------------------------
# 不用「一行 nn.Sequential 二选一」，而是用工业界常见的三件套：
#   (a) 可复用的基础块 MLPBlock：把若干层打包成一个语义单元；
#   (b) 主模型 HousePriceNet：按 Config 程序化堆叠 Block；
#   (c) 工厂函数 build_model：统一「配置 -> 模型实例」的入口。
# 这样改网络结构只需改配置，且 Block 可被其它模型复用。
# ============================================================
class MLPBlock(nn.Module):
    """一个标准的前馈单元：Linear -> (BatchNorm) -> 激活 -> (Dropout)

    为什么打包成 Block：
      - 这几层几乎总是「成套出现」，封装后堆叠时不易写错顺序；
      - 想给整个网络换激活/加正则，只改 Block 一处；
      - Block 可以被任何需要 MLP 的模型复用（可移植性）。
    """

    def __init__(self, in_dim, out_dim, activation="relu",
                 use_batchnorm=True, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if use_batchnorm:
            # BN 放在激活前：规范化线性输出，缓解内部协变量偏移，训练更快更稳。
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(_ACTIVATIONS[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class HousePriceNet(nn.Module):
    """房价回归网络：根据配置自动堆叠若干 MLPBlock，再接一个回归输出头。

    hidden_dims 为空时只剩输出头 -> 等价于线性回归，便于和 MLP 对比。
    """

    def __init__(self, input_dim, cfg: Config):
        super().__init__()

        # 用 ModuleList + 循环「程序化」搭建隐藏层：
        # 配置说几层、每层多宽，这里就生成多少个 Block，无需手写每一层。
        blocks = []
        prev_dim = input_dim
        for h_dim in cfg.hidden_dims:
            blocks.append(
                MLPBlock(
                    prev_dim, h_dim,
                    activation=cfg.activation,
                    use_batchnorm=cfg.use_batchnorm,
                    dropout=cfg.dropout,
                )
            )
            prev_dim = h_dim
        self.backbone = nn.Sequential(*blocks)

        # 输出头：回归到 1 维（房价的 log 值）。输出层不加激活/正则。
        self.head = nn.Linear(prev_dim, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """权重初始化：好的初始化能显著影响收敛速度与稳定性，是工程必备习惯。

        - Linear：用 Kaiming(He) 初始化，适配 ReLU 系激活，保持各层方差稳定；
        - BatchNorm：weight 初始化为 1、bias 为 0（即初始时不改变分布）。
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(input_dim: int, cfg: Config) -> nn.Module:
    """工厂函数：把「配置 -> 模型实例」封装成唯一入口。

    K 折评估和最终全量训练都调它，保证每次构建的模型完全一致。
    """
    return HousePriceNet(input_dim, cfg)


# ============================================================
# 3. 损失 / 指标 (Loss & Metric)
# ------------------------------------------------------------
# 比赛官方指标是「对 log 价格算 RMSE」。由于标签本身已是 log 价格，
# 训练时直接对预测与标签用 MSE，其平方根就是 log-RMSE，二者一致。
# ============================================================
def log_rmse(net: nn.Module, features: torch.Tensor, labels: torch.Tensor,
             device: torch.device) -> float:
    """计算 log 空间的 RMSE（评估用，eval 模式 + 不记录梯度）。"""
    net.eval()
    with torch.no_grad():
        preds = net(features.to(device))
        loss = nn.functional.mse_loss(preds, labels.to(device))
    return float(torch.sqrt(loss).item())


# ============================================================
# 4. 训练 (Train)
# ------------------------------------------------------------
# 标准训练循环：DataLoader 取小批量 -> 前向 -> 反向 -> 更新。
# 返回每个 epoch 的 train/val log-RMSE，便于观察是否过拟合/欠拟合。
# ============================================================
def train(net, train_features, train_labels, val_features, val_labels, cfg: Config):
    device = cfg.device
    net = net.to(device)

    train_ds = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    criterion = nn.MSELoss()
    # Adam：自适应学习率，对特征尺度不敏感、收敛快；weight_decay 即 L2 正则。
    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_history, val_history = [], []
    for _ in range(cfg.epochs):
        net.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(X), y)
            loss.backward()
            optimizer.step()

        train_history.append(log_rmse(net, train_features, train_labels, device))
        if val_features is not None:
            val_history.append(log_rmse(net, val_features, val_labels, device))

    return train_history, val_history


# ============================================================
# 5. 评估 (Eval) —— K 折交叉验证
# ------------------------------------------------------------
# 为什么要 K 折：单次划分的验证分数受「恰好分到哪些样本」影响大，不可靠。
# K 折把数据切成 K 份，轮流用其中 1 份做验证、其余做训练，取平均，
# 得到对泛化误差更稳健的估计，是调参时判断好坏的主要依据。
# ============================================================
def _get_kth_fold(features, labels, k, i):
    """取第 i 折作为验证集，其余作为训练集。"""
    fold_size = features.shape[0] // k
    val_idx = slice(i * fold_size, (i + 1) * fold_size)

    val_features = features[val_idx]
    val_labels = labels[val_idx]
    # 训练部分 = 验证段之前 + 验证段之后
    train_features = torch.cat([features[: i * fold_size],
                                features[(i + 1) * fold_size:]], dim=0)
    train_labels = torch.cat([labels[: i * fold_size],
                              labels[(i + 1) * fold_size:]], dim=0)
    return train_features, train_labels, val_features, val_labels


def k_fold(cfg: Config, features, labels):
    """跑 K 折交叉验证，返回平均训练/验证 log-RMSE。"""
    input_dim = features.shape[1]
    train_scores, val_scores = [], []

    for i in range(cfg.k_folds):
        tr_X, tr_y, val_X, val_y = _get_kth_fold(features, labels, cfg.k_folds, i)
        net = build_model(input_dim, cfg)
        tr_hist, val_hist = train(net, tr_X, tr_y, val_X, val_y, cfg)
        train_scores.append(tr_hist[-1])
        val_scores.append(val_hist[-1])
        print(f"  fold {i + 1}/{cfg.k_folds}: "
              f"train log-RMSE={tr_hist[-1]:.4f}, valid log-RMSE={val_hist[-1]:.4f}")

    avg_train = sum(train_scores) / len(train_scores)
    avg_val = sum(val_scores) / len(val_scores)
    return avg_train, avg_val


# ============================================================
# 6. 预测 (Predict) —— 全量重训并生成提交文件
# ------------------------------------------------------------
# K 折只是用来「评估」结构/超参；确定方案后，要用「全部」训练数据
# 重新训练一个模型（数据越多越好），再对测试集预测。
# ============================================================
def train_and_pred(cfg: Config, train_features, train_labels,
                   test_features, test_ids):
    input_dim = train_features.shape[1]
    net = build_model(input_dim, cfg)
    # 用全部训练数据训练（无验证集）
    train(net, train_features, train_labels, None, None, cfg)

    device = cfg.device
    net.eval()
    with torch.no_grad():
        preds_log = net(test_features.to(device)).cpu().numpy().reshape(-1)
    # log 的逆变换还原成真实价格量纲（标签用的是 log，故用 exp）。
    preds = np.exp(preds_log)

    out_path = os.path.join(cfg.data_dir, "submission.csv")
    pd.DataFrame({"Id": test_ids, "SalePrice": preds}).to_csv(out_path, index=False)
    print(f"\n提交文件已保存：{out_path}")
    print(f"预测价格 min={preds.min():.0f}, max={preds.max():.0f}, mean={preds.mean():.0f}")


# ============================================================
# 主流程
# ============================================================
def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 56)
    print("House Prices 预测 - 工程化 PyTorch 实现")
    print("=" * 56)
    print(f"设备: {cfg.device} | 模型结构(hidden_dims): {cfg.hidden_dims}")

    # 1. 数据
    print("\n[1/3] 加载并预处理数据 ...")
    train_features, train_labels, test_features, test_ids = load_data(cfg)
    print(f"特征维度: {train_features.shape[1]} | "
          f"训练样本: {train_features.shape[0]} | 测试样本: {test_features.shape[0]}")

    # 2. K 折评估
    print(f"\n[2/3] {cfg.k_folds} 折交叉验证（评估模型泛化能力）...")
    avg_train, avg_val = k_fold(cfg, train_features, train_labels)
    print(f"=> 平均 train log-RMSE={avg_train:.4f} | "
          f"平均 valid log-RMSE={avg_val:.4f}")

    # 3. 全量训练 + 预测 + 提交
    print("\n[3/3] 用全部数据重训并生成提交文件 ...")
    train_and_pred(cfg, train_features, train_labels, test_features, test_ids)

    print("\n完成！")


if __name__ == "__main__":
    main()
