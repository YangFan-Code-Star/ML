"""
================================================================================
4.4 模型选择、欠拟合和过拟合 (Model Selection, Underfitting and Overfitting)
参考教材：https://zh.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html

核心概念：
1. 训练误差 vs 泛化误差
   - 训练误差（training error）：模型在训练数据集上计算得到的误差
   - 泛化误差（generalization error）：模型应用在从原始样本分布中抽取的新数据时，误差的期望
   - 实际中我们无法准确计算泛化误差，只能通过独立的测试集来估计

2. 欠拟合（Underfitting）
   - 模型无法降低训练误差
   - 原因：模型过于简单（表达能力不足），无法捕获数据中的模式
   - 表现：训练损失和测试损失都很高

3. 过拟合（Overfitting）
   - 模型将训练数据拟合得比潜在真实分布更近
   - 原因：模型过于复杂或训练样本不足
   - 表现：训练损失明显低于测试损失，泛化能力差

4. 模型复杂度与数据集大小的关系
   - 给定训练数据集，复杂度过低 → 欠拟合
   - 给定训练数据集，复杂度过高 → 过拟合
   - 一般规则：训练数据越多，模型可以越复杂
================================================================================
"""

import math
import numpy as np
import torch
from torch import nn
import Pytorch_matplotlib  # 使用独立的绘图模块
from d2l import torch as d2l


# ================================================================================
# 生成数据集
# 我们使用一个已知的3阶多项式（真实模型）来生成数据，然后尝试用不同复杂度的模型拟合它
# 这样可以直观地观察欠拟合和过拟合现象
# 真实多项式：y = 5 + 1.2*x - 3.4*(x^2/2!) + 5.6*(x^3/3!) + 噪声
# ================================================================================

max_degree = 20  # 多项式的最大阶数（即特征维度，决定模型复杂度上限）
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间（只使用前4个非零值）
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)

# 将特征转换为多项式特征：features^k，然后除以 k! 防止数值爆炸
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!

# 生成标签：用真实权重与多项式特征相乘，加上噪声
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = Accumulator(2)  # 损失的总和, 样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), 0, y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400, title=None):
    """
    训练模型并绘制训练/测试损失曲线
    
    通过比较训练损失和测试损失，可以判断模型是否欠拟合或过拟合：
    - 两者都很高 → 欠拟合（模型太简单）
    - 训练低，测试高 → 过拟合（模型太复杂或数据不足）
    - 两者都低且接近 → 正常拟合（模型复杂度适中）
    """
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它（x^0 = 1 提供了偏置项）
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    # 使用 Pytorch_matplotlib.Animator（添加标题支持）
    # yscale='log' 使用对数刻度，便于观察训练初期和后期的损失变化
    animator = Pytorch_matplotlib.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                                 xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                                 legend=['train', 'test'], title=title)
    
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
    animator.show()  # 显示最终图像


if __name__ == '__main__':
    # ================================================================================
    # 实验1：正常拟合（使用与真实模型相同维度的特征）
    # 
    # 理论基础：模型复杂度与真实数据生成过程匹配时，模型能学到数据的本质规律。
    # 这里使用4维特征，与真实3阶多项式匹配（包含 x^0, x^1, x^2, x^3）。
    # 
    # 期望结果：训练损失和测试损失都较低且接近，说明模型泛化能力好。
    # 这对应于"Goldilocks Zone"（恰到好处区域），模型复杂度适中。
    # ================================================================================
    print("=== 正常拟合 (4维特征) ===")
    print("特征维度: 1, x, x^2/2!, x^3/3!")
    print("期望效果: 训练损失和测试损失都较低且接近")
    train(poly_features[:n_train, :4], poly_features[n_train:, :4],
          labels[:n_train], labels[n_train:], title="a")
    
    # ================================================================================
    # 实验2：欠拟合（使用比真实模型更简单的特征）
    # 
    # 理论基础：当模型复杂度过低时，模型的表达能力不足以拟合训练数据。
    # 这里只使用2维特征（仅 x^0, x^1），远低于真实模型的复杂度。
    # 
    # 期望结果：训练损失和测试损失都很高。
    # 这是因为模型无法捕捉数据的非线性模式（如 x^2, x^3 项）。
    # 类似一个学生只学了一点基础知识，无法解答复杂问题。
    # ================================================================================
    print("\n=== 欠拟合 (2维特征) ===")
    print("特征维度: 1, x")
    print("期望效果: 训练损失和测试损失都很高")
    train(poly_features[:n_train, :2], poly_features[n_train:, :2],
          labels[:n_train], labels[n_train:], title="b")
    
    # ================================================================================
    # 实验3：过拟合（使用比真实模型复杂得多的特征）
    # 
    # 理论基础：当模型复杂度过高而训练数据不足时，模型会过度拟合训练数据中的噪声，
    # 而不是学习到数据的本质规律。
    # 这里使用全部20维特征（x^0 到 x^19），但只有100个训练样本。
    # 模型参数（20个权重）相对于训练样本（100个）过多。
    # 
    # 期望结果：训练损失很低，但测试损失很高。
    # 这是因为模型为了降低训练误差，记住了训练数据中的噪声和细节，
    # 而不是学到泛化的模式。类似一个学生死记硬背考试答案，换一套题就不会了。
    # ================================================================================
    print("\n=== 过拟合 (20维特征) ===")
    print("特征维度: 所有20个维度")
    print("期望效果: 训练损失很低，但测试损失很高")
    train(poly_features[:n_train, :], poly_features[n_train:, :],
          labels[:n_train], labels[n_train:], title="c")
