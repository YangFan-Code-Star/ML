import torch
from torch import nn
from d2l import torch as d2l
import Pytorch_matplotlib

# 故意构造高维小样本场景(20训练样本/200维特征)，容易产生过拟合
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = Pytorch_matplotlib.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
    animator.show() 

# 从零实现L2正则化：lambd=3表示使用权重衰减，可缓解过拟合
train(lambd=3)


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = Pytorch_matplotlib.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
    animator.show()     

# 简洁实现：wd=0表示未启用权重衰减，测试损失会明显高于训练损失，呈现过拟合
train_concise(0)


# ==================== 代码分析与总结 ====================
# 1. 整体功能
#    本代码演示L2正则化（权重衰减/Weight Decay）的两种实现方式：
#    - train(lambd)：从零手动构造惩罚项并更新参数
#    - train_concise(wd)：利用PyTorch优化器的weight_decay参数
#
# 2. 关键设计
#    - 数据：20个训练样本、200维特征 → 故意制造高维小样本，易过拟合
#    - 真实权重true_w=0.01，若模型学到的w范数过大，说明过拟合严重
#    - 简洁实现中对weight设置weight_decay，bias不设置，体现不同参数可有不同更新行为
#
# 3. 输出特征
#    - train(lambd=3)：有正则化，训练/测试损失差距小，泛化好
#    - train_concise(0)：无正则化，测试损失显著高于训练损失，明显过拟合


# ==================== 4.5.4 小结 ====================
# - 正则化是处理过拟合的常用方法：在训练集损失函数中加入惩罚项，降低模型复杂度
# - 保持模型简单的一个特别选择是使用L2惩罚的权重衰减
# - 权重衰减功能在深度学习框架的优化器中提供
# - 在同一训练代码实现中，不同参数集可以有不同的更新行为


# ==================== 4.5.5 练习回答 ====================
# 1. λ值实验
#    λ太小（如0）：训练误差低，测试误差高 → 过拟合
#    λ适中：训练误差略高，测试误差最低 → 最佳泛化
#    λ太大：训练误差和测试误差都高 → 欠拟合
#    应绘制训练/测试精度关于λ的曲线，观察测试误差的最低点
#
# 2. 验证集找到的最佳λ
#    验证集找到的是经验最优值，不一定是理论最优值；在工程上非常有关系，是实际调参的标准做法
#
# 3. L1正则化的更新方程
#    惩罚项为 Σ|w_i|，次梯度为sign(w)
#    w ← w - η * (∂ℓ/∂w + λ·sign(w))
#    L1会产生稀疏解（部分权重精确为0），具有特征选择效果
#
# 4. 矩阵范数
#    对于矩阵W，Frobenius范数满足 ||W||_F^2 = Σ w_ij^2 = tr(W^T W)
#    这与向量L2范数完全类似，是矩阵元素的平方和
#
# 5. 其他防止过拟合的方法
#    Dropout、数据增强(Data Augmentation)、早停(Early Stopping)、
#    批量归一化(Batch Normalization)、集成学习、参数共享/卷积约束等
#
# 6. 贝叶斯统计中的正则化
#    通过引入先验分布P(w)实现正则化：
#    - L2正则化 ↔ 高斯先验 P(w) ∝ exp(-λ||w||_2^2)
#    - L1正则化 ↔ 拉普拉斯先验 P(w) ∝ exp(-λ||w||_1)
#    MAP估计：argmax_w [log P(y|X,w) + log P(w)]，其中log P(w)即正则化项