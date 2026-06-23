"""
4.1 多层感知机
===============

本文件对应 D2L 4.1「多层感知机」：
https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html

学习重点：
1. softmax 回归是单层线性模型：Flatten -> Linear(784, 10)。
2. MLP 在输入和输出之间加入隐藏层：Flatten -> Linear -> ReLU -> Linear。
3. 只堆叠 Linear 仍然等价于一个 Linear，关键区别是隐藏层后面的非线性激活函数。
4. ReLU、sigmoid、tanh 都是常见激活函数，但隐藏层中最常用的是 ReLU。
"""

import torch
from torch import nn
import torchvision
from torchvision import transforms
from d2l import torch as d2l


def load_data_fashion_mnist(batch_size, resize=None):
    """加载 Fashion-MNIST 数据集。

    Fashion-MNIST 每张图片形状是 1 x 28 x 28。
    对 MLP 来说，我们不直接利用图像的二维空间结构，后面会用 nn.Flatten
    把每张图片展开为长度 784 的向量。
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=0),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=0))


def init_weights(m):
    """初始化全连接层权重。

    D2L 这里常用均值为 0、标准差为 0.01 的正态分布初始化权重。
    偏置默认已经初始化为 0，因此这里不单独处理 bias。
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def explain_activation_functions():
    """用少量数值观察 3 个常见激活函数。

    激活函数的作用：给线性变换增加非线性。
    如果没有激活函数：
        X -> Linear -> Linear
    两个线性层可以合并成一个线性层，本质上仍然无法表达复杂模式。
    """
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("输入 x:")
    print(x)

    # ReLU(x) = max(x, 0)：负数截断为 0，正数原样通过。
    # 隐藏层中常优先使用 ReLU，因为它简单、梯度传播通常更稳定。
    print("\nReLU(x) = max(x, 0):")
    print(torch.relu(x))

    # sigmoid 把任意实数压缩到 (0, 1)，常用于二分类输出概率。
    # 但当输入绝对值很大时，梯度接近 0，隐藏层中容易训练变慢。
    print("\nsigmoid(x) in (0, 1):")
    print(torch.sigmoid(x))

    # tanh 把任意实数压缩到 (-1, 1)，以 0 为中心。
    # 它也会在输入绝对值很大时出现梯度很小的问题。
    print("\ntanh(x) in (-1, 1):")
    print(torch.tanh(x))


def build_mlp(num_inputs=784, num_hiddens=256, num_outputs=10):
    """构建一个单隐藏层 MLP。

    输入输出形状：
    - 原始图片:      (batch_size, 1, 28, 28)
    - Flatten 后:   (batch_size, 784)
    - 隐藏层 H:      (batch_size, 256)
    - 输出层 O:      (batch_size, 10)

    数学形式：
    H = ReLU(X W1 + b1)
    O = H W2 + b2

    注意：最后一层不写 softmax。
    nn.CrossEntropyLoss 会在内部结合 LogSoftmax 和 NLLLoss，
    因此模型直接输出未归一化的 logits 即可。
    """
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )
    net.apply(init_weights)
    return net


def print_shape_trace(net, train_iter):
    """打印一个小批量在 MLP 中的形状变化。

    形状是理解神经网络最重要的调试线索之一。
    这里不训练，只取一个 batch 走一遍前向传播。
    """
    X, y = next(iter(train_iter))
    print("\n一个 batch 的输入形状:", X.shape)
    print("一个 batch 的标签形状:", y.shape)

    hidden_input = net[0](X)
    hidden_linear = net[1](hidden_input)
    hidden_activation = net[2](hidden_linear)
    logits = net[3](hidden_activation)

    print("Flatten 后:", hidden_input.shape)
    print("第 1 个 Linear 后:", hidden_linear.shape)
    print("ReLU 后:", hidden_activation.shape)
    print("第 2 个 Linear 输出 logits:", logits.shape)


def main():
    batch_size = 256
    num_epochs = 10
    lr = 0.1

    explain_activation_functions()

    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    net = build_mlp()
    print_shape_trace(net, train_iter)

    # 多分类任务使用交叉熵损失。
    # 输入 logits 形状是 (batch_size, 10)，标签 y 形状是 (batch_size,)。
    loss = nn.CrossEntropyLoss()

    # SGD 会根据反向传播得到的梯度更新 W1、b1、W2、b2。
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # train_ch3 更贴近 D2L 线性网络和 MLP 章节的训练接口。
    # 如果你想使用 GPU 版本训练函数，也可以改回 train_ch6 + d2l.try_gpu()。
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == "__main__":
    main()
