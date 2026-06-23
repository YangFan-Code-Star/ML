import os

import matplotlib.pyplot as plt
import torch


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))


def main():
    output_dir = r"D:\Work\ML\d2l\code\outputs"
    os.makedirs(output_dir, exist_ok=True)

    x = torch.linspace(-4, 4, 900).reshape(-1, 1)

    # 第一层：4个线性函数。每个神经元都是一条直线。
    w1 = torch.tensor([[1.0, -1.2, 0.8, -0.7]])
    b1 = torch.tensor([1.7, 1.4, -0.5, -1.8])
    z1 = x @ w1 + b1
    h1 = relu(z1)

    # 第二层：把第一层的4个ReLU折线重新线性组合成3条新曲线。
    w2 = torch.tensor(
        [
            [1.0, -0.7, 0.4],
            [-0.6, 1.0, 0.5],
            [0.8, 0.5, -1.1],
            [-0.5, 0.7, 0.9],
        ]
    )
    b2 = torch.tensor([-0.8, 0.2, -0.4])
    z2 = h1 @ w2 + b2
    h2 = relu(z2)

    # 输出层：再把第二层激活后的折线组合成最终输出。
    w3 = torch.tensor([[1.2], [-0.8], [0.9]])
    b3 = torch.tensor([0.1])
    y = h2 @ w3 + b3

    x_np = x.squeeze().numpy()

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for i in range(z1.shape[1]):
        axes[0, 0].plot(x_np, z1[:, i].numpy(), linewidth=1.8, label=f"z1[{i}]")
    axes[0, 0].set_title("1. 第一层线性变换：Linear1(x)")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("z1")
    axes[0, 0].axhline(0, color="black", linewidth=0.8, alpha=0.35)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(ncol=2)

    for i in range(h1.shape[1]):
        axes[0, 1].plot(x_np, h1[:, i].numpy(), linewidth=1.8, label=f"h1[{i}]")
    axes[0, 1].set_title("2. 第一层激活：ReLU(Linear1(x))")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("h1")
    axes[0, 1].axhline(0, color="black", linewidth=0.8, alpha=0.35)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(ncol=2)

    for i in range(z2.shape[1]):
        axes[1, 0].plot(x_np, z2[:, i].numpy(), linewidth=1.8, label=f"z2[{i}]")
    axes[1, 0].set_title("3. 第二层线性组合：Linear2(h1)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("z2")
    axes[1, 0].axhline(0, color="black", linewidth=0.8, alpha=0.35)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(ncol=2)

    for i in range(h2.shape[1]):
        axes[1, 1].plot(x_np, h2[:, i].numpy(), linewidth=1.7, alpha=0.75, label=f"h2[{i}]")
    axes[1, 1].plot(x_np, y.squeeze().numpy(), linewidth=3.0, color="black", label="最终输出 y")
    axes[1, 1].set_title("4. 第二层激活和最终输出")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("h2 / y")
    axes[1, 1].axhline(0, color="black", linewidth=0.8, alpha=0.35)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(ncol=2)

    fig.suptitle("Linear -> ReLU -> Linear -> ReLU 如何增加表达能力", fontsize=15)
    plt.tight_layout()

    process_path = os.path.join(output_dir, "relu_network_expressivity_process.png")
    plt.savefig(process_path, dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for i in range(h2.shape[1]):
        contribution = h2[:, i] * w3[i, 0]
        plt.plot(x_np, contribution.numpy(), linewidth=1.8, alpha=0.8, label=f"h2[{i}] 对输出的贡献")
    plt.plot(x_np, y.squeeze().numpy(), linewidth=3.0, color="black", label="最终输出：加权求和")
    plt.title("多个ReLU折线特征叠加成更复杂的分段线性函数")
    plt.xlabel("x")
    plt.ylabel("输出值")
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()

    combine_path = os.path.join(output_dir, "relu_network_expressivity_combination.png")
    plt.savefig(combine_path, dpi=180)
    plt.close()

    print(f"saved: {process_path}")
    print(f"saved: {combine_path}")


if __name__ == "__main__":
    main()
