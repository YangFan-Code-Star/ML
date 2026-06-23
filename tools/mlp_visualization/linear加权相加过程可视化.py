import os

import matplotlib.pyplot as plt
import torch


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))


def draw_network(ax, w_hidden, b_hidden, w_output, b_output):
    x_nodes = [("x1", 0.10, 0.15), ("x2", 0.35, 0.15), ("x3", 0.60, 0.15), ("x4", 0.85, 0.15)]
    h_nodes = [("h1", 0.18, 0.50), ("h2", 0.50, 0.50), ("h3", 0.82, 0.50)]
    o_nodes = [("o1", 0.25, 0.84), ("o2", 0.50, 0.84), ("o3", 0.75, 0.84)]

    for _, x0, y0 in x_nodes:
        for _, x1, y1 in h_nodes:
            ax.annotate(
                "",
                xy=(x1, y1 - 0.055),
                xytext=(x0, y0 + 0.055),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black", alpha=0.45),
            )

    for _, x0, y0 in h_nodes:
        for _, x1, y1 in o_nodes:
            ax.annotate(
                "",
                xy=(x1, y1 - 0.055),
                xytext=(x0, y0 + 0.055),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black", alpha=0.65),
            )

    def node(label, x, y, color="#9ed0f6"):
        circle = plt.Circle((x, y), 0.062, facecolor=color, edgecolor="black", linewidth=1.4, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=4)

    for label, x, y in x_nodes:
        node(label, x, y)
    for label, x, y in h_nodes:
        node(label, x, y, "#67b7dc")
    for label, x, y in o_nodes:
        node(label, x, y, "#b9ddfb")

    ax.text(0.50, 0.03, "输入层 x", ha="center", fontsize=11)
    ax.text(0.50, 0.39, "隐藏层 h = ReLU(W1 x + b1)", ha="center", fontsize=11)
    ax.text(0.50, 0.95, "输出层 o = W2 h + b2", ha="center", fontsize=11)
    ax.set_title("网络结构：输出层 Linear 把隐藏单元加权相加")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def main():
    output_dir = r"D:\Work\ML\d2l\code\outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 为了能画成函数曲线，这里只让 x1 变化，x2/x3/x4 固定。
    x1 = torch.linspace(-4, 4, 800)
    x2 = torch.full_like(x1, 0.8)
    x3 = torch.full_like(x1, -0.6)
    x4 = torch.full_like(x1, 1.2)
    x = torch.stack([x1, x2, x3, x4], dim=1)

    w_hidden = torch.tensor(
        [
            [1.2, -0.7, 0.5],
            [-0.4, 0.8, 0.3],
            [0.7, 0.2, -0.9],
            [0.2, -0.5, 0.6],
        ]
    )
    b_hidden = torch.tensor([0.4, -0.9, 0.2])

    w_output = torch.tensor(
        [
            [1.1, -0.8, 0.6],
            [-0.5, 1.2, 0.4],
            [0.7, 0.5, -1.1],
        ]
    )
    b_output = torch.tensor([0.1, -0.2, 0.3])

    z_hidden = x @ w_hidden + b_hidden
    h = relu(z_hidden)
    o = h @ w_output.T + b_output

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1])

    ax_net = fig.add_subplot(grid[:, 0])
    draw_network(ax_net, w_hidden, b_hidden, w_output, b_output)

    ax_h = fig.add_subplot(grid[0, 1])
    for i in range(h.shape[1]):
        ax_h.plot(x1.numpy(), h[:, i].numpy(), linewidth=2.0, label=f"h{i + 1}(x)")
    ax_h.set_title("隐藏层：每个 h_i 都是 ReLU 截断后的折线特征")
    ax_h.set_xlabel("变化的输入 x1")
    ax_h.set_ylabel("h_i")
    ax_h.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    ax_h.grid(alpha=0.25)
    ax_h.legend(ncol=3)

    ax_o = fig.add_subplot(grid[1, 1])
    colors = ["#d62728", "#2ca02c", "#9467bd"]
    for j in range(o.shape[1]):
        ax_o.plot(x1.numpy(), o[:, j].numpy(), linewidth=2.4, color=colors[j], label=f"o{j + 1}(x)")
    ax_o.set_title("输出层：Linear 将隐藏层特征加权相加")
    ax_o.set_xlabel("变化的输入 x1")
    ax_o.set_ylabel("o_j")
    ax_o.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    ax_o.grid(alpha=0.25)
    ax_o.legend(ncol=3)

    fig.suptitle("从隐藏层 h 到输出层 o：Linear 的加权相加过程", fontsize=16)
    plt.tight_layout()

    overview_path = os.path.join(output_dir, "linear_weighted_sum_overview.png")
    plt.savefig(overview_path, dpi=180)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for j, ax in enumerate(axes):
        total = torch.zeros_like(x1) + b_output[j]
        ax.plot(x1.numpy(), total.numpy(), "--", linewidth=1.4, color="gray", label=f"bias b{j + 1}")

        for i in range(h.shape[1]):
            contribution = h[:, i] * w_output[j, i]
            total = total + contribution
            ax.plot(
                x1.numpy(),
                contribution.numpy(),
                linewidth=1.7,
                alpha=0.78,
                label=f"w{j + 1}{i + 1} * h{i + 1}",
            )

        ax.plot(x1.numpy(), o[:, j].numpy(), linewidth=3.0, color="black", label=f"o{j + 1} = sum")
        ax.set_title(
            f"o{j + 1}(x) = "
            f"{w_output[j, 0]:.1f}h1 + {w_output[j, 1]:.1f}h2 + {w_output[j, 2]:.1f}h3 + {b_output[j]:.1f}"
        )
        ax.set_ylabel(f"o{j + 1}")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax.grid(alpha=0.25)
        ax.legend(ncol=5, fontsize=9)

    axes[-1].set_xlabel("变化的输入 x1")
    fig.suptitle("每个输出 o_j 由隐藏单元 h_i 的贡献叠加而成", fontsize=15)
    plt.tight_layout()

    detail_path = os.path.join(output_dir, "linear_weighted_sum_contributions.png")
    plt.savefig(detail_path, dpi=180)
    plt.close()

    print(f"saved: {overview_path}")
    print(f"saved: {detail_path}")


if __name__ == "__main__":
    main()
