import os

import matplotlib.pyplot as plt
import torch
from torch import nn


torch.manual_seed(7)


class CurveMLP(nn.Module):
    """Two-hidden-layer MLP for visualizing 1D curve fitting."""

    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 32)
        self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(32, 16)
        self.act2 = nn.Tanh()
        self.output = nn.Linear(16, 1)

    def forward(self, x, return_second_hidden=False):
        h1 = self.act1(self.hidden1(x))
        h2 = self.act2(self.hidden2(h1))
        y_hat = self.output(h2)
        if return_second_hidden:
            return y_hat, h2
        return y_hat


def target_function(x):
    return torch.sin(2.5 * x) + 0.35 * torch.cos(5.0 * x) + 0.15 * x**2


def main():
    output_dir = r"D:\Work\ML\d2l\code\outputs"
    os.makedirs(output_dir, exist_ok=True)

    x_train = torch.linspace(-3, 3, 160).reshape(-1, 1)
    y_train = target_function(x_train)
    y_train_noisy = y_train + 0.08 * torch.randn_like(y_train)

    x_plot = torch.linspace(-3.2, 3.2, 500).reshape(-1, 1)
    y_plot = target_function(x_plot)

    net = CurveMLP()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    snapshots = {}
    snapshot_epochs = {0, 50, 200, 800, 2000}

    with torch.no_grad():
        snapshots[0] = net(x_plot).detach().clone()

    for epoch in range(1, 2001):
        optimizer.zero_grad()
        y_hat = net(x_train)
        loss = loss_fn(y_hat, y_train_noisy)
        loss.backward()
        optimizer.step()

        if epoch in snapshot_epochs:
            with torch.no_grad():
                snapshots[epoch] = net(x_plot).detach().clone()

    with torch.no_grad():
        final_y_hat, second_hidden = net(x_plot, return_second_hidden=True)

    x_np = x_plot.squeeze().numpy()
    y_np = y_plot.squeeze().numpy()
    final_np = final_y_hat.squeeze().numpy()
    train_x_np = x_train.squeeze().numpy()
    train_y_np = y_train_noisy.squeeze().numpy()

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(11, 7))
    plt.scatter(train_x_np, train_y_np, s=12, alpha=0.45, label="带噪声训练点")
    plt.plot(x_np, y_np, color="black", linewidth=2.2, label="真实目标曲线 y")
    for epoch in sorted(snapshots):
        plt.plot(
            x_np,
            snapshots[epoch].squeeze().numpy(),
            linewidth=1.6,
            alpha=0.9,
            label=f"MLP输出 y_hat: epoch {epoch}",
        )
    plt.title("多层感知机拟合一维非线性曲线")
    plt.xlabel("输入 x")
    plt.ylabel("输出 y")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fit_path = os.path.join(output_dir, "mlp_curve_fit_progress.png")
    plt.savefig(fit_path, dpi=180)
    plt.close()

    plt.figure(figsize=(11, 7))
    hidden_np = second_hidden.numpy()
    for unit_index in range(8):
        plt.plot(
            x_np,
            hidden_np[:, unit_index],
            linewidth=1.8,
            label=f"第2隐藏层激活 h2[{unit_index}]",
        )
    plt.title("第二隐藏层经过激活函数后的输出曲线")
    plt.xlabel("输入 x")
    plt.ylabel("激活输出")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    hidden_path = os.path.join(output_dir, "mlp_second_layer_activations.png")
    plt.savefig(hidden_path, dpi=180)
    plt.close()

    plt.figure(figsize=(11, 7))
    plt.plot(x_np, y_np, color="black", linewidth=2.4, label="真实目标曲线 y")
    plt.plot(x_np, final_np, color="#d62728", linewidth=2.4, label="训练后 MLP 输出 y_hat")
    plt.scatter(train_x_np, train_y_np, s=12, alpha=0.35, label="带噪声训练点")
    plt.title("训练完成后的曲线拟合效果")
    plt.xlabel("输入 x")
    plt.ylabel("输出 y")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    final_path = os.path.join(output_dir, "mlp_final_fit.png")
    plt.savefig(final_path, dpi=180)
    plt.close()

    print(f"final loss: {loss.item():.6f}")
    print(f"saved: {fit_path}")
    print(f"saved: {hidden_path}")
    print(f"saved: {final_path}")


if __name__ == "__main__":
    main()
