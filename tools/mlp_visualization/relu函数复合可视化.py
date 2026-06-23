import os

import matplotlib.pyplot as plt
import torch


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))


def main():
    output_dir = r"D:\Work\ML\d2l\code\outputs"
    os.makedirs(output_dir, exist_ok=True)

    x = torch.linspace(-5, 5, 800)
    fx = relu(x)
    gfx = relu(fx)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x.numpy(), fx.numpy(), linewidth=2.5, color="#1f77b4")
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].axvline(0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].set_title("第一步：f(x) = ReLU(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x.numpy(), gfx.numpy(), linewidth=2.5, color="#d62728")
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.35)
    axes[1].axvline(0, color="black", linewidth=0.8, alpha=0.35)
    axes[1].set_title("第二步：g(f(x)) = ReLU(ReLU(x))")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("g(f(x))")
    axes[1].grid(alpha=0.25)

    fig.suptitle("ReLU 函数复合：x -> ReLU(x) -> ReLU(ReLU(x))", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "relu_composition.png")
    plt.savefig(save_path, dpi=180)
    plt.close()

    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
