import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from d2l import torch as d2l
except ImportError:
    d2l = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


FEATURE_NAMES = [
    "intensity_main",
    "light_count",
    "ambient_intensity",
    "main_height",
    "main_angle",
    "fill_ratio",
]


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def synthetic_lighting_data(num_examples):
    """生成灯光属性与中值灰度标签。"""
    intensity_main = torch.rand(num_examples, 1) * 3.5 + 0.5
    light_count = torch.rand(num_examples, 1) * 7.0 + 1.0
    ambient_intensity = torch.rand(num_examples, 1) * 1.2
    main_height = torch.rand(num_examples, 1) * 8.0 + 1.0
    main_angle = torch.rand(num_examples, 1) * 80.0 + 10.0
    fill_ratio = torch.rand(num_examples, 1) * 0.9

    angle_rad = main_angle * math.pi / 180.0
    angle_factor = torch.cos(angle_rad).clamp(min=0.0)
    height_decay = 1.0 / (1.0 + 0.18 * torch.square(main_height - 2.5))
    count_gain = torch.log1p(light_count)
    interaction = intensity_main * fill_ratio

    gray = (
        0.08
        + 0.26 * torch.sigmoid(1.15 * intensity_main)
        + 0.18 * torch.sigmoid(2.8 * ambient_intensity)
        + 0.14 * count_gain / math.log(9.0)
        + 0.18 * angle_factor
        + 0.10 * height_decay
        + 0.10 * torch.sigmoid(1.35 * interaction)
    )
    gray += 0.02 * torch.sin(intensity_main * 1.4 + fill_ratio * 2.2)
    gray += torch.normal(0, 0.01, gray.shape)
    gray = gray.clamp(0.0, 1.0)

    features = torch.cat(
        [
            intensity_main,
            light_count,
            ambient_intensity,
            main_height,
            main_angle,
            fill_ratio,
        ],
        dim=1,
    )
    return features, gray.reshape((-1, 1))


def _load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def load_data(batch_size, test_ratio=0.2, num_examples=5000):
    features, labels = synthetic_lighting_data(num_examples)
    num_test = int(num_examples * test_ratio)
    indices = torch.randperm(num_examples)
    test_idx = indices[:num_test]
    train_idx = indices[num_test:]

    train_features = features[train_idx]
    test_features = features[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std

    if d2l is not None:
        train_iter = d2l.load_array((train_features, train_labels), batch_size)
        test_iter = d2l.load_array((test_features, test_labels), batch_size, is_train=False)
    else:
        train_iter = _load_array((train_features, train_labels), batch_size, is_train=True)
        test_iter = _load_array((test_features, test_labels), batch_size, is_train=False)

    stats = {
        "train_mean": mean,
        "train_std": std,
        "raw_features": features,
        "raw_labels": labels,
        "train_indices": train_idx,
        "test_indices": test_idx,
    }
    return train_iter, test_iter, train_features, test_features, train_labels, test_labels, stats


def get_net():
    return nn.Sequential(
        nn.Linear(6, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def evaluate(net, data_iter):
    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")
    net.eval()
    mse_total = 0.0
    mae_total = 0.0
    count = 0
    with torch.no_grad():
        for X, y in data_iter:
            pred = net(X)
            mse_total += mse_loss(pred, y).item()
            mae_total += mae_loss(pred, y).item()
            count += y.numel()
    return {"mse": mse_total / count, "mae": mae_total / count}


def train(net, train_iter, test_iter, num_epochs, lr):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {
        "epoch": [],
        "train_mse": [],
        "test_mse": [],
        "train_mae": [],
        "test_mae": [],
    }

    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_metrics = evaluate(net, train_iter)
        test_metrics = evaluate(net, test_iter)
        history["epoch"].append(epoch + 1)
        history["train_mse"].append(train_metrics["mse"])
        history["test_mse"].append(test_metrics["mse"])
        history["train_mae"].append(train_metrics["mae"])
        history["test_mae"].append(test_metrics["mae"])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"epoch {epoch + 1:03d} | "
                f"train_mse {train_metrics['mse']:.6f} | "
                f"test_mse {test_metrics['mse']:.6f} | "
                f"test_mae {test_metrics['mae']:.6f}"
            )

    return history


def predict(net, features):
    net.eval()
    with torch.no_grad():
        return net(features)


def print_section(title):
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def summarize_samples(raw_features, raw_labels, indices, num_samples=5):
    print("样本灯光参数与中值灰度示例：")
    for idx in indices[:num_samples]:
        feature_values = ", ".join(
            f"{name}={raw_features[idx, i].item():.3f}" for i, name in enumerate(FEATURE_NAMES)
        )
        print(f"  [{idx.item():04d}] {feature_values}, gray={raw_labels[idx].item():.3f}")


def print_prediction_table(test_features, test_labels, predictions, num_rows=8):
    print("\n测试样本真实值与预测值对比：")
    print("  index | true_gray | pred_gray | abs_error")
    errors = (predictions - test_labels).abs().reshape(-1)
    for i in range(min(num_rows, len(test_labels))):
        print(
            f"  {i:5d} | "
            f"{test_labels[i].item():9.4f} | "
            f"{predictions[i].item():9.4f} | "
            f"{errors[i].item():9.4f}"
        )


def save_figures(output_dir, stats, history, test_labels, predictions):
    if plt is None:
        print("\n未检测到 matplotlib，跳过图像保存。")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    raw_features = stats["raw_features"]
    raw_labels = stats["raw_labels"].reshape(-1)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(raw_features[:, 0].numpy(), raw_labels.numpy(), s=6, alpha=0.5)
    plt.xlabel("intensity_main")
    plt.ylabel("median_gray")
    plt.title("主光强度与中值灰度")
    plt.subplot(1, 2, 2)
    plt.scatter(raw_features[:, 1].numpy(), raw_labels.numpy(), s=6, alpha=0.5)
    plt.xlabel("light_count")
    plt.ylabel("median_gray")
    plt.title("灯光数量与中值灰度")
    scatter_path = output_dir / "feature_scatter.png"
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=160)
    plt.close()
    saved_files.append(scatter_path)

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_mse"], label="train_mse")
    plt.plot(history["epoch"], history["test_mse"], label="test_mse")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title("训练与测试损失")
    plt.legend()
    loss_path = output_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()
    saved_files.append(loss_path)

    plt.figure(figsize=(5, 5))
    plt.scatter(test_labels.reshape(-1).numpy(), predictions.reshape(-1).numpy(), s=8, alpha=0.6)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("true_gray")
    plt.ylabel("pred_gray")
    plt.title("预测值与真实值对比")
    pred_path = output_dir / "prediction_scatter.png"
    plt.tight_layout()
    plt.savefig(pred_path, dpi=160)
    plt.close()
    saved_files.append(pred_path)

    return saved_files


def run_experiment():
    set_seed(42)
    batch_size = 64
    num_epochs = 60
    lr = 0.01
    output_dir = Path(__file__).resolve().parent / "outputs"

    print_section("1. 问题背景与任务定义")
    print("任务：使用多层感知机根据灯光属性预测渲染结果的中值灰度。")
    print("标签定义：将渲染图像转换为灰度图后，取所有像素的中位数作为照度近似。")

    print_section("2. 生成合成数据集")
    train_iter, test_iter, train_features, test_features, train_labels, test_labels, stats = load_data(
        batch_size=batch_size,
        test_ratio=0.2,
        num_examples=5000,
    )
    print(f"特征张量形状：train={tuple(train_features.shape)}, test={tuple(test_features.shape)}")
    print(f"标签张量形状：train={tuple(train_labels.shape)}, test={tuple(test_labels.shape)}")
    print(f"标签范围：min={stats['raw_labels'].min().item():.4f}, max={stats['raw_labels'].max().item():.4f}")
    summarize_samples(stats["raw_features"], stats["raw_labels"], stats["train_indices"])

    print_section("3. 读取数据与小批量迭代")
    for X, y in train_iter:
        print(f"一个小批量的形状：X={tuple(X.shape)}, y={tuple(y.shape)}")
        break

    print_section("4. 定义多层感知机")
    net = get_net()
    print(net)

    print_section("5. 定义损失函数与优化器")
    print("损失函数：MSELoss")
    print(f"优化器：Adam(lr={lr})")

    print_section("6. 训练模型")
    history = train(net, train_iter, test_iter, num_epochs=num_epochs, lr=lr)

    print_section("7. 可视化训练误差与预测效果")
    predictions = predict(net, test_features)
    metrics = evaluate(net, test_iter)
    baseline_mse = torch.var(test_labels, unbiased=False).item()
    print(f"测试集 MSE：{metrics['mse']:.6f}")
    print(f"测试集 MAE：{metrics['mae']:.6f}")
    print(f"标签方差基线：{baseline_mse:.6f}")
    print_prediction_table(test_features, test_labels, predictions)
    saved_files = save_figures(output_dir, stats, history, test_labels, predictions)
    if saved_files:
        print("\n已保存图像：")
        for path in saved_files:
            print(f"  {path}")

    print_section("8. 小结")
    print("多层感知机能够捕捉主光强度、灯光数量、角度衰减和交互项带来的非线性关系。")
    print("在该合成数据上，测试误差应明显低于标签方差基线，说明模型学到了有效映射。")

    print_section("9. 练习")
    print("1. 将模型改成线性回归，对比测试集 MSE，分析为什么误差更大。")
    print("2. 额外加入颜色温度或阴影强度特征，观察是否提升预测效果。")
    print("3. 调整隐藏层宽度与层数，比较模型复杂度对收敛速度和泛化误差的影响。")


if __name__ == "__main__":
    run_experiment()
