"""运行：uv run demos/02-circle-classification/main.py"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


def make_data(count, generator):
    # x 的形状是 [样本数, 2]，每行是一点的两个坐标。
    x = torch.rand(count, 2, generator=generator) * 2 - 1
    # 圆内为 1，圆外为 0。半径 0.8，两类数量大致接近。
    y = (x.square().sum(dim=1) < 0.8**2).float().unsqueeze(1)
    return x, y


def train(model, x, y, steps=1500):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    model.train()
    for step in range(steps + 1):
        logits = model(x)
        loss = loss_fn(logits, y)
        history.append(loss.item())
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return history


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(42)
    train_x, train_y = make_data(600, generator)
    test_x, test_y = make_data(300, generator)
    # 重置种子，使两个多层模型的 Linear 参数初值完全相同。
    torch.manual_seed(7)
    linear = nn.Linear(2, 1)
    torch.manual_seed(7)
    network = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    torch.manual_seed(7)
    no_activation = nn.Sequential(nn.Linear(2, 16), nn.Identity(), nn.Linear(16, 1))
    models = [("Linear", linear), ("Hidden + Tanh", network),
              ("Hidden, no activation", no_activation)]

    axis = torch.linspace(-1, 1, 180)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    loss_fig, loss_ax = plt.subplots(figsize=(7, 4), layout="constrained")
    results = []
    for ax, (name, model) in zip(axes, models):
        history = train(model, train_x, train_y)
        model.eval()
        with torch.no_grad():
            train_accuracy = ((model(train_x) >= 0) == train_y.bool()).float().mean().item()
            test_logits = model(test_x)
            test_accuracy = ((test_logits >= 0) == test_y.bool()).float().mean().item()
            test_loss = nn.functional.binary_cross_entropy_with_logits(test_logits, test_y).item()
            probability = model(grid).sigmoid().reshape(gx.shape)
        result = {"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                  "train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                  "train_loss": history[-1], "test_loss": test_loss}
        results.append(result)
        print(f"{name:24s} params={result['parameters']:2d} "
              f"train={train_accuracy:.1%} test={test_accuracy:.1%}")
        ax.contourf(gx, gy, probability, levels=[0, 0.5, 1], colors=["#d8e9f7", "#ffdfbd"])
        if probability.min() < 0.5 < probability.max():
            ax.contour(gx, gy, probability, levels=[0.5], colors="#333333", linewidths=1)
        ax.scatter(test_x[:, 0], test_x[:, 1], c=test_y[:, 0], cmap="coolwarm",
                   vmin=0, vmax=1, s=10, edgecolors="white", linewidths=0.2)
        ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, linestyle="--", color="#555555"))
        ax.set(title=f"{name}\nTest accuracy: {test_accuracy:.1%}", xlabel="x1", ylabel="x2", aspect="equal")
        loss_ax.plot(history, label=name)
    loss_ax.set(xlabel="Completed updates", ylabel="Train binary cross-entropy", title="Same training data and SGD settings")
    loss_ax.legend()
    loss_ax.grid(alpha=0.2)
    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "boundaries.png", dpi=160)
    loss_fig.savefig(output / "loss.png", dpi=160)
    plt.close("all")
    summary = {"device": "cpu", "train_count": 600, "test_count": 300,
               "learning_rate": 0.1, "updates": 1500, "results": results,
               "elapsed_seconds": time.perf_counter() - started}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Elapsed: {summary['elapsed_seconds']:.2f}s; outputs: {output}")


if __name__ == "__main__":
    main()
