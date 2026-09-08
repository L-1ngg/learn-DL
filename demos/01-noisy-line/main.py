"""运行：uv run demos/01-noisy-line/main.py"""

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def train(x, y, learning_rate, steps=100):
    # y_hat = w*x + b；每个实验都从相同的两个参数开始。
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=learning_rate)
    history = []
    for step in range(steps + 1):
        prediction = x * w + b
        loss = ((prediction - y) ** 2).mean()
        if not torch.isfinite(loss):
            break
        history.append(loss.item())
        if step == steps:
            break
        optimizer.zero_grad()  # 清掉上一轮梯度，避免意外累加。
        loss.backward()  # PyTorch 自动计算 w 和 b 的梯度。
        optimizer.step()  # 用当前梯度更新这两个参数。
    return w.detach(), b.detach(), history


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    # 人为构造已知规律，噪声模拟观测误差。无需下载数据。
    x = torch.rand(240) * 4 - 2
    y = 2 * x + 1 + torch.randn(240) * 0.4
    train_x, test_x = x[:180], x[180:]
    train_y, test_y = y[:180], y[180:]

    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    results = []
    # 对照实验只改变学习率；数据、初始参数和更新次数均一致。
    for label, rate in [("slow", 0.001), ("suitable", 0.05), ("too large", 1.0)]:
        w, b, history = train(train_x, train_y, rate)
        with torch.no_grad():
            test_loss = ((test_x * w + b - test_y) ** 2).mean().item()
        result = {
            "label": label, "learning_rate": rate,
            "w": w.item(), "b": b.item(),
            "initial_train_mse": history[0], "last_finite_train_mse": history[-1],
            "test_mse": test_loss if math.isfinite(test_loss) else None,
            "updates": len(history) - 1,
            "diverged": not math.isfinite(test_loss) or len(history) < 101,
        }
        results.append(result)
        axes[1].plot(history, label=f"lr={rate}")
        print(f"{label:10s} lr={rate:<5} w={w.item():.4g} b={b.item():.4g} "
              f"train MSE={history[-1]:.5g} test MSE={test_loss:.5g}")
        if label == "suitable":
            grid = torch.linspace(-2, 2, 100)
            axes[0].plot(grid, grid * w + b, color="#08786d", label="Learned line")

    axes[0].scatter(train_x, train_y, s=15, alpha=0.5, label="Train (180)")
    axes[0].scatter(test_x, test_y, s=25, marker="x", label="Test (60)")
    grid = torch.linspace(-2, 2, 100)
    axes[0].plot(grid, 2 * grid + 1, "k--", label="True noiseless relation")
    axes[0].set(xlabel="x", ylabel="y", title="One shared w and b")
    axes[1].set(xlabel="Completed updates", ylabel="Train MSE (log scale)",
                yscale="log", title="Same data, different learning rates")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "results.png", dpi=160)
    plt.close(fig)
    summary = {"seed": 42, "train_count": 180, "test_count": 60,
               "device": "cpu", "results": results,
               "elapsed_seconds": time.perf_counter() - started}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Elapsed: {summary['elapsed_seconds']:.2f}s; chart: {output / 'results.png'}")


if __name__ == "__main__":
    main()
