"""uv run demos/03-overfitting/main.py --flip-count 12"""

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


def make_data(count, generator):
    x = torch.rand(count, 2, generator=generator) * 2 - 1
    y = (x.square().sum(1) < 0.8**2).float().unsqueeze(1)
    return x, y


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y).item()
        accuracy = ((logits >= 0) == y.bool()).float().mean().item()
    return {"loss": loss, "accuracy": accuracy}


def train(width, train_x, train_y, val_x, val_y):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(2, width), nn.Tanh(),
                          nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
    # Adam 同样根据梯度更新，但会自适应缩放步长；本课先保持它不变。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = []
    best_loss = float("inf")
    best_state = None
    patience = 40  # 连续 40 次验证检查未改善，就确定早停时刻。
    stale = 0
    stop_step = None
    best_step = 0
    for step in range(2001):
        if step % 10 == 0:
            training = evaluate(model, train_x, train_y)
            validation = evaluate(model, val_x, val_y)
            history.append({"step": step, "train_loss": training["loss"],
                            "val_loss": validation["loss"]})
            # 早停一旦触发，就冻结选定的状态；后续继续仅用于教学对照。
            if stop_step is None:
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    stop_step = step
        if step == 2000:
            break
        model.train()
        loss = nn.functional.binary_cross_entropy_with_logits(model(train_x), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    selected = copy.deepcopy(model)
    selected.load_state_dict(best_state)
    return model, selected, {"width": width, "parameters": sum(p.numel() for p in model.parameters()),
                             "best_step": best_step, "stop_step": stop_step,
                             "best_val_loss": best_loss, "history": history,
                             "final_train": evaluate(model, train_x, train_y),
                             "final_val": evaluate(model, val_x, val_y)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip-count", type=int, default=12, choices=range(0, 81), metavar="0..80")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(42)
    train_x, clean_y = make_data(80, generator)
    val_x, val_y = make_data(300, generator)
    train_y = clean_y.clone()
    flipped = torch.randperm(80, generator=torch.Generator().manual_seed(99))[:args.flip_count]
    train_y[flipped] = 1 - train_y[flipped]
    runs = [train(width, train_x, train_y, val_x, val_y) for width in (4, 64)]
    # 只依据验证损失选择网络和检查点，此前不生成或访问测试数据。
    winner = min(range(len(runs)), key=lambda i: runs[i][2]["best_val_loss"])
    test_x, test_y = make_data(1000, torch.Generator().manual_seed(2026))
    final_test = evaluate(runs[winner][1], test_x, test_y)
    for _, _, report in runs:
        print(f"width={report['width']} params={report['parameters']} best_step={report['best_step']} "
              f"stop_step={report['stop_step']} best_val={report['best_val_loss']:.4f} "
              f"final_train={report['final_train']['loss']:.4f} final_val={report['final_val']['loss']:.4f}")
    print(f"Selected width={runs[winner][2]['width']}; final test accuracy={final_test['accuracy']:.1%}")

    output = Path(__file__).parent / 'outputs' / f'flips-{args.flip_count}'
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, (_, _, report) in zip(axes, runs):
        h = report['history']
        ax.plot([r['step'] for r in h], [r['train_loss'] for r in h], label='Train (noisy labels)')
        ax.plot([r['step'] for r in h], [r['val_loss'] for r in h], label='Validation (clean labels)')
        ax.axvline(report['best_step'], color='green', linestyle='--', label='Saved checkpoint')
        if report['stop_step'] is not None:
            ax.axvline(report['stop_step'], color='gray', linestyle=':', label='Early stop trigger')
        ax.set(title=f"Hidden width {report['width']}", xlabel='Updates', ylabel='Binary cross-entropy')
        ax.legend(fontsize=8)
    fig.savefig(output/'loss.png', dpi=160)
    plt.close(fig)

    axis = torch.linspace(-1, 1, 160)
    gx, gy = torch.meshgrid(axis, axis, indexing='xy')
    grid = torch.stack([gx.flatten(), gy.flatten()], 1)
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout='constrained')
    for row, (last, selected, report) in zip(axes, runs):
        for ax, model, title in zip(row, (selected, last), ('Early checkpoint', 'After 2000 updates')):
            with torch.no_grad():
                probability = model(grid).sigmoid().reshape(gx.shape)
            ax.contourf(gx, gy, probability, levels=[0, .5, 1], colors=['#d8e9f7', '#ffdfbd'])
            ax.scatter(train_x[:,0], train_x[:,1], c=train_y[:,0], cmap='coolwarm', vmin=0, vmax=1, s=22)
            ax.scatter(train_x[flipped,0], train_x[flipped,1], facecolors='none', edgecolors='black', s=85, label='Flipped label')
            ax.add_patch(plt.Circle((0,0), .8, fill=False, linestyle='--', color='gray'))
            ax.set(title=f"Width {report['width']}: {title}", xlabel='x1', ylabel='x2', aspect='equal')
    fig.savefig(output/'boundaries.png', dpi=160)
    plt.close(fig)
    summary = {'flip_count': args.flip_count, 'train_count':80, 'val_count':300, 'test_count':1000,
               'runs':[r[2] for r in runs], 'selected_width':runs[winner][2]['width'],
               'final_test':final_test, 'seconds':time.perf_counter()-started}
    (output/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f"Elapsed {summary['seconds']:.2f}s; outputs: {output}")


if __name__ == '__main__':
    main()
