"""uv run demos/04-attention/main.py；固定向量演示，不训练语言模型。"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def attention(q, k, v, causal=False):
    scores = q @ k.T / math.sqrt(q.shape[-1])
    if causal:
        # 行是查询位置，列是被读取的位置；禁止列号大于行号。
        future = torch.ones(scores.shape, dtype=torch.bool).triu(diagonal=1)
        scores = scores.masked_fill(future, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return scores, weights, output


def main():
    torch.set_num_threads(2)
    # 三个位置的人工表示，不是真实词向量，不赋予坐标语言含义。
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    # 为减少变量，查询、键投影取单位矩阵；真实模型会学习投影参数。
    wq = torch.eye(2)
    wk = torch.eye(2)
    wv = torch.tensor([[10., 0.], [0., 20.]])
    q, k, v = x @ wq, x @ wk, x @ wv
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, (name, causal) in zip(axes, [('Full attention', False), ('Causal attention', True)]):
        scores, weights, output = attention(q, k, v, causal)
        results[name] = {'weights': weights.tolist(), 'output': output.tolist()}
        print(f'\n{name}\nweights:\n{weights}\noutput:\n{output}')
        ax.imshow(weights, vmin=0, vmax=1, cmap='Blues')
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f'{weights[row,col]:.3f}', ha='center', va='center',
                        color='white' if weights[row,col] > .6 else 'black')
        ax.set(title=name, xlabel='Key / value position', ylabel='Query position',
               xticks=[0,1,2], yticks=[0,1,2], xticklabels=['A','B','C'], yticklabels=['A','B','C'])
    folder = Path(__file__).parent/'outputs'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/'weights.png', dpi=160)
    plt.close(fig)
    (folder/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f'\nQ:\n{q}\nK:\n{k}\nV:\n{v}\nChart: {folder / "weights.png"}')


if __name__ == '__main__':
    main()
