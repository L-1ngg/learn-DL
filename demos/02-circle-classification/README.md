# Demo 2：小神经网络怎样画出弯曲的分类边界

```bash
uv run demos/02-circle-classification/main.py
```

CPU、本地生成数据、无需新依赖。600 个训练点，300 个独立测试点；每个点有两个坐标。圆内标签为 1，圆外为 0。这是学习模型表达能力的人工任务，实际已知圆规则时可以直接计算距离。

## 先看图

`outputs/boundaries.png` 中：点的颜色表示真实类别（红为 1、蓝为 0），背景表示预测类别（橙为 1、浅蓝为 0），灰色虚线为真实圆，黑色实线为模型的分类边界。直线模型此次把可视区域全部判为 0，因此图中没有黑色边界，不能把灰色虚线当作它学到的边界。

`outputs/loss.png` 比较训练损失；`outputs/metrics.json` 保存准确率和运行时间。首次基准 CPU 运行约 3.9 秒，不含 Python 导入时间。测试准确率：直线模型 48.3%，小网络 96.7%，去激活函数的网络 48.3%。数值来自固定种子的人工数据，不是通用性能结论。

## 对照的核心

```python
# 3 个参数：两个坐标权重和一个偏置。
nn.Linear(2, 1)

# 65 个参数：16*(2+1) + (16+1)。
nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))

# 同样 65 个参数，但去掉非线性变换。
nn.Sequential(nn.Linear(2, 16), nn.Identity(), nn.Linear(16, 1))
```

`Sequential` 按顺序执行；`Linear(2, 16)` 把每个点的两个坐标转换为 16 个数，称为隐藏层的表示；`Tanh` 分别对这 16 个数做非线性变换；最后一层将它们组合成一个分类分数。张量形状依次是 `[600, 2] → [600, 16] → [600, 16] → [600, 1]`。

多个带偏置的线性变换连续叠加，仍可合并成一个带偏置的线性变换，因此第三个模型仍无法形成包围圆内区域的边界。Tanh 不增加参数，但改变了可表达的函数。两个多层模型使用完全相同的初始 Linear 参数，同一数据、优化器、学习率和更新次数，区别只有激活函数。

## 输出与损失

这次预测类别，不再预测一个连续目标值。模型输出未经限制的分数 logit，正数判为 1，负数判为 0；代码将恰好为零归为 1。`sigmoid(logit)` 可以把它转成 0 到 1 的模型概率估计，0.5 阈值等价于 logit 的零阈值，不代表这个概率一定校准准确。

训练用 `BCEWithLogitsLoss`（二分类交叉熵），它已经结合 sigmoid，不要在输入这个损失函数之前再做 sigmoid。标签与输出都是 `[样本数, 1]`，默认取样本平均。准确率是判对的数量除以总数；训练优化的是可求导的交叉熵，不是直接优化硬分类后的准确率。

## 动手改一个条件

先把两个多层模型中的隐藏维度同时从 `16` 改成 `2`，每个模型的两处都改：`Linear(2, 2)` 与 `Linear(2, 1)`。保留种子和其余设置，比较弯曲边界与准确率变化。更少单元不保证在所有任务都更差，依据本次图和数据解释。每次运行覆盖 outputs，修改前记录基准结果。

先读模型定义，再读熟悉的 train 循环，最后看 make_data 和评估。绘图先跳过。疑问或修改后的结果发回对话；不需要一口气学会所有张量语法。

来源：[PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)、[Tanh](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html)、[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)。
