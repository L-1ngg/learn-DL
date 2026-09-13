# Demo 6：训练一个下一字预测器

```bash
uv run demos/06-next-token/main.py
```

在项目根目录运行。CPU 上从随机参数训练，不下载模型或数据、不保存文件。延续前两节，把 embedding、绝对位置向量、单头 causal attention 连到一个预测层。它还不是完整 Transformer：没有残差、LayerNorm 或前馈子层；先把预测与训练闭环接通。

## 这次只学一个新目标：预测下一个字

训练材料只有“猫爱吃鱼。”“狗爱吃肉。”，每个汉字或句号都是一个 token。

```text
原句：猫 爱 吃 鱼 。
输入：猫 爱 吃 鱼
答案：爱 吃 鱼 。
```

输入用 `text[:-1]`（去掉最后一个字），答案用 `text[1:]`（去掉第一个字）。每个位置预测它后面的字：

| 位置 | 可以读取的输入 | 要预测的字 |
| --- | --- | --- |
| 0 | 猫 | 爱 |
| 1 | 猫爱 | 吃 |
| 2 | 猫爱吃 | 鱼 |
| 3 | 猫爱吃鱼 | 。 |

训练时一次传入四个输入字，但 causal mask 阻止每个位置看见后面的答案。预测“鱼”的位置当前输入是“吃”，必须结合前文；另一句同样以“爱吃”接续，但目标为“肉”。这提供了使用上下文的训练需求，不代表注意力图本身能证明语义理解。

## 按这三个地方读代码

1. `examples`：看懂输入和答案怎样错开一位。
2. `NextTokenModel.forward`：文字编号经过 embedding → 相加位置向量 → Q/K/V → causal attention → 预测层。
3. `main` 的训练循环：预测 → CrossEntropyLoss → zero_grad → backward → step。

本例一条输入有四个位置、向量维度为 16、词表大小为 7：

```text
token IDs       (4,)
x / q / k / v   (4, 16)
weights         (4, 4)
mixed           (4, 16)
logits          (4, 7)
targets         (4,)
```

`nn.Linear` 负责可训练的线性变换，本例用它生成 Q/K/V。内部权重存储约定使其运算为 `x @ weight.T`，用途与上一节的投影相同。预测层把每个位置的 16 个数转换成词表中每个字的原始分数，叫 logits。

`CrossEntropyLoss` 用这些分数与正确字的 ID 计算损失；它接收 logits，不要提前 softmax。末尾为了展示概率才使用 softmax。注意这里有两个 softmax：attention 的 softmax 沿输入位置分配读取比例；预测概率的 softmax 沿词表候选字分配概率。

## 训练与生成有什么区别？

训练时每个位置读取真实前缀（teacher forcing），四个位置的损失都参与训练；`model.parameters()` 包含两张 embedding 表、Q/K/V 投影和预测层，它们一起更新。

生成时只给“猫”或“狗”，用最后位置的 logits 选最高分的字，追加到文本，再重复预测：

```text
猫 → 猫爱 → 猫爱吃 → 猫爱吃鱼 → 猫爱吃鱼。
```

`generate` 使用 `torch.no_grad()`，不计算梯度，也不更新参数；本例使用 argmax，不做随机采样。前缀须非空、来自当前词表，且长度不超过四个字；停止于句号或总长五个字，不实现无限续写。

## 观察什么？

- 训练前后的续写有何变化？训练 loss 是否下降？
- token embedding 的参数变化是否大于零？这次表中的数真的被更新了。
- 打印的 attention 权重右上角是否为零？
- “猫爱吃”的下一字预测，在词表上如何分配概率？

这里只检查拟合训练材料，没有独立验证/测试集，不据此判断泛化或语言理解。固定 seed 便于复现；不同 PyTorch/硬件版本数值和随机初始输出可能不同。

先尝试把训练句子中的“猫爱吃鱼。”改为“猫爱吃肉。”，重新运行后观察“猫”的续写。先预测变化，再运行；词表会自动从两句材料重新构建。

## 本次运行证据

2026-09-13，CPU、seed=7、Adam lr=0.005，更新 300 次：训练 loss 从 1.762534 降至 0.001283，分别续写出“猫爱吃鱼。”“狗爱吃肉。”；token embedding 最大绝对参数变化约 0.3206。

另行验证：修改未来输入不改变前面位置的 logits；单独输入前缀与完整输入的对应前缀 logits 一致；所有参数组都收到有限非零梯度。这检查了 causal mask 和训练链路，未检查泛化或其他设备。

来源：[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)、[scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。本次核对了安装版本的官方损失文档。前置：[Demo 5](../05-token-embedding/README.md)。遇到不清楚的一行，发回对话，我们小步拆解。
