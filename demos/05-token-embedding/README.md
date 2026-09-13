# Demo 5：文字怎样变成带位置信息的向量

在项目根目录运行：

```bash
uv run demos/05-token-embedding/main.py
```

直接读 `main.py`，按打印顺序观察。不下载模型、不训练、不写输出文件。使用实际 `torch.nn.Embedding`，表中数值人工指定，便于手算；它们尚未学到语言含义。

## 先看查表

本例把每个汉字作为一个 token。词表规定 `我=0、爱=1、猫=2`，所以“我爱猫”的 token IDs 是 `[0,1,2]`。真实 tokenizer 不一定按字切分。

```python
token_vectors = token_embedding(token_ids)
```

这行按 ID 取出表中的对应行，得到 `[[1,0],[0,1],[1,1]]`。ID 是行号，不是重要性或语义距离。`from_pretrained` 在本例接收代码里创建的张量，并不下载预训练模型；`freeze=False` 表示表中参数允许训练更新，但本程序没有更新步骤。通常直接用 `torch.nn.Embedding(3, 2)` 会随机初始化三个二维向量。

## 再看位置

```python
position_ids = torch.arange(len(tokens))
position_vectors = position_embedding(position_ids)
x = token_vectors + position_vectors
```

三个位置编号始终是 `[0,1,2]`，与 token IDs 是两套编号。查位置表，再逐项相加：

```text
我爱猫：
我：[1,0] + [0.0,0.0] = [1.0,0.0]
爱：[0,1] + [0.1,0.2] = [0.1,1.2]
猫：[1,1] + [0.2,0.4] = [1.2,1.4]
```

两份向量维度相同才能这样相加；相加后 `x.shape` 仍是 `(3,2)`。这是可学习绝对位置 embedding 的形式，当前表值是示意，不是标准正弦位置编码；RoPE 等方案后续再学。

## 对照输出，然后只改一个条件

程序依次打印“我爱猫”“猫爱我”“我爱我”。比较“我”对应的行：token 向量始终为 `[1,0]`；在位置 0 和位置 2，相加后的结果分别是 `[1,0]`、`[1.2,0.4]`。

1. 把其中一个文本改成“猫猫我”。先猜 token IDs、position IDs、最终 x，再运行。
2. 把位置表最后一行的 `0.2` 改成 `0.8`。哪些输出坐标变化？

核对第 2 题：三个文本最后位置的输出第一维均增加 `0.6`；token 查表结果、其他位置及第二维不变。修改前先解释原因，不只比较终端数字。

示例词表只含“我、爱、猫”，位置表最多支持三个位置。学习扩充词表和序列长度之前，修改文本时先使用这个范围。

## 为什么还需要位置？

没有位置表示的 Full attention 不把“第几行”当作特征参与匹配，重排输入会使输出相应重排。位置表示让模型显式利用位置；能否学会语序关系仍取决于训练。causal mask 限制可读取范围，也引入方向约束，但与位置向量作用不同。

后面用这里的 x 计算 `q = x @ wq`、`k = x @ wk`、`v = x @ wv`，就能接上 [Demo 4](../04-attention/README.md)。不明白的代码直接发回对话，我们逐行看。

来源：[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)；[Attention Is All You Need §3.5](https://arxiv.org/html/1706.03762v7#S3.SS5)。本节已核对官方安装版本查表文档及论文位置编码正文。
