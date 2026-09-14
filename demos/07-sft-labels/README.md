# Demo 7：SFT 到底训练哪一段？

```bash
uv run demos/07-sft-labels/main.py
```

只需要已有 PyTorch。没有模型下载、训练或输出文件。本段只学一个目标：能指出回答的第一个 token 由哪个位置预测，哪些目标计入 loss。

## 从 Demo 6 走过来

Demo 6 用一句文本中的每个下一个字做目标。SFT（Supervised Fine-Tuning，监督微调）通常从已有模型继续训练，用“问题/指令 → 期望回答”的示例教它完成任务。预测下一个 token 的原理没有变。

本课选用常见的 **answer-only loss**：问题作为上下文，但只对回答计算损失。SFT 也可以对完整序列计算损失，不要把 answer-only 当作 SFT 唯一定义；全参数更新还是 LoRA，则是另一条独立选择。

```text
prompt：问：猫吃什么？\n答：
answer：鱼。
完整输入：问：猫吃什么？\n答：鱼。<eos>
```

模型仍会读完整输入并使用 causal mask。本 demo 不创建模型，仅构造输入、标签和人工 logits，让损失计算可观察。

## 先只看这两行

```python
labels = ids.clone()
labels[:len(prompt)] = -100
```

labels 与输入最初逐位相同。问题部分的标签改为 `-100`，表示 CrossEntropyLoss 忽略这些目标；回答与 EOS 的 ID 保留。

然后才错开一位：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

所以“答：”最后的冒号所在位置，要预测第一个回答 token“鱼”，这项必须计算 loss。不是简单把 prompt 所在位置的所有 logits 都忽略。

## 三种 mask 不要混在一起

| 内容 | 做什么 |
| --- | --- |
| causal mask | 不许读取未来 token |
| padding 的 attention_mask | 不把补齐位置当成有效上下文 |
| labels 中的 -100 | 不对某些目标计入直接损失 |

这里为对齐两条不同长度样本使用右侧 padding；padding 的标签也必须是 -100。每条回答“鱼。”/“肉。”加 EOS 都有三个监督 token。

程序打印的零梯度指某些位置 **logits 的直接梯度**。真实模型仍通过回答 loss 使用 prompt 的表示，所以 prompt 对应 embedding/前文计算仍可能获得梯度；不要理解为 prompt 完全不参与学习。

## 改一个条件

把第一条回答从“鱼。”改为“鱼”，猜监督 token 数。运行后应从三个变成两个：鱼、EOS。EOS 是独立 token，教模型何时结束。

本 demo 使用字符级 tokenizer；真实 chat template 和子词边界在 [Demo 9](../09-sft-lora/README.md) 中再看。下一步：[Demo 8](../08-sft-tiny/README.md)。不懂某一行就发回对话，我们按当前节奏逐步解释。

来源：[PyTorch CrossEntropyLoss 的 ignore_index](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[Hugging Face causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)。
