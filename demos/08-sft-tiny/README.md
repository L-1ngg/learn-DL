# Demo 8：完整做一次小模型 SFT

```bash
uv run demos/08-sft-tiny/main.py
```

CPU 上完成，两阶段各更新 150 次；结果和模型写入本目录 `outputs/`。再次运行对应阶段会覆盖其输出。

本节只学：**加载已有参数 → 只对期望回答算 loss → 更新参数 → 保存 → 重载对比**。

## 为什么先建立一个玩具 checkpoint？

为避免初学时先下载大模型，第一阶段从随机参数训练一个很小的模型，保存为 `base.pt`；这只是便于观察的教学起点，不是经过大规模预训练的语言模型。真实公开模型在 Demo 9 使用。

基础阶段学习普通回答：

```text
问：猫吃什么？
答：鱼。
```

第二阶段从磁盘重新加载 `base.pt`，使用同样的问题，教它换成 JSON 回答：

```text
问：猫吃什么？
答：{"答案":"鱼"}
```

SFT 阶段把 prompt 标签改成 -100，只监督回答与 EOS；更新的是全部模型参数，所以这是**全参数微调**。训练 loss 的目标范围不同，基础阶段与 SFT 阶段的 loss 不可当作同一指标横向比较。

## 分阶段运行

```bash
uv run demos/08-sft-tiny/main.py --stage base
uv run demos/08-sft-tiny/main.py --stage sft
uv run demos/08-sft-tiny/main.py --stage compare
```

`sft` 需要先有 `base.pt`，`compare` 需要两个 checkpoint。`compare` 不训练，重新加载并对相同 prompt 生成回答。checkpoint 包含参数和词表，不包含优化器状态，不用于精确续训。

## 代码阅读顺序

1. `make_rows()`：同一个问题的原回答与期望回答。
2. `train()`：answer_only 为 True 时，什么标签被忽略？
3. `load()`/`save()`：微调从已有参数开始，保存后怎样恢复？
4. `generate()`：推理时只有问题，没有把正确回答偷偷喂进去。

`model.py` 提供小型 causal Transformer。它使用了残差、LayerNorm 和前馈层，当前作为训练载体使用，不要求你先学完架构才能观察 SFT。TransformerEncoderLayer 的名称不决定可见范围，本例显式传入 causal mask。

## 如何读结果

`outputs/comparison.json` 分别标明 `train` 和 `held_out`。训练问题输出正确 JSON，只说明拟合成功；未训练过的问法仍可能格式错误。本例只有八条训练样本，没有验证集、早停或可靠的泛化估计；固定训练步数，不依据 held_out 调整参数。

本次 CPU 运行中，“猫吃什么？”从“鱼。”变成 `{"答案":"鱼"}`，“狗吃什么？”从“肉。”变成 `{"答案":"肉"}`；未训练过的“猫吃？”输出了错误 JSON。保留这个结果，用来区分训练拟合与新输入上的表现。

## 动手

先改 `make_rows()` 中期望 JSON 的字段名，再重新运行全部阶段（词表也会随训练材料改变）。比较生成结果是否跟着改。不要把一个 checkpoint 的词表与另一个不同词表的模型混用。

前置：[Demo 7](../07-sft-labels/README.md)。下一步：[Demo 9 LoRA](../09-sft-lora/README.md)。有不清楚的部分先问，不用一次读完整个模型文件。

来源：[PyTorch 模型保存与加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
