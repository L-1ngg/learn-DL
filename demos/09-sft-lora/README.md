# Demo 9：给真实 Qwen 模型做 LoRA SFT

任务：把中文客服消息分类，并输出 `{"类别":"退款"}` 或 `{"类别":"物流"}`。

这次使用真正经过预训练和指令微调的 `Qwen/Qwen2.5-0.5B-Instruct`，在其基础上继续做任务 SFT。固定模型 revision，首次下载约 1 GB。依赖是项目的可选 `sft` extra，沿用 uv 和现有 PyTorch。

## 按顺序运行

先只下载 tokenizer、看一条样本的模板和标签：

```bash
uv run --extra sft demos/09-sft-lora/main.py --stage inspect
```

再做训练：

```bash
uv run --extra sft demos/09-sft-lora/main.py --stage train
```

默认有 CUDA 就用 CUDA，否则用 CPU。已检测到本机 RTX 4060 Laptop 8 GB；短序列、单样本 batch 的这个 0.5B LoRA 示例使用 BF16（GPU 支持时），不需要量化。CPU 使用 float32，会慢很多。`--device cpu`/`--device cuda` 可明确选择设备。

最后，从保存产物单独加载推理：

```bash
uv run --extra sft demos/09-sft-lora/main.py --stage infer --prompt '我的包裹现在到哪里了？'
```

默认输出到本目录 `outputs/`；重新训练会覆盖这里的最佳 adapter 和报告。对照实验用 `--output-dir demos/09-sft-lora/outputs/experiment-2` 保留两次结果，推理时也传入对应目录。`--epochs` 默认 6。

## SFT 和 LoRA 是两个概念

SFT 说明训练任务：从示范回答学习。LoRA 说明更新参数的方法：冻结原始权重 W，只训练额外的小矩阵，产生增量 ΔW。前向计算使用 W + ΔW。

本例 `r=8`，只在 Q/V 投影上加 LoRA。`lora_alpha=16` 控制缩放，`lora_dropout=0` 便于观察；`print_trainable_parameters()` 打印实际可训练参数比例。optimizer 只拿到 LoRA 参数，保存的 adapter 也只包含增量，不是完整基座。推理必须同时加载相同 revision 的基座与 adapter。

因此：Demo 8 是全参数 SFT；Demo 9 是 LoRA SFT。LoRA 不是另一种“答案格式”，也不意味着训练不用 GPU 内存。

## 先读 data.py

每条数据包含 `user` 和示范 `assistant`。训练 16 条、验证 4 条、测试 4 条，事先分开：

- `TRAIN` 用于参数更新。
- `VALIDATION` 只算回答 loss，用于选择六个已训练 epoch 中的最佳 adapter。
- `TEST` 只用于最终前后对照，不用于梯度或选 checkpoint。

system prompt、用户问题、解码方式在微调前后相同。测试报告分别检查输出能否解析为期望 JSON 对象及类别是否正确，不把训练 loss 当作业务质量。这里的四条测试只是流程演示，覆盖面远不足以证明业务可靠性。

数据使用两类明确请求，没有其他类别、多意图、拒答或多轮对话。修改业务任务时需要重新设计标签与数据覆盖。

## 标签怎样准备？

使用该模型自己的 `apply_chat_template`，不手写特殊 token、不按字符数量估算答案边界。完整训练对话 `add_generation_prompt=False`；只有问题的生成前缀 `add_generation_prompt=True`。

代码确认生成前缀的 token IDs 恰好是训练序列的前缀，把这部分 labels 设成 -100，保留 assistant 回答、结束标记与模板尾部。超过 256 tokens 的样本会报错，不会静默截断到只剩 prompt。

**和 Demo 7/8 的重要差别：Hugging Face causal LM 内部已经移动一位。** 传入的 labels 必须与 input_ids 等长，不要再次手动错位。

本例一条样本一个 batch，没有 padding；attention_mask 全为 1。梯度累积 4 条样本才更新一次参数。改成有 padding 的 batch 时，padding labels 同样要设 -100。

## 再读 main.py

按 `load_base → get_peft_model → loss.backward → optimizer.step → save_pretrained` 阅读。默认每 epoch 16 次前向/反向、4 次参数更新，六个 epoch 总共 24 次参数更新；epoch、样本次数与更新次数要区分。

训练结束后释放模型，再通过 `PeftModel.from_pretrained` 加载基座和磁盘中的最佳 adapter，然后生成对比结果。这验证了保存/加载链路。每次运行保存 `outputs/results.json`，包含固定 revision、最佳 epoch、loss、训练/测试输出、运行耗时和峰值显存。

`model.train()` 和 `model.eval()` 切换运行模式；是否更新参数还取决于反向传播和 optimizer。生成使用 `torch.no_grad()`，不更新参数。

## 做一个小实验

先读原始结果，再把训练条数减半，用新输出目录重新训练。固定验证/测试样本，不把失败测试逐条搬进训练集。观察训练 loss、验证 loss、格式正确性是否一起改善；它们可能不一致。

## 本机实际运行证据

2026-09-13，RTX 4060 Laptop 8 GB，PyTorch 2.14.0+cu130、Transformers 4.57.6、PEFT 0.18.1，默认六个 epoch：

- 可训练参数 540,672，占包装后总参数约 0.1093%；检查了一块原始 Q 投影权重保持不变。
- 回答验证 loss 从 4.0643 降至 0.001323，选择第六个 epoch。
- 固定四条 test 的完整 JSON 对象/类别正确数从 0/4 到 4/4；未根据这些测试结果调整训练设置。
- 默认训练命令约 26.6 秒，含已缓存基座的加载、前后生成与 adapter 重载，不含首次下载。PyTorch 峰值 allocated 显存约 1.06 GiB，不代表整机总 GPU 占用。
- 独立运行 infer，输入“我想知道快递的配送进度。”得到 `{"类别":"物流"}`。

微调前 system prompt 没有指定 JSON schema，微调后的 schema 来自训练示范；这个对照展示示范学习，不是与精心设计的零样本提示词比较，也不说明这项简单任务必须微调。真实任务应另外比较合适的 prompt-only 基线。

前置：[Demo 8](../08-sft-tiny/README.md)。先把 inspect 的输出发给我也可以，我们从一条真实样本开始解释。

来源：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)、[PEFT quicktour](https://huggingface.co/docs/peft/quicktour)、[LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)。本次已核对官方模型卡及对应 API 文档。
