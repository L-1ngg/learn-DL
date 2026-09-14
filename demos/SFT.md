# 从下一字预测到 SFT

目标：用真实 Python 代码，逐步理解 SFT 数据、损失、参数更新和微调前后对照。用户选定方向为中文问答与固定格式输出，当前从短问答和客服 JSON 分类开始。

**每次只学一个 demo。先运行 Demo 7，不需要先读完后三个模型文件。**

| 阶段 | 只回答的问题 | 运行条件 |
| --- | --- | --- |
| [07 · SFT labels](07-sft-labels/README.md) | 为什么 prompt 不计分、回答要计分？ | CPU，无下载、不训练 |
| [08 · 完整小型 SFT](08-sft-tiny/README.md) | 怎样从已有 checkpoint 继续训练？ | CPU，玩具起点、全参数更新 |
| [09 · Qwen LoRA](09-sft-lora/README.md) | 怎样给真实模型做低参数量微调？ | 首次约 1 GB 下载，推荐 CUDA |

```bash
uv run demos/07-sft-labels/main.py
uv run demos/08-sft-tiny/main.py
uv run --extra sft demos/09-sft-lora/main.py --stage inspect
uv run --extra sft demos/09-sft-lora/main.py --stage train
uv run --extra sft demos/09-sft-lora/main.py --stage infer --prompt '我想知道快递的配送进度。'
```

Demo 8/9 的 outputs 被 Git 忽略；脚本、数据、说明和依赖锁文件被版本管理。下载权重保留在 Hugging Face 缓存，不进入仓库。重新运行训练会覆盖默认目录，Demo 9 可通过 --output-dir 保留不同实验。

## 学会的标准

1. 能指出第一个回答 token 由哪个位置预测，区分 causal mask、padding mask 和 loss mask。
2. 能说明从随机初始化训练与从已有 checkpoint 微调的区别。
3. 能区分 SFT 任务和 LoRA 参数更新方法，说明 adapter 为什么离不开基座。
4. 能用同一组问题比较微调前后输出，并区分训练拟合、验证选模、测试覆盖。

前置尚不熟的概念随用随补，不要求先推导 Transformer 所有部件；助手运行通过也不等于用户已经掌握。

## 实现检查

先运行 Demo 9 inspect 缓存 tokenizer，再运行：

```bash
uv run --extra sft python -m unittest discover -s tests -v
```

检查回答边界、EOS、padding、prompt 的上下文梯度、未来信息隔离、HF 内部标签错位和数据 split 不重叠。测试不下载基座权重；tokenizer 必须已缓存。训练质量仍看各 demo 的实际对比，不能靠这些实现检查代替。

来源：[Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)、[PEFT quicktour](https://huggingface.co/docs/peft/quicktour)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
