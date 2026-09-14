# 深度学习 Resources

## Knowledge

- [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)：2026-09-13 通过 Context7 读取官方文档示例，核对训练完整消息与生成前缀的 add_generation_prompt 用法；实际 Qwen tokenizer 检查答案边界与 EOS。
- [PEFT quicktour](https://huggingface.co/docs/peft/quicktour)：2026-09-13 读取官方 get_peft_model/LoraConfig、save_pretrained、PeftModel.from_pretrained 工作流，Demo 9 完成真实训练、重载和推理验证。
- [Qwen2.5-0.5B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)：2026-09-13 读取模型结构、语言支持与加载示例；Demo 9 固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`，这是已有指令微调的模型，继续做任务 SFT。

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：2026-09-13 核对安装版本官方文档，确认接收原始 logits 与类别 ID，不预先 softmax；用于 Demo 6 各位置的下一字训练损失，Context7 检索作为补充。

- [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)：2026-09-13 读取安装版本官方 docstring，核对整数索引查表、参数形状与初始化；Context7 检索作为补充。本节用人为固定表演示，不宣称已有语义学习。
- [Attention Is All You Need §3.5](https://arxiv.org/html/1706.03762v7#S3.SS5)：2026-09-13 已读取位置编码正文，核对与输入 embedding 同维相加、固定和可学习方案；入门课先用绝对位置查表，暂不展开正弦公式。

- [PyTorch：Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
  2026-09-08 已核对官方正文的 zero_grad、backward、step 顺序；用于噪声直线 demo 的自动求导训练循环。
- [PyTorch：Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
  后续阅读：requires_grad 与计算图；本轮未完整阅读此页。

- [《动手学深度学习》作者维护版：Introduction](https://d2l.ai/chapter_introduction/index.html)
  2026-09-07 已读取导言及 1.1、1.2 相关正文，并核对 1.2.2 对模型的计算定义。用于第一课的规则与例子，以及第二课的模型含义。邮件与客服为课程自拟示例，不是实际训练结果；按课选读小节，无需通读。
- [《动手学深度学习》作者维护版：Linear Regression](https://d2l.ai/chapter_linear-regression/linear-regression.html)
  用于后移的参数与损失实验。2026-09-07 已读取正文；入门概念完成后再阅读 3.1.1，暂不要求推导。实验使用无偏置的单参数模型，并省略教材损失中的 1/2 常数。
- [Vaswani et al.: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  Transformer 的原始论文，用于后续核对架构。2026-09-07 已读取摘要，目前不要求初学者阅读；授课涉及具体部件时需进一步核对正文。

## Wisdom (Communities)

尚未选定；当前先完成入门练习。

## Gaps

- Python 数值计算、自动求导及微调实践的具体资料，待课程推进时查阅官方文档并补充。

## Demo 2 补充来源

- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)：分类损失与 sigmoid 的组合，Context7 检索未命中精确条目，改为核对当前安装的 PyTorch 类文档。

- [PyTorch 保存与加载模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)：已核对保存最佳参数需 deepcopy 而非可变引用，用于 Demo 3 检查点。

- [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：2026-09-09 读取已安装版本官方 docstring，核对缩放、softmax 与 causal mask，手写 demo 与实际算子比较通过。
