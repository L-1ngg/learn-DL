# 深度学习 Resources

## Knowledge

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
