# 深度学习学习工作区

目标：[亲手实现简化的 Transformer](MISSION.md)。

**当前：[Demo 2：从直线到弯曲边界](lessons/0005-circle-classification.html)** · [代码与操作说明](demos/02-circle-classification/README.md)

上一实验：[Demo 1：拟合带噪声的直线](lessons/0004-noisy-line-demo.html)

```bash
uv run demos/02-circle-classification/main.py
```

- [第一课：为什么需要机器学习？](lessons/0001-why-machine-learning.html)
- [第二课：模型到底是什么？](lessons/0002-what-is-a-model.html)
- [第三课：让程序尝试参数](lessons/0003-try-parameters.html) · [Python 实验](exercises/0001-try-parameters.py)
- [单例实验：平方误差梯度下降](exercises/0002-gradient-descent.py)
- [基础实验：多例子的全批量梯度下降](exercises/0003-batch-gradient-descent.py)
- [速查：写规则与从例子中学习](reference/learning-from-examples.html)
- [学习路线与当前进度](PROGRESS.md)
- [学习资料](RESOURCES.md)
- [初始基础](learning-records/0001-starting-point.md) · [已验证：输入、标签与预测](learning-records/0002-input-label-prediction.md)

HTML 课程可直接在浏览器中打开，无需安装依赖。Python 实验使用 uv 管理，默认 Python 3.12，已安装 PyTorch 与 Matplotlib。在仓库根目录运行 `uv run exercises/0003-batch-gradient-descent.py`，首次运行会自动准备 `.venv`。此前的枚举实验使用 `uv run exercises/0001-try-parameters.py`。每次只学一个核心问题，根据回答推进。数学与 Python 按需要补充，不按日期赶进度。

原参数实验保留在 [待拆分草稿](drafts/model-parameters.html)，其中参数与损失会拆开教授；它不再是第一课，也不计为已完成。

多例子程序执行 5 次更新，每轮先打印更新前的参数与均方误差，最后额外打印训练后的参数和新输入 4 的预测。新输入未参与训练，这一个人为构造的例子不代表全面验证泛化能力。
