"""uv run demos/05-token-embedding/main.py；CPU 查表演示，不训练。"""
import torch


def main():
    torch.set_printoptions(precision=2, sci_mode=False)

    # 本例按字切分，词表只包含这三个字；ID 不是位置或语义大小。
    vocab = {"我": 0, "爱": 1, "猫": 2}

    # 每个 token 一行，每行两个数。人为指定数值，便于手算。
    # from_pretrained 在这里加载下面的张量，不会下载任何模型。
    # freeze=False 允许训练更新；本程序没有训练步骤，参数不会更新。
    token_embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
        freeze=False,
    )

    # 按位置编号查另一张表，最多支持三个位置，向量维度同样为 2。
    # 这是绝对位置 embedding 的形式，数值只是示意，不是正弦公式。
    position_embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor([[0., 0.], [0.1, 0.2], [0.2, 0.4]]),
        freeze=False,
    )

    print("词表：", vocab)
    print("token embedding 表：\n", token_embedding.weight.detach())
    print("position embedding 表：\n", position_embedding.weight.detach())

    # 对照：换顺序，以及同一个 token 出现在两个不同位置。
    for text in ["我爱猫", "猫爱我", "我爱我"]:
        tokens = list(text)
        token_ids = torch.tensor([vocab[token] for token in tokens])
        position_ids = torch.arange(len(tokens))

        token_vectors = token_embedding(token_ids)
        position_vectors = position_embedding(position_ids)
        x = token_vectors + position_vectors

        print(f"\n文本：{text}")
        print("tokens：", tokens)
        print("token IDs：", token_ids.tolist())
        print("position IDs：", position_ids.tolist())
        # detach 只用于让打印更清楚；上面的 x 仍保留计算图。
        print("按 token 查出的向量：\n", token_vectors.detach())
        print("按位置查出的向量：\n", position_vectors.detach())
        print("相加后的 x：\n", x.detach())
        print("x.shape：", tuple(x.shape))

    # 这就是上一节输入 x 的一种来源。后续再使用投影参数：
    # q = x @ wq
    # k = x @ wk
    # v = x @ wv


if __name__ == "__main__":
    main()
