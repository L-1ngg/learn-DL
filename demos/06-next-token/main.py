"""uv run demos/06-next-token/main.py；训练一个带 attention 的下一字预测器。"""
import math

import torch
from torch import nn


SENTENCES = ["猫爱吃鱼。", "狗爱吃肉。"]
VOCAB = list(dict.fromkeys("".join(SENTENCES)))
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
MAX_CONTEXT = max(len(text) for text in SENTENCES) - 1


def encode(text):
    return torch.tensor([TOKEN_TO_ID[token] for token in text])


class NextTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        width = 16
        self.token_embedding = nn.Embedding(len(VOCAB), width)
        self.position_embedding = nn.Embedding(MAX_CONTEXT, width)
        self.wq = nn.Linear(width, width, bias=False)
        self.wk = nn.Linear(width, width, bias=False)
        self.wv = nn.Linear(width, width, bias=False)
        # 每个位置的向量 → 每个候选字的分数。
        self.output_layer = nn.Linear(width, len(VOCAB))

    def forward(self, token_ids):
        positions = torch.arange(len(token_ids), device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        scores = q @ k.T / math.sqrt(q.shape[-1])
        future = torch.ones_like(scores, dtype=torch.bool).triu(diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        mixed = weights @ v
        logits = self.output_layer(mixed)
        return logits, weights


@torch.no_grad()
def generate(model, prefix):
    model.eval()
    text = prefix
    # 本例每句最多五个字（含句号），位置表最多接收四个输入字。
    for _ in range(MAX_CONTEXT + 1 - len(prefix)):
        if text.endswith("。"):
            break
        logits, _ = model(encode(text))
        next_id = logits[-1].argmax().item()  # 只取最后位置的下一字预测。
        text += VOCAB[next_id]
    return text


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    torch.set_printoptions(precision=3, sci_mode=False)

    # 输入与答案错开一位：猫爱吃鱼 → 爱吃鱼。
    examples = [(encode(text[:-1]), encode(text[1:])) for text in SENTENCES]
    print("词表：", TOKEN_TO_ID)
    for text in SENTENCES:
        print(f"输入：{text[:-1]}  答案：{text[1:]}")

    model = NextTokenModel()
    before = model.token_embedding.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    print("\n训练前（随机参数）：")
    for prefix in ["猫", "狗"]:
        print(f"{prefix} → {generate(model, prefix)}")

    model.train()
    for step in range(301):
        # 一句一句前向计算，避免现在引入 batch 和 padding。
        losses = []
        for inputs, targets in examples:
            logits, _ = model(inputs)
            # logits 为 [位置数, 词表大小]，targets 为 [位置数]。
            # CrossEntropyLoss 接收原始分数，这里不要先 softmax。
            losses.append(criterion(logits, targets))
        loss = torch.stack(losses).mean()
        if step % 100 == 0:
            print(f"已更新 {step:3d} 次，训练 loss = {loss.item():.6f}")
        if step == 300:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n训练后（只检查训练句子的续写）：")
    for prefix in ["猫", "狗"]:
        print(f"{prefix} → {generate(model, prefix)}")

    change = (model.token_embedding.weight.detach() - before).abs().max().item()
    print(f"token embedding 参数最大绝对变化：{change:.4f}")
    with torch.no_grad():
        logits, weights = model(encode("猫爱吃"))
        probabilities = torch.softmax(logits[-1], dim=-1)
        print("\n输入“猫爱吃”的读取比例（行读列，顺序均为猫/爱/吃）：\n", weights)
        print("最后位置预测的下一字概率：")
        for token, probability in zip(VOCAB, probabilities.tolist()):
            print(f"  {token}：{probability:.3f}")


if __name__ == "__main__":
    main()
