"""uv run demos/07-sft-labels/main.py；看清 SFT 在哪里计算 loss。"""
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def make_example(prompt, answer, vocab):
    # EOS 是一个独立 token，教模型何时结束回答。
    ids = torch.tensor([vocab[c] for c in prompt + answer] + [vocab["<eos>"]])
    labels = ids.clone()
    labels[:len(prompt)] = -100
    return ids, labels


def main():
    examples = [("问：猫吃什么？\n答：", "鱼。"), ("问：狗吃什么？请简答。\n答：", "肉。")]
    tokens = ["<pad>", "<eos>"] + sorted(set("".join(p + a for p, a in examples)))
    vocab = {token: i for i, token in enumerate(tokens)}
    rows = [make_example(p, a, vocab) for p, a in examples]
    input_ids = pad_sequence([r[0] for r in rows], batch_first=True, padding_value=0)
    labels = pad_sequence([r[1] for r in rows], batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(vocab["<pad>"])

    print("第一条完整输入：", examples[0][0] + examples[0][1] + "<eos>")
    print("当前输入 → 要预测的下一个 token → 是否计入 loss")
    # 模型第 i 个位置的 logits 对应第 i+1 个位置的答案。
    for i in range(input_ids.shape[1] - 1):
        current = tokens[input_ids[0, i].item()]
        target = tokens[input_ids[0, i + 1].item()]
        counted = labels[0, i + 1].item() != -100
        print(f"{current!r:10} → {target!r:10} → {'计算' if counted else '忽略'}")

    # 使用均匀预测来专门观察 loss mask，不创建或训练模型。
    logits = torch.zeros(*input_ids.shape, len(tokens), requires_grad=True)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss = F.cross_entropy(shift_logits.reshape(-1, len(tokens)), shift_labels.reshape(-1))
    loss.backward()
    print("\ninput_ids：\n", input_ids)
    print("labels（prompt 和 padding 为 -100）：\n", labels)
    print("attention_mask（非 padding 为 True）：\n", attention_mask)
    print("每条样本计入 loss 的 token 数：", shift_labels.ne(-100).sum(1).tolist())
    print("均匀预测的 loss：", round(loss.item(), 4))
    print("第一条各预测位置的 logits 梯度绝对值之和：")
    print(logits.grad[0].abs().sum(-1))
    print("prompt 仍是上下文；-100 只取消对应目标的直接损失，不等于 attention 遮罩。")


if __name__ == "__main__":
    main()
