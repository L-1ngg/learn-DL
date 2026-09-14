"""uv run demos/08-sft-tiny/main.py；先建立玩具起点，再做完整参数 SFT。"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from model import TinyLM, generate


ROOT = Path(__file__).parent
PAIRS = [("猫", "鱼"), ("狗", "肉"), ("兔", "草"), ("牛", "草")]


def make_rows():
    rows = []
    for animal, food in PAIRS:
        for question in [f"{animal}吃什么？", f"{animal}喜欢吃什么？"]:
            rows.append({"prompt": f"问：{question}\n答：", "plain": food + "。",
                         "answer": json.dumps({"答案": food}, ensure_ascii=False, separators=(",", ":"))})
    return rows


def train(model, rows, vocab, *, field, answer_only, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    examples = []
    for row in rows:
        ids = torch.tensor([vocab[c] for c in row["prompt"] + row[field]] + [vocab["<eos>"]])
        labels = ids.clone()
        if answer_only:
            labels[:len(row["prompt"])] = -100
        examples.append((ids, labels))
    history = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        losses = []
        for ids, labels in examples:
            logits = model(ids[None, :])
            losses.append(F.cross_entropy(logits[0, :-1], labels[1:]))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if step == 0 or (step + 1) % 50 == 0:
            print(f"  step={step + 1:3d} loss={loss.item():.4f}")
    return history


def save(model, tokens, path):
    torch.save({"model": model.state_dict(), "tokens": tokens}, path)


def load(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = TinyLM(len(checkpoint["tokens"]))
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["tokens"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "base", "sft", "compare"], default="all")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    folder = ROOT / "outputs"
    folder.mkdir(exist_ok=True)
    rows = make_rows()
    # 字符集合取自训练材料；所有检查问题也只使用已知字符。
    tokens = ["<eos>"] + sorted(set("".join(r["prompt"] + r["plain"] + r["answer"] for r in rows)))
    vocab = {token: i for i, token in enumerate(tokens)}
    if args.stage in ("all", "base"):
        print("阶段 1：从随机参数建立玩具基础 checkpoint（不是公开预训练模型）")
        model = TinyLM(len(tokens))
        train(model, rows, vocab, field="plain", answer_only=False, steps=150)
        save(model, tokens, folder / "base.pt")
    if args.stage in ("all", "sft"):
        print("阶段 2：重新加载 base.pt，SFT 只监督 JSON 回答及 EOS")
        model, tokens = load(folder / "base.pt")
        vocab = {token: i for i, token in enumerate(tokens)}
        before = model.token_embedding.weight.detach().clone()
        history = train(model, rows, vocab, field="answer", answer_only=True, steps=150)
        save(model, tokens, folder / "sft.pt")
        print("embedding 最大变化：", (model.token_embedding.weight.detach() - before).abs().max().item())
        (folder / "loss.json").write_text(json.dumps(history, indent=2) + "\n")
    if args.stage in ("all", "compare"):
        base, tokens = load(folder / "base.pt")
        tuned, _ = load(folder / "sft.pt")
        comparisons = []
        # 前两条是训练问题；第三条是未参与训练的短问法，不保证正确。
        for split, question in [("train", "猫吃什么？"), ("train", "狗吃什么？"), ("held_out", "猫吃？")]:
            prompt = f"问：{question}\n答："
            row = {"split": split, "prompt": prompt, "before": generate(base, prompt, tokens),
                   "after": generate(tuned, prompt, tokens)}
            comparisons.append(row)
            print(json.dumps(row, ensure_ascii=False))
        (folder / "comparison.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
