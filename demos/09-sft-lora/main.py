"""uv run --extra sft demos/09-sft-lora/main.py --stage train"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import TEST, TRAIN, VALIDATION, messages, tokenize_example


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ROOT = Path(__file__).parent


def load_base(device):
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=dtype, attn_implementation="sdpa",
    )
    return model.to(device)


def batch(row, device):
    # 一条样本一个 batch，不引入 packing 或 padding；labels 与 input_ids 等长。
    return {key: torch.tensor([value], device=device) for key, value in row.items()}


@torch.no_grad()
def reply(model, tokenizer, example, device):
    model.eval()
    ids = tokenizer.apply_chat_template(
        messages(example), tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)
    output = model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
        do_sample=False, temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def mean_loss(model, rows, device):
    model.eval()
    total, count = 0.0, 0
    for row in rows:
        n = sum(label != -100 for label in row["labels"][1:])
        total += model(**batch(row, device), use_cache=False).loss.item() * n
        count += n
    return total / count


def correct_json(text, expected):
    try:
        return json.loads(text) == json.loads(expected)
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["inspect", "train", "infer"], default="inspect")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", default="我的包裹现在到哪里了？")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须大于零")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    train_rows = [tokenize_example(tokenizer, row) for row in TRAIN]
    if args.stage == "inspect":
        row = train_rows[0]
        print("完整对话：\n", tokenizer.decode(row["input_ids"]))
        print("计入 loss 的部分：\n", tokenizer.decode([i for i in row["labels"] if i != -100]))
        print("总 token 数：", len(row["input_ids"]), "监督 token 数：", sum(i != -100 for i in row["labels"]))
        print("只对 assistant 回答及模板结束部分计分；模型内部会 shift，不手动错位 labels。")
        return

    adapter_dir = args.output_dir / "adapter"
    if args.stage == "infer":
        model = PeftModel.from_pretrained(load_base(device), adapter_dir)
        print(reply(model, tokenizer, {"user": args.prompt}, device))
        return

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    base = load_base(device)
    probes = [("train", row) for row in TRAIN[:2]] + [("test", row) for row in TEST]
    before = [reply(base, tokenizer, row, device) for _, row in probes]
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none", revision=REVISION,
    ))
    model.print_trainable_parameters()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    trainable_count = sum(p.numel() for _, p in trainable)
    # 跟踪一块原始权重，确认 LoRA 训练没有直接改动它。
    frozen = next(p for name, p in model.named_parameters() if "q_proj.base_layer.weight" in name)
    frozen_before = frozen.detach().cpu().clone()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=3e-4)
    validation_rows = [tokenize_example(tokenizer, row) for row in VALIDATION]
    history = [{"epoch": 0, "validation_loss": mean_loss(model, validation_rows, device)}]
    best_loss = float("inf")
    best_epoch = None
    generator = torch.Generator().manual_seed(7)
    accumulation = 4
    assert len(train_rows) % accumulation == 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        losses = []
        for step, index in enumerate(torch.randperm(len(train_rows), generator=generator).tolist(), 1):
            # HF causal LM 内部完成 logits[:-1] 与 labels[1:] 的对齐。
            loss = model(**batch(train_rows[index], device), use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("loss 非有限值，停止并检查数值设置。")
            (loss / accumulation).backward()
            losses.append(loss.item())
            if step % accumulation == 0:
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                optimizer.step()
                optimizer.zero_grad()
        validation_loss = mean_loss(model, validation_rows, device)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation_loss": validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)

    assert torch.equal(frozen_before, frozen.detach().cpu()), "原始权重意外变化"
    del model, base, optimizer, trainable, frozen, loss
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    # 重新加载原始基座 + 磁盘 adapter，实际验证保存产物可以推理。
    restored = PeftModel.from_pretrained(load_base(device), adapter_dir)
    comparisons = []
    for (split, example), old in zip(probes, before):
        new = reply(restored, tokenizer, example, device)
        comparisons.append({"split": split, "prompt": example["user"], "expected": example["assistant"],
                            "before": old, "after": new,
                            "before_correct": correct_json(old, example["assistant"]),
                            "after_correct": correct_json(new, example["assistant"])})
    results = {"model": MODEL_ID, "revision": REVISION, "seed": 7, "device": device,
               "epochs": args.epochs, "best_epoch": best_epoch, "trainable_parameters": trainable_count,
               "history": history, "comparisons": comparisons, "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for row in comparisons:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    test = [row for row in comparisons if row["split"] == "test"]
    print(f"固定 test 的 JSON/类别正确数：{sum(r['before_correct'] for r in test)}/{len(test)} → {sum(r['after_correct'] for r in test)}/{len(test)}")
    print(f"已从磁盘重载 adapter；结果：{args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
