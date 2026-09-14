"""uv run --extra sft python -m unittest discover -s tests -v"""
import importlib.util
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


labels_demo = load_file("sft_labels", "demos/07-sft-labels/main.py")
tiny = load_file("sft_tiny", "demos/08-sft-tiny/model.py")
data = load_file("sft_data", "demos/09-sft-lora/data.py")


class SFTContracts(unittest.TestCase):
    def test_first_answer_eos_and_padding(self):
        vocab = {"<pad>": 0, "<eos>": 1, "问": 2, "：": 3, "鱼": 4, "。": 5}
        short = labels_demo.make_example("问：", "鱼。", vocab)
        long = labels_demo.make_example("问问：", "鱼。", vocab)
        labels = pad_sequence([short[1], long[1]], batch_first=True, padding_value=-100)
        # 最后一个 prompt 位置负责预测第一个回答 token，而不是将它误遮住。
        self.assertEqual(labels[0, 2].item(), vocab["鱼"])
        self.assertEqual(labels[0, 4].item(), vocab["<eos>"])
        self.assertEqual(labels[0, 5].item(), -100)
        self.assertEqual(labels[:, 1:].ne(-100).sum(1).tolist(), [3, 3])
        logits = torch.randn(2, 6, len(vocab))
        baseline = F.cross_entropy(logits[:, :-1].reshape(-1, len(vocab)), labels[:, 1:].reshape(-1))
        changed = logits[:, :-1].clone()
        changed[labels[:, 1:] == -100] = torch.randn_like(changed[labels[:, 1:] == -100]) * 10
        actual = F.cross_entropy(changed.reshape(-1, len(vocab)), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(baseline, actual)

    def test_prompt_still_affects_answer_and_receives_gradient(self):
        torch.manual_seed(7)
        model = tiny.TinyLM(8)
        model.eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])
        logits = model(ids)
        changed_future = model(torch.tensor([[1, 2, 3, 5]]))
        torch.testing.assert_close(logits[:, :3], changed_future[:, :3])
        changed_prompt = model(torch.tensor([[6, 2, 3, 4]]))
        self.assertFalse(torch.allclose(logits[:, 1], changed_prompt[:, 1]))
        F.cross_entropy(logits[0, :-1], labels[0, 1:]).backward()
        self.assertGreater(model.token_embedding.weight.grad[1].abs().sum().item(), 0)

    def test_data_splits_do_not_overlap(self):
        splits = [{row["user"] for row in rows} for rows in (data.TRAIN, data.VALIDATION, data.TEST)]
        self.assertFalse(splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])

    def test_huggingface_shifts_labels_once(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        )).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])
        result = model(input_ids=ids, labels=labels)
        expected = F.cross_entropy(result.logits[0, :-1].float(), labels[0, 1:])
        torch.testing.assert_close(result.loss, expected)

    def test_real_chat_template_answer_boundary(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
            local_files_only=True,
        )
        for example in data.TRAIN + data.VALIDATION + data.TEST:
            row = data.tokenize_example(tokenizer, example)
            first = next(i for i, label in enumerate(row["labels"]) if label != -100)
            supervised = row["labels"][first:]
            self.assertEqual(row["labels"][:first], [-100] * first)
            self.assertEqual(supervised, row["input_ids"][first:])
            self.assertTrue(tokenizer.decode(supervised).startswith(example["assistant"]))
            self.assertIn(tokenizer.eos_token_id, supervised)
            self.assertEqual(len(row["labels"]), len(row["input_ids"]))
        with self.assertRaises(ValueError):
            data.tokenize_example(tokenizer, data.TRAIN[0], max_length=1)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
