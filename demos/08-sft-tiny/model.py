"""小型 causal Transformer，仅作为 SFT 流程的可训练载体。"""
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab_size, width=48, max_length=96):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_length, width)
        # 使用现成层，当前先关注数据、loss 和微调，而不展开 Transformer 内部。
        layer = nn.TransformerEncoderLayer(
            width, nhead=2, dim_feedforward=96, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        future = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device).triu(1)
        # 名字虽叫 EncoderLayer，这里用 causal mask，只做前缀内的自注意力。
        x = self.layers(x, mask=future)
        return self.output(x)


@torch.no_grad()
def generate(model, prompt, tokens):
    model.eval()
    vocab = {token: i for i, token in enumerate(tokens)}
    ids = [vocab[c] for c in prompt]
    answer = []
    for _ in range(min(32, model.max_length - len(ids))):
        next_id = model(torch.tensor([ids]))[0, -1].argmax().item()
        if next_id == vocab["<eos>"]:
            break
        ids.append(next_id)
        answer.append(tokens[next_id])
    return "".join(answer)
