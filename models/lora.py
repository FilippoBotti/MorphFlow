import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def _set_module(root, name, module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def add_lora_to_cross_attention(model, rank=8, alpha=16, dropout=0.0, target_modules=("to_q", "to_kv")):
    replaced = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if ".cross_attn." not in name:
            continue

        short_name = name.split(".")[-1]
        if short_name not in target_modules:
            continue

        _set_module(
            model,
            name,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        replaced.append(name)

    return replaced


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def trainable_parameters(module):
    return [p for p in module.parameters() if p.requires_grad]


def print_trainable_parameters(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")