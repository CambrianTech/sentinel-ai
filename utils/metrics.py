import math
import torch
import torch.nn.functional as F

def compute_perplexity(loss):
    return math.exp(loss)

def gate_statistics(model):
    stats = {}
    for layer_idx, block in enumerate(model.blocks):
        attn = block["attn"]
        gates = attn.gate.detach().cpu()
        stats[f"layer_{layer_idx}"] = {
            "mean_gate": gates.mean().item(),
            "min_gate": gates.min().item(),
            "max_gate": gates.max().item(),
            "num_active": (gates > 0.01).sum().item(),
        }
    return stats

def attention_entropy(attn_weights):
    # attn_weights: (batch, heads, seq_len, seq_len)
    eps = 1e-8
    p = attn_weights + eps  # avoid log(0)
    entropy = -torch.sum(p * torch.log(p), dim=-1)  # (batch, heads, seq_len)
    return entropy.mean(dim=-1).mean(dim=0)  # mean over seq and batch per head

def log_gate_stats(model):
    print("\n[Sentinel Gate Statistics]")
    stats = gate_statistics(model)
    for layer, s in stats.items():
        print(f"{layer}: mean={s['mean_gate']:.3f} | min={s['min_gate']:.3f} | max={s['max_gate']:.3f} | active={s['num_active']}")
    print()
    return stats