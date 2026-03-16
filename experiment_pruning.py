#!/usr/bin/env python
"""
Sentinel-AI Pruning Validation Experiment

Proves whether learned sentinel gates preserve model quality after pruning.
Trains distilgpt2 with L1 gate regularization, then prunes low-gate heads.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from datasets import load_dataset
from models.loaders.gpt2_loader_clean import load_adaptive_model_gpt_clean
import math

print("=== SENTINEL-AI ADAPTIVE TRAINING EXPERIMENT ===")
print("Model: distilgpt2 (82M params, 6 layers, 12 heads)")
print("Goal: Train with L1 gate regularization, measure pruning quality")
print()

# Load everything
config = GPT2Config.from_pretrained("distilgpt2")
base_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

adaptive_model = load_adaptive_model_gpt_clean(
    "distilgpt2", base_model, config, device="cpu", quiet=True
)
transformer = adaptive_model.transformer

# Load wikitext for training data
print("Loading wikitext-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in dataset["text"] if len(t) > 50][:200]


def tokenize(texts, max_len=128):
    encoded = tokenizer(
        texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt"
    )
    return encoded["input_ids"]


train_ids = tokenize(texts)
print(f"Training data: {train_ids.shape[0]} sequences, {train_ids.shape[1]} tokens each")


def measure_perplexity(model, ids):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for i in range(0, min(len(ids), 50), 5):
            batch = ids[i : i + 5]
            logits = model(batch).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size), batch[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
            n += 1
    return math.exp(total_loss / n)


base_ppl = measure_perplexity(adaptive_model, train_ids)
print(f"Baseline perplexity: {base_ppl:.2f}")

# ADAPTIVE TRAINING with gate regularization
print()
print("=== TRAINING (500 steps with L1 gate regularization, reg=0.05) ===")

adaptive_model.train()

gate_params = []
model_params = []
for name, param in adaptive_model.named_parameters():
    if "gate" in name:
        gate_params.append(param)
    else:
        model_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": model_params, "lr": 5e-5},
        {"params": gate_params, "lr": 5e-3},  # High LR so gates differentiate fast
    ]
)

gate_reg_weight = 0.15  # Strong L1 to drive unimportant gates toward 0
batch_size = 8
n_steps = 500

for step in range(n_steps):
    idx = torch.randint(0, len(train_ids), (batch_size,))
    batch = train_ids[idx]

    logits = adaptive_model(batch).logits
    lm_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size), batch[:, 1:].reshape(-1)
    )

    gate_l1 = sum(block.attn.gate.abs().sum() for block in transformer.blocks)
    total_gates = sum(block.attn.gate.shape[0] for block in transformer.blocks)
    gate_reg = gate_reg_weight * gate_l1 / total_gates

    loss = lm_loss + gate_reg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 100 == 0:
        all_gates = torch.cat([block.attn.gate.detach() for block in transformer.blocks])
        active = (all_gates > 0.5).sum().item()
        near_zero = (all_gates < 0.1).sum().item()
        print(
            f"Step {step+1:3d}: loss={loss.item():.3f} (lm={lm_loss.item():.3f} reg={gate_reg.item():.3f}) "
            f"gates: min={all_gates.min():.3f} max={all_gates.max():.3f} "
            f"active(>0.5)={active}/{len(all_gates)} near_zero(<0.1)={near_zero}"
        )

# Show final gate values per layer
print()
print(f"=== FINAL GATE VALUES (after {n_steps} training steps) ===")
for i, block in enumerate(transformer.blocks):
    g = block.attn.gate.detach()
    sorted_g = g.sort()[0]
    active = (g > 0.5).sum().item()
    print(f"Layer {i}: {active}/12 active | gates: [" + ", ".join(f"{v:.3f}" for v in sorted_g.tolist()) + "]")

# Prune based on learned gates
print()
print("=== LEARNED PRUNING (gate < 0.5 -> pruned) ===")
total_heads = 0
pruned_count = 0
for block in transformer.blocks:
    g = block.attn.gate.detach()
    for h in range(len(g)):
        total_heads += 1
        if g[h] < 0.5:
            block.attn.gate.data[h] = 0.001
            pruned_count += 1

pct = pruned_count / total_heads * 100
print(f"Pruned {pruned_count}/{total_heads} heads ({pct:.0f}%)")

# Measure post-pruning quality
pruned_ppl = measure_perplexity(adaptive_model, train_ids)
print(f"Post-pruning perplexity: {pruned_ppl:.2f}")
print(f"Baseline perplexity:     {base_ppl:.2f}")
print(f"Degradation ratio:       {pruned_ppl/base_ppl:.2f}x")

# Generate text comparison
adaptive_model.eval()
base_model.eval()
prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")


def greedy_gen(model, ids, n=30):
    g = ids.clone()
    with torch.no_grad():
        for _ in range(n):
            logits = model(g).logits
            g = torch.cat([g, logits[0, -1:].argmax(-1).unsqueeze(0)], dim=1)
    return tokenizer.decode(g[0])


print()
print(f'Baseline:  "{greedy_gen(base_model, input_ids)}"')
print(f'Pruned:    "{greedy_gen(adaptive_model, input_ids)}"')
