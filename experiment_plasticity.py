#!/usr/bin/env python
"""
Sentinel-AI Full Plasticity Experiment

Runs the complete biological cycle:
  1. Train with gate gradient flow (gates differentiate)
  2. Prune underutilized heads (gate < threshold)
  3. Clone/split overutilized heads into freed slots (mitosis)
  4. Continue training — cloned heads diverge and specialize

This proves the full adaptive architecture: a transformer that reshapes
itself during training, growing where it needs capacity and pruning where
it doesn't.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from datasets import load_dataset
from models.loaders.gpt2_loader_clean import load_adaptive_model_gpt_clean
from sentinel.models.adaptive_head_cloning import AdaptiveHeadManager
import math

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"=== SENTINEL-AI FULL PLASTICITY EXPERIMENT ===")
print(f"Device: {DEVICE}")
print()

# Load model
config = GPT2Config.from_pretrained("distilgpt2")
base_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

adaptive_model = load_adaptive_model_gpt_clean(
    "distilgpt2", base_model, config, device=DEVICE, quiet=True
)
transformer = adaptive_model.transformer

# Load training data
print("Loading wikitext-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in dataset["text"] if len(t) > 50][:500]

encoded = tokenizer(
    texts, truncation=True, max_length=128, padding="max_length", return_tensors="pt"
)
train_ids = encoded["input_ids"].to(DEVICE)
print(f"Training data: {train_ids.shape[0]} sequences")


def measure_perplexity(model, ids, n=50):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i in range(0, min(len(ids), n), 5):
            batch = ids[i : i + 5]
            logits = model(batch).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size),
                batch[:, 1:].reshape(-1),
            )
            total_loss += loss.item()
            count += 1
    model.train()
    return math.exp(total_loss / count)


def print_gate_summary(transformer):
    for i, block in enumerate(transformer.blocks):
        g = block.attn.gate.detach().cpu()
        active = (g > 0.1).sum().item()
        total = len(g)
        sorted_g = g.sort()[0]
        lo = sorted_g[0].item()
        hi = sorted_g[-1].item()
        print(f"  Layer {i}: {active}/{total} active  gates: [{lo:.3f} ... {hi:.3f}]")


def greedy_gen(model, prompt, n=30):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    g = ids.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            logits = model(g).logits
            g = torch.cat([g, logits[0, -1:].argmax(-1).unsqueeze(0)], dim=1)
    model.train()
    return tokenizer.decode(g[0])


# Measure baseline
base_ppl = measure_perplexity(adaptive_model, train_ids)
print(f"Baseline perplexity: {base_ppl:.2f}")
print()

# Initialize AdaptiveHeadManager
manager = AdaptiveHeadManager(
    model=adaptive_model,
    prune_threshold=0.25,
    clone_threshold=0.75,
    min_active_heads=4,
    max_heads_per_layer=12,
    update_frequency=50,
    warmup_steps=200,
)

# Optimizer with gate gradient flow
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
        {"params": gate_params, "lr": 5e-3},
    ]
)

# Training parameters
n_steps = 1000
batch_size = 8
gate_reg_weight = 0.1

print(f"=== TRAINING ({n_steps} steps, reg={gate_reg_weight}, with AdaptiveHeadManager) ===")
print()

for step in range(n_steps):
    idx = torch.randint(0, len(train_ids), (batch_size,))
    batch = train_ids[idx]

    logits = adaptive_model(batch).logits
    lm_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        batch[:, 1:].reshape(-1),
    )

    # L1 gate regularization
    gate_l1 = sum(block.attn.gate.abs().sum() for block in transformer.blocks)
    total_gates = sum(block.attn.gate.shape[0] for block in transformer.blocks)
    gate_reg = gate_reg_weight * gate_l1 / total_gates

    loss = lm_loss + gate_reg

    optimizer.zero_grad()
    loss.backward()

    # Collect gate gradients for the manager
    batch_gradients = {}
    for layer_idx, block in enumerate(transformer.blocks):
        if block.attn.gate.grad is not None:
            for head_idx in range(block.attn.num_heads):
                grad_mag = abs(block.attn.gate.grad[head_idx].item())
                batch_gradients[(layer_idx, head_idx)] = grad_mag

    optimizer.step()

    # Feed gradients to the plasticity manager
    manager.step(batch_gradients)

    # Progress reports
    if (step + 1) % 200 == 0:
        all_gates = torch.cat([block.attn.gate.detach().cpu() for block in transformer.blocks])
        active = (all_gates > 0.1).sum().item()
        near_zero = (all_gates < 0.05).sum().item()
        above_one = (all_gates > 1.0).sum().item()
        ppl = measure_perplexity(adaptive_model, train_ids)
        print(f"\n--- Step {step+1} ---")
        print(f"Loss: {loss.item():.3f} (lm={lm_loss.item():.3f} reg={gate_reg.item():.3f})")
        print(f"Perplexity: {ppl:.2f}")
        print(f"Gates: {active}/{len(all_gates)} active, {near_zero} near-zero, {above_one} above 1.0")
        print(f"Range: [{all_gates.min():.3f} ... {all_gates.max():.3f}]")
        print_gate_summary(transformer)

        # Show utilization scores from manager
        all_utils = [s.utilization_score for s in manager.head_stats.values()]
        print(f"  Utilization scores: min={min(all_utils):.3f} max={max(all_utils):.3f} "
              f"mean={sum(all_utils)/len(all_utils):.3f}")
        prune_candidates = sum(1 for u in all_utils if u < manager.prune_threshold)
        clone_candidates = sum(1 for u in all_utils if u > manager.clone_threshold)
        print(f"  Prune candidates (<{manager.prune_threshold}): {prune_candidates}  "
              f"Clone candidates (>{manager.clone_threshold}): {clone_candidates}")

# Final results
print()
print("=" * 60)
print("=== FINAL RESULTS ===")
print("=" * 60)

final_ppl = measure_perplexity(adaptive_model, train_ids)
all_gates = torch.cat([block.attn.gate.detach().cpu() for block in transformer.blocks])
active = (all_gates > 0.1).sum().item()
pruned = (all_gates <= 0.1).sum().item()
above_one = (all_gates > 1.0).sum().item()

print(f"Baseline perplexity:  {base_ppl:.2f}")
print(f"Final perplexity:     {final_ppl:.2f}")
print(f"Heads active:         {active}/{len(all_gates)}")
print(f"Heads pruned:         {pruned}/{len(all_gates)} ({pruned/len(all_gates)*100:.0f}%)")
print(f"Heads growing (>1.0): {above_one}/{len(all_gates)}")
print()
print_gate_summary(transformer)

# Architecture report
print()
print(manager.get_architecture_report())

# Generation comparison
print()
prompt = "The meaning of life is"
base_model.eval()
base_model.to(DEVICE)
print(f'Baseline:  "{greedy_gen(base_model, prompt)}"')
print(f'Adaptive:  "{greedy_gen(adaptive_model, prompt)}"')
