# Continuous Defrag: Accelerating Training Through Structural Compression

**The key insight**: as you prune and defrag during training, the model gets physically smaller. Smaller model = less VRAM = bigger batch sizes = faster training. Each cycle accelerates the next. The model trains faster as it learns what it doesn't need.

## The Problem With Current Approach

Current forge pipeline:
```
Cycle 1: train (batch=1, 27B) → prune → [hooks mask dead heads]
Cycle 2: train (batch=1, 27B) → prune → [more hooks]
Cycle 3: train (batch=1, 27B) → prune → [even more hooks]
Save: 27B model with masked heads (same size, wasted compute)
```

Every cycle processes the FULL model. Dead heads still consume:
- **VRAM**: weight tensors still allocated
- **Compute**: forward pass calculates attention for masked heads
- **Time**: batch_size stays at 1 because VRAM is full

## Continuous Defrag

```
Cycle 1: train (batch=1, 27B, 17.9GB) → merge LoRA → prune → DEFRAG
  → Model shrinks to 24.5B (16.2GB). Freed 1.7GB. Increase batch to 2.

Cycle 2: train (batch=2, 24.5B, 16.2GB) → merge LoRA → prune → DEFRAG
  → Model shrinks to 22B (14.5GB). Freed 1.7GB. Increase batch to 3.

Cycle 3: train (batch=3, 22B, 14.5GB) → merge LoRA → prune → DEFRAG
  → Model shrinks to 19.5B (12.8GB). Training 3x faster than cycle 1.

Save: 19.5B model (physically smaller, actually faster inference)
```

**Cycle 1**: 500 steps × 3.1s/step = 26 min
**Cycle 2**: 500 steps × 1.6s/step = 13 min (batch=2, 2x faster)
**Cycle 3**: 500 steps × 1.0s/step = 8 min (batch=3, 3x faster)
**Total**: 47 min instead of 78 min (40% faster, same quality)

## Architecture

### The Forge-Defrag Cycle

```
┌─────────────────────────────────────────────────────────┐
│                    FORGE CYCLE N                        │
│                                                         │
│  1. MEASURE        Estimate defrag savings              │
│     └─ estimate_defrag_savings(model)                   │
│                                                         │
│  2. TRAIN          LoRA on domain data                  │
│     └─ train_lora(model, batch_size=adaptive)           │
│     └─ Inference sample every 200 steps (proof)         │
│                                                         │
│  3. MERGE          Fold LoRA back into base weights     │
│     └─ model = model.merge_and_unload()                 │
│                                                         │
│  4. EVALUATE       Measure quality on held-out data     │
│     └─ post_train_ppl = evaluate(model, val_loader)     │
│                                                         │
│  5. PRUNE          Identify low-importance heads        │
│     └─ dead_heads = compute_head_importance(model)      │
│     └─ Mark heads for removal (don't zero yet)          │
│                                                         │
│  6. DEFRAG         Structurally remove dead heads       │
│     └─ freed = defrag_live_model(model, dead_heads)     │
│     └─ Update batch_size based on freed VRAM            │
│     └─ torch.cuda.empty_cache()                         │
│                                                         │
│  7. EVALUATE       Verify quality after defrag          │
│     └─ post_defrag_ppl = evaluate(model, val_loader)    │
│     └─ If ppl degraded > threshold: STOP (went too far) │
│                                                         │
│  8. CHECKPOINT     Save intermediate state              │
│     └─ model.save_pretrained(checkpoint_dir)            │
│     └─ Save cycle metadata (heads removed, ppl, size)   │
│                                                         │
│  9. ADAPT          Recalculate training config           │
│     └─ new_batch = calculate_batch_for_vram(model)      │
│     └─ new_accum = effective_batch / new_batch          │
│     └─ Log: "Cycle N+1: batch {old}→{new}, {freed}MB"  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Adaptive Batch Sizing

The core optimization: after each defrag, recalculate how much VRAM is free and increase the batch size.

```python
def calculate_adaptive_batch(model, vram_total_gb, seq_len=256):
    """
    Calculate max safe batch size given current model size and VRAM.

    VRAM budget:
      model_weights + lora_params + optimizer_states +
      batch_size × seq_len × hidden × bytes_per_activation

    Solve for batch_size.
    """
    model_vram = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    lora_overhead = 0.5  # GB, approximate
    optimizer_overhead = 0.3  # GB for 8-bit Adam on LoRA params
    system_overhead = 2.0  # GB for PyTorch allocator, KV cache, etc.

    available = vram_total_gb - model_vram - lora_overhead - optimizer_overhead - system_overhead

    # Each batch element needs ~activation_bytes for forward+backward
    # With gradient checkpointing: ~2 × hidden_size × num_layers × seq_len × 2 (fp16)
    tc = getattr(model.config, "text_config", model.config)
    hidden = getattr(tc, "hidden_size", 4096)
    layers = getattr(tc, "num_hidden_layers", 32)
    activation_per_sample_gb = (2 * hidden * layers * seq_len * 2) / 1e9

    # Add logit tensor: seq_len × vocab_size × 2 bytes
    vocab = getattr(tc, "vocab_size", 248320)
    logit_per_sample_gb = (seq_len * vocab * 2) / 1e9

    per_sample = activation_per_sample_gb + logit_per_sample_gb

    max_batch = max(1, int(available / per_sample))
    return min(max_batch, 8)  # Cap at 8 — diminishing returns beyond that
```

### Defrag Savings Per Cycle (Qwen3.5-27B)

Starting: 23.6B params, 64 layers, 24 query heads, 4 KV heads, head_dim=213

| Cycle | Heads Pruned | Params After | VRAM (4-bit) | Batch Size | Steps/sec |
|-------|-------------|-------------|-------------|-----------|----------|
| 0 (baseline) | 0/1536 | 23.6B | 17.9GB | 1 | 0.32 |
| 1 (10% prune) | 154/1536 | 21.5B | 16.3GB | 2 | 0.55 |
| 2 (20% cumulative) | 307/1536 | 19.4B | 14.7GB | 3 | 0.75 |
| 3 (30% cumulative) | 461/1536 | 17.3B | 13.1GB | 4 | 0.90 |

**Cycle 3 trains 2.8x faster than cycle 1.** Same number of steps, less wall time.

### The GQA Complication

Qwen3.5-27B: 24 query heads, 4 KV heads (groups of 6).

A KV head can only be removed when ALL 6 of its query heads are dead. This means:
- Early cycles: mostly Q/O projection savings (individual heads)
- Later cycles: once full groups are pruned, K/V projections also shrink
- The savings accelerate as more complete groups become dead

```
Cycle 1: prune 10% → ~10% of Q/O shrinks, 0% of K/V (no full groups yet)
Cycle 2: prune 20% → ~20% of Q/O, maybe 5% of K/V (a few full groups)
Cycle 3: prune 30% → ~30% of Q/O, maybe 15% of K/V (many full groups)
```

### Checkpointing Between Cycles

Each cycle saves a checkpoint. If the model degrades past a threshold, roll back:

```
output/forged/qwen3.5-27b/
  ├── cycle-0/           # Baseline (no pruning)
  │   ├── model/         # Full model checkpoint
  │   └── metadata.json  # {ppl: 2.62, params: 23.6B, heads: 1536}
  ├── cycle-1/
  │   ├── model/         # After train + prune + defrag
  │   └── metadata.json  # {ppl: 2.15, params: 21.5B, heads: 1382}
  ├── cycle-2/
  │   ├── model/
  │   └── metadata.json  # {ppl: 2.08, params: 19.4B, heads: 1229}
  └── cycle-3/
      ├── model/         # Final
      └── metadata.json  # {ppl: 2.05, params: 17.3B, heads: 1075}
```

If cycle 3 degrades: publish cycle 2's model instead. No wasted work.

### Quality Gates

Defrag must not destroy model quality. Two gates:

1. **Post-train gate**: if training loss diverges (NaN or increasing), stop training, keep previous cycle's model.

2. **Post-defrag gate**: if perplexity increases more than 5% after defrag, the structural removal broke something. Roll back to pre-defrag state.

```python
# After defrag
post_defrag_ppl = evaluate(model, val_loader)
degradation = (post_defrag_ppl - post_train_ppl) / post_train_ppl * 100

if degradation > 5.0:
    print(f"  DEFRAG DEGRADED QUALITY by {degradation:.1f}% — rolling back")
    model = load_checkpoint(previous_cycle_path)
    break  # Stop forging, publish last good cycle
```

## Integration With Forge Pipeline

### forge_model.py Changes

```python
from defrag_inline import defrag_live_model, estimate_defrag_savings

for cycle in range(1, args.cycles + 1):
    # Train
    model = train_lora(model, train_loader, cfg, args.steps, args.lr, out)
    post_train = evaluate(model, eval_loader, out)

    # Prune (identify dead heads)
    importance = compute_head_importance(model, info)
    dead_heads, _ = select_heads_to_prune(importance, cycle_prune)

    # Estimate savings before committing
    est_savings, _ = estimate_defrag_savings(model, dead_heads)
    print(f"  Defrag would free {est_savings/1e9:.2f}GB")

    # Defrag (structural removal)
    freed = defrag_live_model(model, dead_heads)
    print(f"  Defragged: freed {freed/1e9:.2f}GB VRAM")

    # Quality gate
    post_defrag = evaluate(model, eval_loader, out)
    if post_defrag["perplexity"] > post_train["perplexity"] * 1.05:
        print("  Quality degraded >5% — stopping")
        break

    # Adapt batch size for next cycle
    new_batch = calculate_adaptive_batch(model, vram_gb)
    if new_batch > cfg.batch_size:
        print(f"  Batch size: {cfg.batch_size} → {new_batch}")
        cfg.batch_size = new_batch
        cfg.grad_accum_steps = max(1, 8 // new_batch)
        train_loader = remake_dataloader(train_loader, new_batch)

    # Checkpoint
    save_cycle_checkpoint(model, out / f"cycle-{cycle}", metadata)
```

## Compound Effect: The Numbers

For Qwen3.5-27B with continuous defrag vs without:

### Without Defrag (current)
- 3 cycles × 500 steps × 3.1s/step = 4650s = **78 min**
- Final model: 23.6B params (same size, hooks masking dead heads)
- GGUF Q4: ~15GB

### With Continuous Defrag
- Cycle 1: 500 × 3.1s = 1550s (26 min)
- Cycle 2: 500 × 1.6s = 800s (13 min)  ← batch=2
- Cycle 3: 500 × 1.0s = 500s (8 min)   ← batch=3
- **Total: 2850s = 47 min (40% faster)**
- Final model: 17.3B params (physically smaller)
- GGUF Q4: ~10GB (33% smaller than without defrag)

### The Flywheel
```
Prune → Defrag → Free VRAM → Bigger batch → Faster training
  → Better head importance estimates (more data per step)
  → Smarter pruning → More to defrag → ...
```

Each step enables the next to be better AND faster.

## Future: Continuous Defrag + Grid

On the grid, continuous defrag enables dynamic work redistribution:

```
Node A (5090, 32GB): forging 27B, cycle 2, model now fits in 15GB
  → Node B (3090, 24GB): "I can take over — model fits now"
  → Node A freed for a new job

Node C (MacBook, 16GB): waiting for a model small enough
  → Cycle 3: model is 13GB → Node C can run inference
  → Grid routes inference traffic to Node C
```

The model "flows downhill" to smaller hardware as it compresses.

## Dependencies

- `defrag_inline.py` — live in-place structural pruning (committed)
- `defrag_model.py` — post-processing structural pruning (committed)
- `forge_model.py` — forge pipeline (needs cycle integration)
- Issue #94 — Structural pruning defrag
- Issue #85 — Forging cost model (defrag changes the cost equation)
