#!/usr/bin/env python3
"""
forge_model.py v2 — Lean model forging with QLoRA for consumer hardware.

The key insight: pruning and evaluation are inference-only (low memory).
Only retraining needs training memory. Use QLoRA for that step.

Memory budget for 32GB VRAM:
  - 4B model fp16: 8GB → leaves 24GB for training overhead
  - 4B model 4-bit: 2.5GB → leaves 29GB 
  - 27B model 4-bit: 14GB → leaves 18GB (tight but works with LoRA)

Usage:
    python forge_model.py Qwen/Qwen3.5-4B
    python forge_model.py Qwen/Qwen3.5-27B --load-in-4bit
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch


def nested_config(config):
    """Get the text config from possibly nested config (Qwen3.5 etc)."""
    return getattr(config, "text_config", config)


def get_model_params(model_name: str) -> dict:
    """Get model architecture info."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    tc = nested_config(config)
    return {
        "num_layers": getattr(tc, "num_hidden_layers", 12),
        "num_heads": getattr(tc, "num_attention_heads", 12),
        "num_kv_heads": getattr(tc, "num_key_value_heads", None),
        "hidden_size": getattr(tc, "hidden_size", 768),
        "head_dim": getattr(tc, "hidden_size", 768) // getattr(tc, "num_attention_heads", 12),
        "vocab_size": getattr(tc, "vocab_size", getattr(config, "vocab_size", 32000)),
        "intermediate_size": getattr(tc, "intermediate_size", 4 * getattr(tc, "hidden_size", 768)),
    }


def estimate_memory(params: dict, load_4bit: bool = False) -> dict:
    """Estimate memory requirements."""
    h = params["hidden_size"]
    n = params["num_layers"]
    v = params["vocab_size"]
    i = params["intermediate_size"]
    total_params = v * h + n * (4 * h * h + 3 * h * i)
    
    fp16_gb = total_params * 2 / 1e9
    q4_gb = total_params * 0.5 / 1e9  # 4-bit
    
    model_gb = q4_gb if load_4bit else fp16_gb
    # LoRA adds ~2-5% parameter overhead
    lora_gb = total_params * 2 * 0.03 / 1e9  
    # Adam optimizer: 2x param size for momentum/variance (but only for LoRA params)
    optim_gb = lora_gb * 8  # fp32 optimizer states
    
    return {
        "model_gb": model_gb,
        "lora_gb": lora_gb,
        "optimizer_gb": optim_gb,
        "total_training_gb": model_gb + lora_gb + optim_gb + 2,  # +2 for activations
        "fp16_gb": fp16_gb,
        "params_b": total_params / 1e9,
    }


def load_model(model_name: str, load_4bit: bool = False):
    """Load model for inference (pruning + evaluation)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    kwargs = {"low_cpu_mem_usage": True}
    
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded. VRAM: {vram:.1f}GB, RAM: {_ram_used():.1f}GB")
    return model, tokenizer


def _ram_used():
    import psutil
    return psutil.virtual_memory().used / 1e9


def make_dataloader(tokenizer, max_length=128, batch_size=4, max_samples=2000):
    """Load limited dataset."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    train = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{max_samples}]")
    val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:200]")

    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")

    train = train.filter(lambda x: len(x["text"].strip()) > 20)
    val = val.filter(lambda x: len(x["text"].strip()) > 20)
    train = train.map(tok_fn, batched=True, remove_columns=["text"])
    val = val.map(tok_fn, batched=True, remove_columns=["text"])
    train.set_format("torch")
    val.set_format("torch")

    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=batch_size)


@torch.no_grad()
def evaluate(model, eval_loader):
    """Compute perplexity."""
    model.eval()
    total_loss, total_tokens = 0, 0
    for batch in eval_loader:
        ids = batch["input_ids"].to(model.device)
        mask = batch["attention_mask"].to(model.device)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        total_loss += out.loss.item() * ids.numel()
        total_tokens += ids.numel()
    avg = total_loss / max(total_tokens, 1)
    return {"loss": avg, "perplexity": torch.exp(torch.tensor(avg)).item()}


def prune_heads_by_zeroing(model, prune_percent=0.1):
    """
    Prune attention heads by zeroing their weights.
    Uses gradient-free entropy estimation via forward pass output variance.
    """
    params = get_model_params(model.config._name_or_path if hasattr(model.config, '_name_or_path') else "unknown")
    tc = nested_config(model.config)
    num_layers = params["num_layers"]
    num_heads = params["num_heads"]
    head_dim = params["head_dim"]
    total = num_layers * num_heads
    num_prune = int(total * prune_percent)

    # Compute importance: L2 norm of Q projection weights per head
    importance = torch.zeros(num_layers, num_heads)
    
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    
    if layers is None:
        print("  ERROR: Cannot find model layers for pruning")
        sys.exit(1)

    for li in range(num_layers):
        layer = layers[li]
        attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
        if attn is None:
            continue
        q = getattr(attn, "q_proj", None)
        if q is not None and hasattr(q, "weight"):
            w = q.weight.data.float()
            for hi in range(num_heads):
                s, e = hi * head_dim, (hi + 1) * head_dim
                if e <= w.shape[0]:
                    importance[li, hi] = w[s:e].norm().item()

    # Prune lowest importance heads
    flat = importance.flatten()
    _, indices = flat.sort()
    prune_indices = indices[:num_prune]

    heads_pruned = {}
    for idx in prune_indices:
        li = idx.item() // num_heads
        hi = idx.item() % num_heads
        heads_pruned.setdefault(li, []).append(hi)

        layer = layers[li]
        attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
        if attn is None:
            continue

        s, e = hi * head_dim, (hi + 1) * head_dim
        kv_heads = getattr(tc, "num_key_value_heads", num_heads)

        for pname in ["q_proj", "o_proj"]:
            proj = getattr(attn, pname, None)
            if proj and hasattr(proj, "weight"):
                with torch.no_grad():
                    if pname == "o_proj":
                        proj.weight.data[:, s:e] = 0
                    else:
                        proj.weight.data[s:e, :] = 0

        # For GQA K/V, only zero if this head's KV group is fully pruned
        if kv_heads < num_heads:
            group_size = num_heads // kv_heads
            kv_head = hi // group_size
            group_heads = list(range(kv_head * group_size, (kv_head + 1) * group_size))
            group_pruned = all(h in heads_pruned.get(li, []) for h in group_heads)
            if group_pruned:
                kv_s = kv_head * head_dim
                kv_e = kv_s + head_dim
                for pname in ["k_proj", "v_proj"]:
                    proj = getattr(attn, pname, None)
                    if proj and hasattr(proj, "weight"):
                        with torch.no_grad():
                            proj.weight.data[kv_s:kv_e, :] = 0

    zeroed = sum((p == 0).sum().item() for p in model.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Pruned {num_prune}/{total} heads → {zeroed/total_p*100:.1f}% weights zeroed")
    return heads_pruned


def train_with_lora(model, train_loader, eval_loader, steps=1000, lr=2e-4):
    """Train using LoRA adapters (memory-efficient)."""
    from peft import LoraConfig, get_peft_model, TaskType

    # Prepare LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )

    step, total_loss = 0, 0
    t0 = time.time()

    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            ids = batch["input_ids"].to(model.device)
            mask = batch["attention_mask"].to(model.device)

            out = model(input_ids=ids, attention_mask=mask, labels=ids)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += out.loss.item()
            step += 1

            if step % 100 == 0 or step == steps:
                elapsed = time.time() - t0
                print(f"  Step {step}/{steps} — loss: {total_loss/step:.4f}, "
                      f"{step/elapsed:.1f} it/s, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Merge LoRA weights back into base model
    model = model.merge_and_unload()
    print(f"  LoRA merged back into base model")
    return model


def generate_samples(model, tokenizer):
    """Generate proof-of-quality samples."""
    model.eval()
    prompts = {
        "reasoning": "Let me think step by step about how to solve this problem:",
        "code": "def fibonacci(n: int) -> list[int]:\n    \"\"\"Return first n Fibonacci numbers.\"\"\"",
        "science": "The relationship between quantum mechanics and general relativity",
        "creative": "In the year 2045, the first truly conscious AI",
    }
    samples = {}
    for name, prompt in prompts.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                do_sample=True, top_p=0.9, repetition_penalty=1.1,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        samples[name] = f"Prompt: {prompt}\n\nGenerated:\n{text}"
        print(f"  [{name}] {text[:80]}...")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Forge a model with experiential plasticity")
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prune-level", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--early-stop", type=float, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    slug = args.model.split("/")[-1].lower()
    out = Path(args.output_dir or f"output/forged/{slug}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "benchmark").mkdir(exist_ok=True)

    # Pre-flight memory check
    params = get_model_params(args.model)
    mem = estimate_memory(params, args.load_in_4bit)
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n{'='*60}")
    print(f"  FORGING: {args.model} ({mem['params_b']:.1f}B params)")
    print(f"  Hardware: {torch.cuda.get_device_name(0)}, {vram_gb:.0f}GB VRAM")
    print(f"  Memory: model={mem['model_gb']:.1f}GB + LoRA+optim={mem['optimizer_gb']:.1f}GB")
    print(f"  Estimated total: {mem['total_training_gb']:.1f}GB (available: {vram_gb:.0f}GB)")
    
    if mem['total_training_gb'] > vram_gb * 0.95:
        if not args.load_in_4bit:
            print(f"\n  WARNING: Tight fit. Consider --load-in-4bit")
    
    print(f"  Cycles: {args.cycles}, Steps: {args.steps}, Prune: {args.prune_level:.0%}")
    print(f"{'='*60}\n")

    # 1. Load
    print("[1] Loading model...")
    model, tokenizer = load_model(args.model, args.load_in_4bit)
    train_loader, eval_loader = make_dataloader(
        tokenizer, args.max_length, args.batch_size, args.max_samples
    )

    # 2. Baseline
    print("[2] Evaluating baseline...")
    baseline = evaluate(model, eval_loader)
    print(f"  Baseline: perplexity={baseline['perplexity']:.2f}")

    # 3. Forge cycles
    cycle_results = []
    for cycle in range(1, args.cycles + 1):
        print(f"\n[3.{cycle}] Cycle {cycle}/{args.cycles}")

        # Prune
        cycle_prune = args.prune_level / args.cycles
        pruned = prune_heads_by_zeroing(model, cycle_prune)
        post_prune = evaluate(model, eval_loader)
        print(f"  After prune: perplexity={post_prune['perplexity']:.2f}")

        # Train with LoRA
        print(f"  Training {args.steps} steps with LoRA...")
        model = train_with_lora(model, train_loader, eval_loader, args.steps, args.lr)
        
        post_train = evaluate(model, eval_loader)
        improvement = (baseline['perplexity'] - post_train['perplexity']) / baseline['perplexity'] * 100
        print(f"  After train: perplexity={post_train['perplexity']:.2f} ({improvement:+.1f}% vs baseline)")

        cycle_results.append({
            "cycle": cycle,
            "post_prune_ppl": round(post_prune["perplexity"], 4),
            "post_train_ppl": round(post_train["perplexity"], 4),
        })

        # Convergence detection
        if args.early_stop and cycle >= 2:
            delta = abs(cycle_results[-2]["post_train_ppl"] - post_train["perplexity"])
            if delta / cycle_results[-2]["post_train_ppl"] * 100 < args.early_stop:
                print(f"  Converged. Stopping.")
                break

    # 4. Final
    final = evaluate(model, eval_loader)
    total_imp = (baseline['perplexity'] - final['perplexity']) / baseline['perplexity'] * 100

    print(f"\n{'='*60}")
    print(f"  {args.model}: {baseline['perplexity']:.2f} → {final['perplexity']:.2f} ({total_imp:+.1f}%)")
    print(f"{'='*60}")

    # 5. Generate samples
    print("\n[4] Generating output samples...")
    samples = generate_samples(model, tokenizer)
    for name, text in samples.items():
        (out / "benchmark" / f"{name}.txt").write_text(text)

    # 6. Save
    print("\n[5] Saving model...")
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"  Saved to {model_dir}")

    # 7. Results
    results = {
        "model": args.model,
        "strategy": "combined",
        "pruning_level": args.prune_level,
        "cycles": len(cycle_results),
        "training_steps": args.steps,
        "baseline_ppl": round(baseline["perplexity"], 4),
        "final_ppl": round(final["perplexity"], 4),
        "improvement_pct": round(total_imp, 2),
        "forged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": torch.cuda.get_device_name(0),
        "load_in_4bit": args.load_in_4bit,
        "training_method": "LoRA (r=16, alpha=32)",
        "cycle_results": cycle_results,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone. {out / 'results.json'}")


if __name__ == "__main__":
    main()
