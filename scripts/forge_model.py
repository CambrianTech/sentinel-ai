#!/usr/bin/env python3
"""
forge_model.py v3 — Memory-architected forging for Qwen3.5 on consumer hardware.

Three tiers based on model-to-VRAM ratio:
  Tier A (model <= 40% VRAM): fp16, batch=4, comfortable
  Tier B (model <= 70% VRAM): fp16, batch=1-2, grad accum
  Tier C (model > VRAM):      4-bit NF4, batch=1, hook-based pruning

Usage:
    python forge_model.py Qwen/Qwen3.5-4B --domain general
    python forge_model.py Qwen/Qwen3.5-9B --domain code
    python forge_model.py Qwen/Qwen3.5-27B --domain general   # auto-detects 4-bit needed
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ForgeConfig:
    """Memory-aware configuration — computed from model size and hardware."""
    tier: str                    # A, B, or C
    load_4bit: bool
    batch_size: int
    seq_len: int
    grad_accum_steps: int
    lora_r: int
    lora_alpha: int
    use_8bit_optim: bool
    pruning_method: str          # "zero_weights" or "forward_hooks"

    @staticmethod
    def auto(model_fp16_gb: float, vram_gb: float, user_4bit: bool = False) -> "ForgeConfig":
        """Select tier based on model size vs available VRAM."""
        ratio = model_fp16_gb / vram_gb

        if user_4bit or ratio > 0.70:
            # Tier C: must use 4-bit
            return ForgeConfig(
                tier="C", load_4bit=True,
                batch_size=1, seq_len=256, grad_accum_steps=8,
                lora_r=16, lora_alpha=32, use_8bit_optim=True,
                pruning_method="forward_hooks",
            )
        elif ratio > 0.40:
            # Tier B: fp16 fits but tight
            return ForgeConfig(
                tier="B", load_4bit=False,
                batch_size=2, seq_len=256, grad_accum_steps=4,
                lora_r=16, lora_alpha=32, use_8bit_optim=False,
                pruning_method="zero_weights",
            )
        else:
            # Tier A: comfortable
            return ForgeConfig(
                tier="A", load_4bit=False,
                batch_size=4, seq_len=256, grad_accum_steps=2,
                lora_r=16, lora_alpha=32, use_8bit_optim=False,
                pruning_method="zero_weights",
            )


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def nested_config(config):
    """Get text config from possibly nested VLM config (Qwen3.5)."""
    return getattr(config, "text_config", config)


def get_model_info(model_name: str) -> dict:
    """Architecture info with accurate parameter count."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    tc = nested_config(config)

    h = getattr(tc, "hidden_size", 768)
    n = getattr(tc, "num_hidden_layers", 12)
    nh = getattr(tc, "num_attention_heads", 12)
    nkv = getattr(tc, "num_key_value_heads", nh)
    v = getattr(tc, "vocab_size", getattr(config, "vocab_size", 32000))
    inter = getattr(tc, "intermediate_size", 4 * h)
    head_dim = h // nh
    tied = getattr(tc, "tie_word_embeddings", getattr(config, "tie_word_embeddings", True))

    # Accurate param count
    embed = v * h
    per_layer_attn = h * (nh * head_dim) + h * (nkv * head_dim) * 2 + (nh * head_dim) * h  # Q + K + V + O
    per_layer_mlp = h * inter * 3  # gate + up + down (SwiGLU)
    per_layer = per_layer_attn + per_layer_mlp
    lm_head = 0 if tied else v * h
    total_params = embed + n * per_layer + lm_head

    # LoRA param count for q/k/v/o_proj at rank r
    def lora_params(r):
        per_layer_lora = (
            (h * r + r * nh * head_dim) +    # q_proj
            (h * r + r * nkv * head_dim) +   # k_proj
            (h * r + r * nkv * head_dim) +   # v_proj
            (nh * head_dim * r + r * h)      # o_proj
        )
        return n * per_layer_lora

    return {
        "num_layers": n, "num_heads": nh, "num_kv_heads": nkv,
        "hidden_size": h, "head_dim": head_dim, "vocab_size": v,
        "intermediate_size": inter, "tied_embeddings": tied,
        "total_params": total_params,
        "fp16_gb": total_params * 2 / 1e9,
        "q4_gb": total_params * 0.5 / 1e9,
        "lora_params_r16": lora_params(16),
    }


def get_layers(model):
    """Find the transformer layer list for any architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find model layers. Unsupported architecture.")


def check_vram(label: str):
    """Log VRAM usage."""
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  [{label}] VRAM: {used:.1f}/{total:.0f}GB ({used/total:.0%})")
    return used


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, load_4bit: bool):
    """Load model with explicit memory strategy."""
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
        print(f"  Loading 4-bit NF4 (double quant)")
    else:
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = "auto"
        print(f"  Loading fp16")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    check_vram("after load")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_dataloaders(tokenizer, cfg: ForgeConfig, max_samples=2000):
    """Load limited dataset with config-driven batch size."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    train = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{max_samples}]")
    val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:200]")

    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True,
                        max_length=cfg.seq_len, padding="max_length",
                        return_tensors="pt")

    train = train.filter(lambda x: len(x["text"].strip()) > 20)
    val = val.filter(lambda x: len(x["text"].strip()) > 20)
    train = train.map(tok_fn, batched=True, remove_columns=["text"])
    val = val.map(tok_fn, batched=True, remove_columns=["text"])
    train.set_format("torch")
    val.set_format("torch")

    return (DataLoader(train, batch_size=cfg.batch_size, shuffle=True),
            DataLoader(val, batch_size=cfg.batch_size))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, eval_loader):
    """Perplexity with eval-batch=1 to minimize VRAM spike from 248K logits."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in eval_loader:
        ids = batch["input_ids"].to(model.device)
        mask = batch["attention_mask"].to(model.device)
        # Process one sample at a time to avoid huge logit tensors
        for i in range(ids.shape[0]):
            out = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=ids[i:i+1])
            n = (mask[i:i+1] > 0).sum().item()
            total_loss += out.loss.item() * n
            total_tokens += n
    avg = total_loss / max(total_tokens, 1)
    return {"loss": avg, "perplexity": torch.exp(torch.tensor(avg)).item()}


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def compute_head_importance(model, info: dict):
    """L2 norm of Q-projection weights per head (works on fp16 and 4-bit)."""
    layers = get_layers(model)
    n_layers = info["num_layers"]
    n_heads = info["num_heads"]
    head_dim = info["head_dim"]
    importance = torch.zeros(n_layers, n_heads)

    for li in range(n_layers):
        attn = getattr(layers[li], "self_attn", getattr(layers[li], "attn", None))
        if attn is None:
            continue
        q = getattr(attn, "q_proj", None)
        if q is None:
            continue
        # For 4-bit models, .weight gives dequantized view — that's fine for importance
        try:
            w = q.weight.data.float()
        except Exception:
            # bitsandbytes may need special handling
            import bitsandbytes as bnb
            w = bnb.functional.dequantize_4bit(q.weight.data, q.weight.quant_state).float()
        for hi in range(n_heads):
            s, e = hi * head_dim, (hi + 1) * head_dim
            if e <= w.shape[0]:
                importance[li, hi] = w[s:e].norm().item()

    return importance


def select_heads_to_prune(importance, prune_percent):
    """Select lowest-importance heads."""
    n_layers, n_heads = importance.shape
    total = n_layers * n_heads
    n_prune = int(total * prune_percent)

    flat = importance.flatten()
    _, indices = flat.sort()
    prune_indices = indices[:n_prune]

    heads = {}
    for idx in prune_indices:
        li = idx.item() // n_heads
        hi = idx.item() % n_heads
        heads.setdefault(li, []).append(hi)
    return heads, n_prune


def prune_by_zeroing(model, heads_to_prune, info):
    """Zero out weight slices — for fp16 models (Tier A/B)."""
    layers = get_layers(model)
    tc = nested_config(model.config)
    head_dim = info["head_dim"]
    n_heads = info["num_heads"]
    n_kv = info["num_kv_heads"]

    for li, head_list in heads_to_prune.items():
        attn = getattr(layers[li], "self_attn", getattr(layers[li], "attn", None))
        if attn is None:
            continue
        for hi in head_list:
            s, e = hi * head_dim, (hi + 1) * head_dim
            for pname in ["q_proj", "o_proj"]:
                proj = getattr(attn, pname, None)
                if proj and hasattr(proj, "weight"):
                    with torch.no_grad():
                        if pname == "o_proj":
                            proj.weight.data[:, s:e] = 0
                        else:
                            proj.weight.data[s:e, :] = 0

            # GQA: only zero K/V if entire group is pruned
            if n_kv < n_heads:
                group_size = n_heads // n_kv
                kv_head = hi // group_size
                group = range(kv_head * group_size, (kv_head + 1) * group_size)
                if all(h in heads_to_prune.get(li, []) for h in group):
                    kv_s, kv_e = kv_head * head_dim, (kv_head + 1) * head_dim
                    for pname in ["k_proj", "v_proj"]:
                        proj = getattr(attn, pname, None)
                        if proj and hasattr(proj, "weight"):
                            with torch.no_grad():
                                proj.weight.data[kv_s:kv_e, :] = 0


def prune_by_hooks(model, heads_to_prune, info):
    """Install forward hooks that mask pruned head outputs — for 4-bit models (Tier C)."""
    layers = get_layers(model)
    head_dim = info["head_dim"]
    hooks = []

    for li, head_list in heads_to_prune.items():
        attn = getattr(layers[li], "self_attn", getattr(layers[li], "attn", None))
        if attn is None:
            continue
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None:
            continue

        def make_hook(pruned_heads, hd):
            def hook_fn(module, inp, output):
                for hi in pruned_heads:
                    s, e = hi * hd, (hi + 1) * hd
                    if isinstance(inp, tuple):
                        # o_proj input is the concatenated head outputs
                        # Zero the input to o_proj for pruned heads
                        x = inp[0]
                        x[:, :, s:e] = 0
                return output
            return hook_fn

        h = o_proj.register_forward_pre_hook(make_hook(head_list, head_dim))
        hooks.append(h)

    return hooks


def prune(model, prune_percent, info, method="zero_weights"):
    """Dispatch to the right pruning strategy."""
    importance = compute_head_importance(model, info)
    heads, n_pruned = select_heads_to_prune(importance, prune_percent)
    total = info["num_layers"] * info["num_heads"]

    hooks = []
    if method == "forward_hooks":
        hooks = prune_by_hooks(model, heads, info)
    else:
        prune_by_zeroing(model, heads, info)

    print(f"  Pruned {n_pruned}/{total} heads ({method})")
    return heads, hooks


# ---------------------------------------------------------------------------
# Training with LoRA
# ---------------------------------------------------------------------------

def train_lora(model, train_loader, cfg: ForgeConfig, steps=1000, lr=2e-4):
    """LoRA training with gradient accumulation and proper checkpointing."""
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable:,} trainable ({trainable/total*100:.2f}%)")

    model.train()
    # use_reentrant=False is CRITICAL for LoRA + gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.use_8bit_optim:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01)
        print(f"  Optimizer: AdamW 8-bit")
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    ga = cfg.grad_accum_steps
    step, accum_loss, total_loss = 0, 0.0, 0.0
    t0 = time.time()

    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            ids = batch["input_ids"].to(model.device)
            mask = batch["attention_mask"].to(model.device)

            out = model(input_ids=ids, attention_mask=mask, labels=ids)
            loss = out.loss / ga
            loss.backward()
            accum_loss += out.loss.item()

            if (step + 1) % ga == 0 or step == steps - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += out.loss.item()
            step += 1

            if step % 100 == 0 or step == steps:
                elapsed = time.time() - t0
                print(f"  Step {step}/{steps} — loss: {total_loss/step:.4f}, "
                      f"{step/elapsed:.1f} it/s, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Merge LoRA back
    model = model.merge_and_unload()
    print(f"  LoRA merged into base model")
    torch.cuda.empty_cache()
    gc.collect()
    return model


# ---------------------------------------------------------------------------
# Generation samples
# ---------------------------------------------------------------------------

def generate_samples(model, tokenizer):
    """Proof-of-quality output samples — prompts that actually challenge the model."""
    model.eval()
    prompts = {
        "reasoning": (
            "A farmer has 3 fields. Field A produces 40% more wheat than Field B. "
            "Field C produces half of what A and B produce combined. "
            "If Field B produces 500kg, how much total wheat is produced? "
            "Show your reasoning step by step."
        ),
        "code": (
            "Write a Python function that implements a thread-safe LRU cache with TTL "
            "(time-to-live) expiration. It should handle concurrent reads and writes, "
            "evict expired entries lazily, and support a max_size parameter. "
            "Include type hints and docstring."
        ),
        "analysis": (
            "Compare the architectural trade-offs between transformer-based and "
            "state-space models (like Mamba) for long-context inference. Consider "
            "memory complexity, training efficiency, and real-world deployment "
            "on consumer hardware with 32GB VRAM."
        ),
        "agentic": (
            "You are an AI assistant with access to a file system and a web browser. "
            "A user asks: 'Find all Python files in my project that import requests "
            "but don't handle connection timeouts, then suggest fixes.' "
            "Describe your step-by-step approach and what tools you'd use."
        ),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Forge a model with experiential plasticity")
    parser.add_argument("model", help="HuggingFace model ID (e.g., Qwen/Qwen3.5-4B)")
    parser.add_argument("--domain", type=str, required=True,
                       help="Training domain: general, code, reasoning, chat, science")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prune-level", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--early-stop", type=float, default=None,
                       help="Stop if per-cycle improvement < this %%")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Force 4-bit loading (auto-detected if model too large)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    slug = args.model.split("/")[-1].lower()
    out = Path(args.output_dir or f"output/forged/{slug}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "benchmark").mkdir(exist_ok=True)

    # --- Pre-flight ---
    info = get_model_info(args.model)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cfg = ForgeConfig.auto(info["fp16_gb"], vram_gb, args.load_in_4bit)

    print(f"\n{'='*60}")
    print(f"  FORGING: {args.model}")
    print(f"  Params: {info['total_params']/1e9:.1f}B, fp16: {info['fp16_gb']:.1f}GB, 4-bit: {info['q4_gb']:.1f}GB")
    print(f"  Hardware: {torch.cuda.get_device_name(0)}, {vram_gb:.0f}GB VRAM")
    print(f"  Tier {cfg.tier}: {'4-bit' if cfg.load_4bit else 'fp16'}, batch={cfg.batch_size}, "
          f"accum={cfg.grad_accum_steps}, seq={cfg.seq_len}")
    print(f"  LoRA r={cfg.lora_r}, prune={cfg.pruning_method}")
    print(f"  Domain: {args.domain}, Cycles: {args.cycles}, Steps: {args.steps}")
    print(f"{'='*60}\n")

    # --- 1. Load ---
    print("[1] Loading model...")
    model, tokenizer = load_model(args.model, cfg.load_4bit)
    train_loader, eval_loader = make_dataloaders(tokenizer, cfg, args.max_samples)

    # --- 2. Baseline ---
    print("[2] Evaluating baseline...")
    baseline = evaluate(model, eval_loader)
    print(f"  Baseline: perplexity={baseline['perplexity']:.2f}")
    check_vram("baseline")
    torch.cuda.empty_cache()

    # --- 3. Forge cycles ---
    cycle_results = []
    all_hooks = []

    for cycle in range(1, args.cycles + 1):
        print(f"\n[3.{cycle}] Cycle {cycle}/{args.cycles}")

        cycle_prune = args.prune_level / args.cycles
        heads, hooks = prune(model, cycle_prune, info, cfg.pruning_method)
        all_hooks.extend(hooks)

        post_prune = evaluate(model, eval_loader)
        print(f"  After prune: perplexity={post_prune['perplexity']:.2f}")
        check_vram("post-prune")

        print(f"  Training {args.steps} steps (LoRA)...")
        model = train_lora(model, train_loader, cfg, args.steps, args.lr)

        post_train = evaluate(model, eval_loader)
        imp = (baseline["perplexity"] - post_train["perplexity"]) / baseline["perplexity"] * 100
        print(f"  After train: perplexity={post_train['perplexity']:.2f} ({imp:+.1f}% vs baseline)")
        check_vram("post-train")

        cycle_results.append({
            "cycle": cycle,
            "post_prune_ppl": round(post_prune["perplexity"], 4),
            "post_train_ppl": round(post_train["perplexity"], 4),
            "improvement_vs_baseline_pct": round(imp, 2),
        })

        # Convergence
        if args.early_stop and cycle >= 2:
            prev = cycle_results[-2]["post_train_ppl"]
            curr = post_train["perplexity"]
            delta_pct = abs(prev - curr) / prev * 100
            if delta_pct < args.early_stop:
                print(f"  Converged ({delta_pct:.3f}% < {args.early_stop}%). Stopping.")
                break

    # Remove any pruning hooks
    for h in all_hooks:
        h.remove()

    # --- 4. Final ---
    final = evaluate(model, eval_loader)
    total_imp = (baseline["perplexity"] - final["perplexity"]) / baseline["perplexity"] * 100

    print(f"\n{'='*60}")
    print(f"  {args.model}: {baseline['perplexity']:.2f} → {final['perplexity']:.2f} ({total_imp:+.1f}%)")
    print(f"{'='*60}")

    # --- 5. Generate samples ---
    print("\n[4] Generating output samples...")
    samples = generate_samples(model, tokenizer)
    for name, text in samples.items():
        (out / "benchmark" / f"{name}.txt").write_text(text)

    # --- 6. Save ---
    print("\n[5] Saving model...")
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"  Saved to {model_dir}")

    # --- 7. Results ---
    results = {
        "model": args.model,
        "domain": args.domain,
        "strategy": "experiential_plasticity",
        "pruning_level": args.prune_level,
        "cycles": len(cycle_results),
        "training_steps": args.steps,
        "baseline_ppl": round(baseline["perplexity"], 4),
        "final_ppl": round(final["perplexity"], 4),
        "improvement_pct": round(total_imp, 2),
        "forged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": torch.cuda.get_device_name(0),
        "tier": cfg.tier,
        "load_4bit": cfg.load_4bit,
        "training_method": f"LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})",
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "seq_len": cfg.seq_len,
        "cycle_results": cycle_results,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone. {out / 'results.json'}")


if __name__ == "__main__":
    main()
