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
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Flush all output immediately — no buffering, ever
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

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
                pruning_method="forward_hooks",  # hooks everywhere — zeroing causes NaN
            )
        else:
            # Tier A: comfortable
            return ForgeConfig(
                tier="A", load_4bit=False,
                batch_size=4, seq_len=256, grad_accum_steps=2,
                lora_r=16, lora_alpha=32, use_8bit_optim=False,
                pruning_method="forward_hooks",  # hooks everywhere — zeroing causes NaN
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

def load_model(model_name: str, load_4bit: bool, free_cache_after_load: bool = False):
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

    # For large models: delete HF cache after loading to free disk for saving later
    if free_cache_after_load:
        import shutil
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache = model_name.replace("/", "--")
        for d in cache_dir.glob(f"models--{model_cache}*"):
            size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(d)
            print(f"  Freed {size_gb:.1f}GB from HF cache ({d.name})")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Domain-driven data — the domain determines EVERYTHING about training
# ---------------------------------------------------------------------------

# Each domain: (dataset_id, config, text_column, train_split, val_split)
DOMAIN_DATASETS = {
    "code": {
        "dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
        "config": None,
        "text_col": "query",
        "answer_col": "answer",
        "train_split": "train[:5000]",
        "val_split": "train[5000:5200]",
    },
    "reasoning": {
        "dataset": "gsm8k",
        "config": "main",
        "text_col": "question",  # Will concat question + answer
        "answer_col": "answer",
        "train_split": "train[:5000]",
        "val_split": "test[:200]",
    },
    "general": {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "text_col": "text",
        "train_split": "train[:5000]",
        "val_split": "validation[:200]",
    },
    "chat": {
        "dataset": "stingning/ultrachat",
        "config": None,
        "text_col": "data",  # needs special handling — list of turns
        "train_split": "train[:5000]",
        "val_split": "train[5000:5200]",
    },
    "science": {
        "dataset": "scientific_papers",
        "config": "arxiv",
        "text_col": "article",
        "train_split": "train[:5000]",
        "val_split": "validation[:200]",
    },
}

# Domain-specific generation prompts — must actually challenge the model
DOMAIN_PROMPTS = {
    "code": {
        "concurrent_system": (
            "Implement a lock-free concurrent hash map in Python using atomics. "
            "Handle resize operations without blocking readers. Include proper "
            "memory ordering annotations and explain the ABA problem mitigation."
        ),
        "system_design": (
            "Design a distributed rate limiter that works across multiple servers "
            "without a central coordinator. Use a sliding window algorithm. "
            "Handle clock skew between nodes. Show the implementation."
        ),
        "debugging": (
            "This async Python server has a memory leak that only manifests under "
            "high concurrency. The leak grows at ~50MB/hour. Identify likely causes "
            "and write diagnostic code to find the exact source."
        ),
        "refactoring": (
            "Refactor this 500-line God class into a proper domain model using "
            "the repository pattern, dependency injection, and event sourcing. "
            "Show the key interfaces and one concrete implementation."
        ),
    },
    "reasoning": {
        "multi_step": (
            "A company has 3 departments. Engineering has twice as many people as Sales. "
            "Marketing has 15 fewer people than Engineering. The total headcount is 285. "
            "Each engineer costs $150K/year, each salesperson $120K, each marketer $95K. "
            "What is the total annual salary budget? Show every step."
        ),
        "logic": (
            "Five houses in a row are painted different colors. Each owner has a different "
            "pet, drink, and hobby. Given: The red house is immediately left of the white house. "
            "The dog owner drinks tea. The painter lives in the yellow house. "
            "The horse owner is next to the painter. Work through this systematically."
        ),
        "optimization": (
            "You have 12 servers, each with different CPU and memory specs. You need to "
            "schedule 50 jobs with varying resource requirements to minimize total completion "
            "time. Describe the algorithm, prove its approximation ratio, and show edge cases."
        ),
        "proof": (
            "Prove that for any continuous function f: [0,1] -> [0,1], there exists a "
            "fixed point x such that f(x) = x. Then generalize to higher dimensions "
            "and explain why this matters for neural network convergence."
        ),
    },
    "general": {
        "analysis": (
            "Analyze the economic implications of widespread adoption of local AI models "
            "running on consumer hardware. Consider impacts on cloud providers, data privacy "
            "regulations, employment in AI services, and the democratization of intelligence. "
            "Use specific examples and data points."
        ),
        "synthesis": (
            "Compare the governance structures of the EU AI Act, China's AI regulations, "
            "and the US executive order on AI. Identify the fundamental philosophical "
            "differences, practical enforcement challenges, and predict convergence or "
            "divergence over the next decade."
        ),
        "technical_writing": (
            "Write a technical blog post explaining how attention head pruning with "
            "experiential plasticity can make large language models smaller AND better. "
            "Target audience: ML engineers who know transformers but not pruning. "
            "Include analogies to biological neural development."
        ),
        "nuanced_opinion": (
            "Make a balanced argument for and against open-sourcing frontier AI models. "
            "Address safety concerns, innovation velocity, competitive dynamics between "
            "nations, and the specific case of models capable of autonomous code execution. "
            "Don't hedge — take clear positions on each sub-question."
        ),
    },
    "science": {
        "hypothesis": (
            "Given recent observations of anomalous galaxy rotation curves that deviate "
            "from both MOND and standard dark matter predictions, propose a testable "
            "hypothesis that could explain the discrepancy. Include the mathematical "
            "framework and specific observational tests."
        ),
        "methodology": (
            "Design an experiment to determine whether transformer attention heads "
            "develop specialized functions analogous to cortical columns in the brain. "
            "Define your metrics, controls, and statistical tests. Address the challenge "
            "of polysemantic neurons."
        ),
    },
    "chat": {
        "complex_request": (
            "I'm building a home automation system and my Zigbee mesh keeps dropping "
            "devices when more than 20 are connected. I've tried 3 different coordinators. "
            "The network map shows some devices routing through others that are unreliable. "
            "How do I fix this without replacing all my devices?"
        ),
        "emotional_intelligence": (
            "My team lead just told me my code review style is 'too aggressive' and "
            "it's making junior developers afraid to submit PRs. I genuinely believe "
            "I'm maintaining quality standards. How do I adjust without lowering the bar?"
        ),
    },
}


def make_dataloaders(tokenizer, cfg: ForgeConfig, domain: str, max_samples=2000):
    """Load domain-appropriate dataset."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    if domain not in DOMAIN_DATASETS:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(DOMAIN_DATASETS.keys())}")

    ds_cfg = DOMAIN_DATASETS[domain]
    dataset_id = ds_cfg["dataset"]
    config = ds_cfg["config"]
    text_col = ds_cfg["text_col"]

    # Tier C (large models) gets fewer val samples — each eval is expensive
    val_size = 50 if cfg.tier == "C" else 200
    print(f"  Dataset: {dataset_id}" + (f" ({config})" if config else ""))
    print(f"  Val samples: {val_size} (tier {cfg.tier})")

    load_kwargs = {"split": ds_cfg["train_split"]}
    val_split = ds_cfg["val_split"].replace("200", str(val_size))
    val_kwargs = {"split": val_split}
    if config:
        load_kwargs["name"] = config
        val_kwargs["name"] = config

    try:
        train = load_dataset(dataset_id, **load_kwargs)
        val = load_dataset(dataset_id, **val_kwargs)
    except Exception as e:
        print(f"  Failed to load {dataset_id}: {e}")
        print(f"  Falling back to wikitext-2 (NOT IDEAL — fix the dataset config)")
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{max_samples}]")
        val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:200]")
        text_col = "text"

    # Handle special cases
    answer_col = ds_cfg.get("answer_col")

    def tok_fn(examples):
        texts = examples[text_col]
        # For Q&A datasets, concatenate question + answer
        if answer_col and answer_col in examples:
            texts = [f"{q}\n{a}" for q, a in zip(texts, examples[answer_col])]
        # For chat datasets with turn lists
        if isinstance(texts[0], list):
            texts = ["\n".join(str(t) for t in turns) for turns in texts]
        return tokenizer(texts, truncation=True,
                        max_length=cfg.seq_len, padding="max_length",
                        return_tensors="pt")

    cols_to_remove = [c for c in train.column_names if c != "input_ids" and c != "attention_mask"]
    train = train.filter(lambda x: len(str(x[text_col]).strip()) > 20)
    val = val.filter(lambda x: len(str(x[text_col]).strip()) > 20)
    train = train.map(tok_fn, batched=True, remove_columns=cols_to_remove)
    val = val.map(tok_fn, batched=True, remove_columns=cols_to_remove)
    train.set_format("torch")
    val.set_format("torch")

    print(f"  Train: {len(train)} samples, Val: {len(val)} samples")

    return (DataLoader(train, batch_size=cfg.batch_size, shuffle=True),
            DataLoader(val, batch_size=cfg.batch_size))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def write_status(output_dir: Path, phase: str, detail: str = "", **extra):
    """Write status.json — any external process can monitor progress."""
    status = {
        "phase": phase,
        "detail": detail,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }
    status_path = output_dir / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status))


@torch.no_grad()
def evaluate(model, eval_loader, output_dir: Path = None, label: str = "eval"):
    """Perplexity with eval-batch=1 to minimize VRAM spike from 248K logits."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    n_batches = len(eval_loader)
    # For 4-bit models with device_map="auto", get the actual device
    device = next(model.parameters()).device
    for bi, batch in enumerate(eval_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        for i in range(ids.shape[0]):
            out = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=ids[i:i+1])
            loss_val = out.loss.float().item()  # force fp32 for accuracy
            n = (mask[i:i+1] > 0).sum().item()
            total_loss += loss_val * n
            total_tokens += n
            # Debug first batch
            if bi == 0 and i == 0:
                print(f"  [eval debug] loss={loss_val:.4f}, tokens={n}, device={device}")
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            ppl_so_far = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
            print(f"  [{label}] {bi+1}/{n_batches} batches, ppl={ppl_so_far:.2f}")
            if output_dir:
                write_status(output_dir, label, f"batch {bi+1}/{n_batches}",
                            perplexity=round(ppl_so_far, 2), batch=bi+1, total_batches=n_batches)
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

def train_lora(model, train_loader, cfg: ForgeConfig, steps=1000, lr=5e-5, output_dir: Path = None):
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
    step, total_loss = 0, 0.0
    t0 = time.time()
    scaler = torch.amp.GradScaler("cuda")  # Handles fp16 gradient scaling

    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            ids = batch["input_ids"].to(model.device)
            mask = batch["attention_mask"].to(model.device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=mask, labels=ids)
                loss = out.loss / ga

            scaler.scale(loss).backward()

            if (step + 1) % ga == 0 or step == steps - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = out.loss.item()
            total_loss += loss_val if math.isfinite(loss_val) else 0
            step += 1

            # Progress every 10 steps — never go blind
            if step % 10 == 0 or step == 1 or step == steps:
                elapsed = time.time() - t0
                it_s = step / elapsed if elapsed > 0 else 0
                eta = (steps - step) / it_s if it_s > 0 else 0
                vram = torch.cuda.memory_allocated() / 1e9
                avg_loss = total_loss / step
                print(f"  [{step}/{steps}] loss={avg_loss:.4f} | {it_s:.1f} it/s | "
                      f"ETA {eta:.0f}s | VRAM {vram:.1f}GB")

                if output_dir:
                    write_status(output_dir, "training",
                                f"Step {step}/{steps}, loss={avg_loss:.4f}",
                                step=step, total_steps=steps,
                                loss=round(avg_loss, 4),
                                it_per_sec=round(it_s, 2),
                                eta_seconds=round(eta))

            # Inference sample every 200 steps — PROVE the model is learning
            if (step % 200 == 0 or step == steps) and output_dir:
                model.eval()
                try:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(model.config._name_or_path)
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    prompt = "def binary_search(arr, target):"
                    inputs = tok(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out_ids = model.generate(**inputs, max_new_tokens=80,
                                                temperature=0.7, do_sample=True, top_p=0.9)
                    sample = tok.decode(out_ids[0], skip_special_tokens=True)
                    print(f"  [SAMPLE@{step}] {sample[:120]}...")
                    write_status(output_dir, "training_sample",
                                f"Step {step} sample",
                                step=step, sample=sample[:300])
                except Exception as e:
                    print(f"  [SAMPLE@{step}] failed: {e}")
                model.train()

    # Merge LoRA back
    model = model.merge_and_unload()
    print(f"  LoRA merged into base model")
    torch.cuda.empty_cache()
    gc.collect()
    return model


# ---------------------------------------------------------------------------
# Generation samples
# ---------------------------------------------------------------------------

def generate_samples(model, tokenizer, domain: str):
    """Domain-appropriate output samples that actually challenge the model."""
    model.eval()
    prompts = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
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
    parser.add_argument("--lr", type=float, default=5e-5)
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
    write_status(out, "loading", f"Loading {args.model}")
    print("[1] Loading model...")
    model, tokenizer = load_model(args.model, cfg.load_4bit,
                                    free_cache_after_load=(cfg.tier == "C"))
    write_status(out, "loading_data", f"Loading {args.domain} dataset")
    train_loader, eval_loader = make_dataloaders(tokenizer, cfg, args.domain, args.max_samples)

    # --- 2. Baseline ---
    write_status(out, "baseline_eval", "Evaluating baseline perplexity")
    print("[2] Evaluating baseline...")
    baseline = evaluate(model, eval_loader, out, "baseline")
    print(f"  Baseline: perplexity={baseline['perplexity']:.2f}")
    write_status(out, "baseline_done", f"Baseline: {baseline['perplexity']:.2f}",
                perplexity=round(baseline['perplexity'], 2))
    check_vram("baseline")
    torch.cuda.empty_cache()

    # --- 3. Forge cycles ---
    cycle_results = []
    all_hooks = []

    for cycle in range(1, args.cycles + 1):
        print(f"\n[3.{cycle}] Cycle {cycle}/{args.cycles}")

        # TRAIN first — strengthen the model on domain data
        write_status(out, "training", f"Cycle {cycle}: LoRA training {args.steps} steps",
                    cycle=cycle, total_cycles=args.cycles, total_steps=args.steps)
        print(f"  Training {args.steps} steps (LoRA)...")
        model = train_lora(model, train_loader, cfg, args.steps, args.lr, out)

        write_status(out, "post_train_eval", f"Cycle {cycle}: evaluating after training",
                    cycle=cycle)
        post_train = evaluate(model, eval_loader, out, f"post-train-c{cycle}")
        print(f"  After train: perplexity={post_train['perplexity']:.2f}")
        check_vram("post-train")

        # THEN prune — remove heads that didn't contribute during training
        write_status(out, "pruning", f"Cycle {cycle}: pruning heads after training",
                    cycle=cycle)
        cycle_prune = args.prune_level / args.cycles
        heads, hooks = prune(model, cycle_prune, info, cfg.pruning_method)
        all_hooks.extend(hooks)

        write_status(out, "post_prune_eval", f"Cycle {cycle}: evaluating after prune",
                    cycle=cycle)
        post_prune = evaluate(model, eval_loader, out, f"post-prune-c{cycle}")
        imp = (baseline["perplexity"] - post_prune["perplexity"]) / baseline["perplexity"] * 100
        print(f"  After prune: perplexity={post_prune['perplexity']:.2f} ({imp:+.1f}% vs baseline)")
        write_status(out, "cycle_done", f"Cycle {cycle}: {imp:+.1f}% vs baseline",
                    cycle=cycle, perplexity=round(post_prune['perplexity'], 2),
                    improvement_pct=round(imp, 2))
        check_vram("post-prune")

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
    samples = generate_samples(model, tokenizer, args.domain)
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
        "training_data": DOMAIN_DATASETS.get(args.domain, {}).get("dataset", "unknown"),
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
