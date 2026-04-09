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


def is_full_attention_layer(config, layer_idx: int) -> bool:
    """True if layer `layer_idx` is a full-attention layer.

    For uniform-architecture models (Qwen2/Qwen2.5/Qwen3 dense/Llama), there
    is no `layer_types` field on the config and EVERY layer is full attention,
    so this returns True unconditionally.

    For hybrid-architecture models (Qwen3.5 with Gated DeltaNet + full attention),
    the config has a `layer_types` list of strings like
    ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]
    and we check the layer's type explicitly. Non-full-attention layers must be
    skipped by all attention-surgery code paths (defrag, importance, pruning).

    This is the Strategy A path from sentinel-ai#163: skip non-full_attention
    layers entirely. Strategy B (per-layer-type compaction) is a separate
    follow-up.
    """
    tc = nested_config(config)
    layer_types = getattr(tc, "layer_types", None)
    if layer_types is None:
        return True  # uniform architecture, every layer is full attention
    if layer_idx >= len(layer_types):
        return True  # safety: layer index out of range, treat as full
    return layer_types[layer_idx] == "full_attention"


def has_hybrid_layers(config) -> bool:
    """True if the model has a non-uniform layer_types list."""
    tc = nested_config(config)
    layer_types = getattr(tc, "layer_types", None)
    if layer_types is None:
        return False
    return len(set(layer_types)) > 1


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

def load_model(
    model_name: str,
    load_4bit: bool,
    free_cache_after_load: bool = False,
    auto_class=None,
):
    """Load model with explicit memory strategy.

    auto_class is the transformers AutoModel class to use for loading.
    Default is AutoModelForCausalLM (the dense LLM case). Family
    adapters can pass their own class (AutoModelForVision2Seq for VL,
    AutoModel for omni-modal, etc.) via family.model_auto_class().
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if auto_class is None:
        auto_class = AutoModelForCausalLM

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
        print(f"  Loading 4-bit NF4 (double quant) via {auto_class.__name__}")
    else:
        kwargs["dtype"] = torch.float16
        # Load to CPU first, then move to CUDA. This avoids sm_120 kernel errors
        # during _init_weights (Mamba-2 A_log init runs torch.uniform_ on CUDA
        # which fails on RTX 5090 with older PyTorch). CPU init always works.
        kwargs["device_map"] = "cpu"
        print(f"  Loading fp16 (CPU → CUDA) via {auto_class.__name__}")

    model = auto_class.from_pretrained(model_name, **kwargs)
    if not load_4bit and str(model.device) == "cpu":
        model = model.to("cuda")
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
    """Perplexity with eval-batch=1 to minimize VRAM spike from 248K logits.

    CRITICAL: labels are masked to -100 at pad positions so the model's
    cross-entropy loss only considers VALID tokens. Without this mask,
    padding-to-max-length 2048 means a 50-token wikitext sample
    contributes ~1998 pad-token losses that dominate the result and
    inflate perplexity by ~30x. (Bug found 2026-04-09 when our published
    qwen2-5-7b-instruct-compacted card showed baseline ppl 263 vs the
    real 8.7 — 30x off because of this exact issue.)
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    n_batches = len(eval_loader)
    # For 4-bit models with device_map="auto", get the actual device
    device = next(model.parameters()).device
    for bi, batch in enumerate(eval_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        for i in range(ids.shape[0]):
            input_ids = ids[i:i+1]
            attention_mask = mask[i:i+1]
            # Mask labels at pad positions so the loss only counts valid tokens.
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_val = out.loss.float().item()  # force fp32 for accuracy
            # n is the number of VALID tokens (the same denominator the
            # model's CE loss used), so the running average matches.
            n = int(attention_mask.sum().item())
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
    """L2 norm of Q-projection weights per head. Handles variable heads per layer after defrag.

    DEPRECATED for selection purposes: see compute_activation_importance() for the
    behaviorally validated metric. This function is retained for backward compatibility
    and for use as a fast proxy when calibration data is unavailable.

    Validation finding (sentinel-ai #155): L2 norm of Q projection is anti-correlated
    with importance for at least Qwen2.5-0.5B. Removing the lowest-L2-norm heads
    produced 9x worse perplexity than removing arbitrary heads.
    """
    layers = get_layers(model)
    n_layers = info["num_layers"]
    n_heads = info["num_heads"]  # Original count — actual may differ after defrag
    importance = torch.full((n_layers, n_heads), float('inf'))

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
            import bitsandbytes as bnb
            w = bnb.functional.dequantize_4bit(q.weight.data, q.weight.quant_state).float()
        # Compute head_dim from actual tensor (may change after defrag)
        actual_heads = min(n_heads, w.shape[0] // max(w.shape[0] // n_heads, 1)) if w.shape[0] >= n_heads else w.shape[0]
        actual_head_dim = w.shape[0] // actual_heads if actual_heads > 0 else 1
        for hi in range(actual_heads):
            s, e = hi * actual_head_dim, (hi + 1) * actual_head_dim
            if e <= w.shape[0]:
                importance[li, hi] = w[s:e].norm().item()

    return importance


# ── Calibration text sets for compute_activation_importance ─────────────────
# DEFAULT (generic English): the v1 set used by every forge run prior to the
# §4.1.3.2 PPL/HumanEval-disconnect finding. These prompts exercise general
# language patterns but not the specific circuits HumanEval problems load.
# Marked as "generic" to make the calibration domain explicit at every call site.
DEFAULT_CALIBRATION_TEXTS_GENERIC = [
    "The quick brown fox jumps over the lazy dog. " * 4,
    "In computer science, a recursive function is one that calls itself.",
    "The capital of France is Paris, located on the Seine river.",
    "Quantum mechanics describes the behavior of matter at atomic scales.",
    "Machine learning models learn patterns from training data.",
    "Climate change refers to long-term shifts in temperature and weather patterns.",
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
    "Shakespeare wrote both tragedies and comedies during the Elizabethan era.",
] * 2  # 16 samples

# CODE-COMPLETION (HumanEval-format, NOT actual HumanEval problems): hand-
# written function-signature + docstring prompts that exercise the same
# circuits HumanEval-style code completion will load. The §4.1.3.2 finding
# is that the activation-magnitude metric overfits to the calibration
# distribution; this set tests whether shifting calibration to be in-domain
# for the held-out task closes the PPL/HumanEval disconnect.
#
# IMPORTANT: these are hand-written, NOT drawn from HumanEval, MBPP, or any
# other benchmark we evaluate against. Using actual HumanEval problems would
# be training on the test set. The function names, docstring patterns, and
# return-value structures are HumanEval-FORMAT but the specific problems are
# original.
DEFAULT_CALIBRATION_TEXTS_CODE = [
    'def reverse_string(s: str) -> str:\n    """Return the reverse of the input string.\n    >>> reverse_string("hello")\n    \'olleh\'\n    """\n    return s[::-1]\n',
    'def is_palindrome(s: str) -> bool:\n    """Return True if s reads the same forwards and backwards.\n    >>> is_palindrome("racecar")\n    True\n    """\n    return s == s[::-1]\n',
    'def sum_of_squares(numbers: list) -> int:\n    """Return the sum of squares of all numbers in the list.\n    >>> sum_of_squares([1, 2, 3])\n    14\n    """\n    return sum(n * n for n in numbers)\n',
    'def count_vowels(text: str) -> int:\n    """Count the vowels (aeiou, case insensitive) in text.\n    >>> count_vowels("Hello World")\n    3\n    """\n    return sum(1 for c in text.lower() if c in "aeiou")\n',
    'def factorial(n: int) -> int:\n    """Compute n! recursively.\n    >>> factorial(5)\n    120\n    """\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n',
    'def filter_even(numbers: list) -> list:\n    """Return only the even numbers from the input list.\n    >>> filter_even([1, 2, 3, 4, 5])\n    [2, 4]\n    """\n    return [n for n in numbers if n % 2 == 0]\n',
    'def find_max(numbers: list) -> int:\n    """Return the largest number in the list.\n    >>> find_max([3, 1, 4, 1, 5, 9, 2, 6])\n    9\n    """\n    return max(numbers)\n',
    'def remove_duplicates(items: list) -> list:\n    """Return the items list with duplicates removed, preserving order.\n    >>> remove_duplicates([1, 2, 2, 3, 1, 4])\n    [1, 2, 3, 4]\n    """\n    seen = set()\n    return [x for x in items if not (x in seen or seen.add(x))]\n',
    'def average(numbers: list) -> float:\n    """Compute the arithmetic mean of the numbers.\n    >>> average([1, 2, 3, 4, 5])\n    3.0\n    """\n    return sum(numbers) / len(numbers) if numbers else 0.0\n',
    'def char_frequency(text: str) -> dict:\n    """Return a dictionary mapping each character to its count in text.\n    >>> char_frequency("aabbc")\n    {\'a\': 2, \'b\': 2, \'c\': 1}\n    """\n    freq = {}\n    for c in text:\n        freq[c] = freq.get(c, 0) + 1\n    return freq\n',
    'def fibonacci_sequence(n: int) -> list:\n    """Return the first n Fibonacci numbers as a list.\n    >>> fibonacci_sequence(6)\n    [0, 1, 1, 2, 3, 5]\n    """\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]\n',
    'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number.\n    >>> is_prime(7)\n    True\n    """\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n',
    'def merge_sorted(a: list, b: list) -> list:\n    """Merge two sorted lists into a single sorted list.\n    >>> merge_sorted([1, 3, 5], [2, 4, 6])\n    [1, 2, 3, 4, 5, 6]\n    """\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    result.extend(a[i:]); result.extend(b[j:])\n    return result\n',
    'def gcd(a: int, b: int) -> int:\n    """Return the greatest common divisor of a and b.\n    >>> gcd(12, 18)\n    6\n    """\n    while b:\n        a, b = b, a % b\n    return a\n',
    'def flatten(nested: list) -> list:\n    """Flatten a list of lists into a single list.\n    >>> flatten([[1, 2], [3, 4], [5]])\n    [1, 2, 3, 4, 5]\n    """\n    return [item for sublist in nested for item in sublist]\n',
    'def word_count(text: str) -> int:\n    """Count the number of whitespace-separated words in text.\n    >>> word_count("hello world foo bar")\n    4\n    """\n    return len(text.split())\n',
]


def get_calibration_texts(source: str) -> list[str]:
    """Return the calibration text set for the requested source.

    Sources:
        "generic": 16 generic English sentences (default, the v1 set)
        "code": 16 hand-written HumanEval-format code completion prompts
                (held-out from any benchmark we evaluate against)
    """
    if source == "generic":
        return DEFAULT_CALIBRATION_TEXTS_GENERIC
    if source == "code":
        return DEFAULT_CALIBRATION_TEXTS_CODE
    raise ValueError(f"unknown calibration source {source!r}; known: generic, code")


def compute_activation_importance(model, tokenizer, info: dict, calibration_texts=None, max_length=128, num_samples=16):
    """Activation-based head importance via forward-hook capture on the O projection.

    For each attention layer, capture the input to o_proj on a small calibration batch.
    The per-head magnitude of that input is a direct measure of how much each head is
    contributing to the residual stream — exactly the quantity that determines whether
    removing the head will degrade output.

    This replaces L2-norm-of-Q ranking, which Layer 4 of the validation harness found
    to be anti-correlated with importance for at least Qwen2.5-0.5B (sentinel-ai #155).

    Args:
        model: live PyTorch model
        tokenizer: matching tokenizer
        info: dict with num_layers, num_heads
        calibration_texts: list of strings to use as calibration data. If None, uses
            a small built-in set. For best results, pass texts representative of the
            target deployment domain.
        max_length: max tokens per calibration sample
        num_samples: number of calibration samples to average

    Returns:
        Tensor of shape (n_layers, n_heads) where higher values = more important.
        Heads with the lowest values are the safest to remove.
    """
    if calibration_texts is None:
        calibration_texts = DEFAULT_CALIBRATION_TEXTS_GENERIC

    layers = get_layers(model)
    n_layers = info["num_layers"]
    n_heads = info["num_heads"]
    device = next(model.parameters()).device

    # Per-layer head importance accumulator
    importance = torch.zeros(n_layers, n_heads)
    counts = torch.zeros(n_layers)

    # Install hooks on every o_proj to capture its INPUT
    # The input to o_proj is the concatenation of per-head attention outputs:
    # shape (batch, seq, num_heads * head_dim). We measure per-head L2 over batch+seq.
    captured = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            x = inputs[0]  # the input to o_proj
            # x.shape: (batch, seq, num_heads * head_dim)
            if x.dim() != 3:
                return
            B, T, D = x.shape
            # Read actual head_dim from the o_proj's input dimension and module's num_heads
            # Use cached attention attribute first, fall back to info
            attn_module = layers[layer_idx]
            attn = getattr(attn_module, "self_attn", getattr(attn_module, "attn", None))
            actual_heads = getattr(attn, "num_heads", n_heads) if attn else n_heads
            if D % actual_heads != 0:
                # Defragged irregularly — just record what we can
                return
            head_dim = D // actual_heads
            # Reshape to (batch, seq, num_heads, head_dim) and compute per-head L2
            xh = x.view(B, T, actual_heads, head_dim)
            # Magnitude per head, averaged across batch and sequence
            mag = xh.float().norm(dim=-1).mean(dim=(0, 1))  # (num_heads,)
            captured.setdefault(layer_idx, []).append(mag.cpu())
        return hook

    # Strategy A: only attach hooks to full_attention layers. For hybrid
    # architectures (Qwen3.5), the linear_attention layers do not have an
    # o_proj in the standard shape and must be skipped. Their importance
    # row stays at zero and gets marked inf below so select_heads_to_prune
    # never picks heads from them.
    config = getattr(model, "config", None)
    handles = []
    skipped_layer_indices = set()
    for li in range(n_layers):
        if config is not None and not is_full_attention_layer(config, li):
            skipped_layer_indices.add(li)
            continue
        attn_module = layers[li]
        attn = getattr(attn_module, "self_attn", getattr(attn_module, "attn", None))
        if attn is None or not hasattr(attn, "o_proj"):
            skipped_layer_indices.add(li)
            continue
        handles.append(attn.o_proj.register_forward_hook(make_hook(li)))

    # Run calibration data through the model
    model.eval()
    with torch.no_grad():
        for text in calibration_texts[:num_samples]:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            if ids["input_ids"].shape[1] < 2:
                continue
            try:
                model(**ids)
            except Exception:
                continue  # skip samples that fail forward

    # Remove hooks
    for h in handles:
        h.remove()

    # Aggregate captured magnitudes
    for li, mags_list in captured.items():
        if not mags_list:
            continue
        # Stack and average across calibration samples
        stacked = torch.stack(mags_list)  # (n_samples, n_heads_at_layer)
        mean_mag = stacked.mean(dim=0)  # (n_heads_at_layer,)
        actual_heads = mean_mag.shape[0]
        for hi in range(min(actual_heads, n_heads)):
            importance[li, hi] = mean_mag[hi].item()
        counts[li] = 1

    # NO FALLBACK to L2-norm. The L2-norm metric is the v1 bug
    # (sentinel-ai#155, VALIDATED-TENSOR-SURGERY Finding 4) — silently
    # falling back to it would mask metric coverage gaps the same way
    # the in-pipeline forward-hook eval masked the v1 LoRA bug.
    #
    # For each layer:
    # - skipped (Strategy A): mark all heads as inf so select_heads_to_prune
    #   never picks them. This is correct for non-full_attention layers in
    #   hybrid architectures.
    # - full_attention but no captured data: this is a real bug. Halt loudly.
    for li in range(n_layers):
        if li in skipped_layer_indices:
            # Strategy A: non-full_attention layer, do not prune any head
            importance[li, :] = float("inf")
            continue
        if counts[li] == 0:
            raise RuntimeError(
                f"compute_activation_importance: layer {li} is full_attention "
                f"but the activation hook captured no data. This is a real bug "
                f"(possibly a missing o_proj on the layer, or all calibration "
                f"forward passes failed). Falling back to L2-norm here would "
                f"mask the bug; halting instead."
            )

    return importance


def select_heads_to_prune(importance, prune_percent, min_surviving_per_layer=4,
                          mode="per_layer"):
    """Select lowest-importance heads.

    Two modes:
        "per_layer" (DEFAULT): prune `prune_percent` of heads from EACH layer
            independently, picking the lowest-importance heads within that layer.
            Eliminates the cross-layer bias of activation-magnitude importance
            (early layers have smaller residual stream norms, so flat-global
            ranking concentrates prunes in the first few layers — see
            sentinel-ai#165 v2-7B investigation).

        "global_flat" (LEGACY): the v1 behavior — sort all heads globally by
            importance and prune the lowest. Kept for v1 reproduction. Do NOT
            use this mode for new forge runs; it has a structural early-layer
            bias that destroys low-level capacity disproportionately.

    The min_surviving_per_layer floor still applies in both modes.
    """
    n_layers, n_heads = importance.shape
    heads = {}
    n_pruned_total = 0

    if mode == "per_layer":
        # Per-layer pruning: prune ceil-or-floor(prune_percent * n_finite_heads_in_layer)
        # heads from each layer, picked from lowest importance within that layer.
        for li in range(n_layers):
            row = importance[li]
            finite_mask = row < float("inf")
            n_finite = int(finite_mask.sum().item())
            if n_finite == 0:
                continue  # non-attention layer (e.g. linear-attention in hybrid models)

            # Number of heads to prune in THIS layer
            n_to_prune = int(round(prune_percent * n_finite))
            # Respect the minimum-surviving floor
            max_prunable = max(0, n_finite - min_surviving_per_layer)
            n_to_prune = min(n_to_prune, max_prunable)
            if n_to_prune == 0:
                continue

            # Pick the n_to_prune lowest-importance heads in this layer
            finite_indices = [h for h in range(n_heads) if row[h] < float("inf")]
            finite_indices.sort(key=lambda h: row[h].item())
            picked = finite_indices[:n_to_prune]
            heads[li] = picked
            n_pruned_total += len(picked)

    elif mode == "global_flat":
        flat = importance.flatten()
        _, indices = flat.sort()
        for idx in indices:
            if importance[idx // n_heads, idx % n_heads] == float("inf"):
                continue
            li = idx.item() // n_heads
            hi = idx.item() % n_heads
            current_pruned = len(heads.get(li, []))
            finite_heads = (importance[li] < float("inf")).sum().item()
            if finite_heads - current_pruned <= min_surviving_per_layer:
                continue
            heads.setdefault(li, []).append(hi)
            n_pruned_total += 1
            if n_pruned_total >= int(importance.numel() * prune_percent):
                break
    else:
        raise ValueError(f"select_heads_to_prune mode must be 'per_layer' or 'global_flat', got {mode!r}")

    return heads, n_pruned_total


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
    """Install forward hooks that mask pruned head outputs."""
    layers = get_layers(model)
    hooks = []

    for li, head_list in heads_to_prune.items():
        attn = getattr(layers[li], "self_attn", getattr(layers[li], "attn", None))
        if attn is None:
            continue
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None:
            continue

        # Get actual head dim from o_proj shape (not config)
        o_dim_per_head = o_proj.weight.shape[1] // info["num_heads"]

        def make_hook(pruned_heads, hd):
            def hook_fn(module, input, output):
                for hi in pruned_heads:
                    s, e = hi * hd, (hi + 1) * hd
                    if e <= output.shape[-1]:
                        output[:, :, s:e] = 0
                return output
            return hook_fn

        h = o_proj.register_forward_hook(make_hook(head_list, o_dim_per_head))
        hooks.append(h)

    return hooks


def prune(model, prune_percent, info, method="zero_weights",
          metric="auto", tokenizer=None, calibration_texts=None,
          distribution="per_layer", calibration_source="generic"):
    """Dispatch to the right pruning strategy.

    Args:
        model: live PyTorch model
        prune_percent: fraction of total heads to remove
        info: model info dict
        method: 'zero_weights' or 'forward_hooks'
        metric: importance metric to use:
            - 'auto'        — activation if tokenizer is provided, else l2_weight
            - 'activation'  — compute_activation_importance (recommended; sentinel-ai #155)
            - 'l2_weight'   — compute_head_importance (DEPRECATED, kept for v1 reproduction)
        tokenizer: required for 'activation' metric
        calibration_texts: optional calibration set for 'activation' metric
    """
    if metric == "auto":
        metric = "activation" if tokenizer is not None else "l2_weight"

    # Resolve calibration_texts: if caller passed an explicit set, use it.
    # Otherwise look up the named calibration_source. The named lookup happens
    # here (not in compute_activation_importance) so the dispatch is visible
    # at the prune() boundary and the diagnostic line below can report it.
    if calibration_texts is None:
        calibration_texts = get_calibration_texts(calibration_source)

    if metric == "activation":
        if tokenizer is None:
            raise ValueError("metric='activation' requires a tokenizer")
        importance = compute_activation_importance(
            model, tokenizer, info, calibration_texts=calibration_texts,
        )
    elif metric == "l2_weight":
        importance = compute_head_importance(model, info)
    else:
        raise ValueError(f"Unknown importance metric: {metric}")

    heads, n_pruned = select_heads_to_prune(importance, prune_percent, mode=distribution)
    total = info["num_layers"] * info["num_heads"]

    hooks = []
    if method == "forward_hooks":
        hooks = prune_by_hooks(model, heads, info)
    else:
        prune_by_zeroing(model, heads, info)

    # Diagnostic: report per-layer prune distribution so the early-layer-bias
    # bug from sentinel-ai#165 is visible at forge time, not after the fact.
    layer_counts = sorted(set(len(v) for v in heads.values())) if heads else [0]
    layers_touched = len(heads)
    print(f"  Pruned {n_pruned}/{total} heads ({method}, metric={metric}, distribution={distribution}, calibration={calibration_source})")
    print(f"    layers touched: {layers_touched}/{info['num_layers']}, per-layer prune counts: {layer_counts}")
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
            # Mask labels at pad positions so the loss only counts valid
            # tokens — same fix as evaluate(). Otherwise the model is
            # trained to predict 0 at every padding position which is
            # both wrong (no signal) and noisy (degrades the LoRA fit).
            labels = ids.clone()
            labels[mask == 0] = -100

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
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
    parser.add_argument("model", nargs="?", help="HuggingFace model ID (e.g., Qwen/Qwen3.5-4B)")
    parser.add_argument("--alloy", type=str, default=None,
                       help="Path to .alloy.json — reads all params from alloy instead of CLI args")
    parser.add_argument("--domain", type=str, default=None,
                       help="Training domain: general, code, reasoning, chat, science")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prune-level", type=float, default=0.3)
    parser.add_argument("--prune-strategy", type=str, default="entropy",
                       help="Pruning strategy: entropy, magnitude, gradient, random")
    parser.add_argument("--prune-metric", type=str, default="auto",
                       choices=["auto", "activation", "l2_weight"],
                       help="Importance metric for head selection. 'activation' is "
                            "recommended (sentinel-ai #155); 'l2_weight' is deprecated "
                            "but kept for v1 reproduction.")
    parser.add_argument("--prune-distribution", type=str, default="per_layer",
                       choices=["per_layer", "global_flat"],
                       help="Head selection distribution mode. 'per_layer' (default) "
                            "prunes prune_level fraction from EACH layer independently, "
                            "eliminating the early-layer bias of activation-magnitude "
                            "importance (sentinel-ai #165). 'global_flat' is the v1 "
                            "behavior, kept for v1 reproduction only.")
    parser.add_argument("--calibration-source", type=str, default="generic",
                       choices=["generic", "code"],
                       help="Calibration text set used by activation-importance metric. "
                            "'generic' (default, v1 behavior) uses 16 generic English sentences. "
                            "'code' uses 16 hand-written HumanEval-format code completion prompts "
                            "(held-out from any benchmark). The §4.1.3.2 PPL/HumanEval-disconnect "
                            "finding suggests calibration domain matters; 'code' is the held-out-aware "
                            "experiment that tests whether shifting calibration to be in-domain "
                            "for the held-out task closes the disconnect.")
    parser.add_argument("--defrag-mode", type=str, default="slice",
                       choices=["slice", "pad", "none"],
                       help="Defrag behavior. 'slice' = physical removal (v1, breaks "
                            "llama.cpp on most modern transformers). 'pad' = physical "
                            "removal in compute, zero-pad q_proj/o_proj back to "
                            "hidden_size on save (Finding 6 fix). 'none' = skip defrag.")
    parser.add_argument("--lr", "--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--early-stop", type=float, default=None,
                       help="Stop if per-cycle improvement < this %%")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Force 4-bit loading (auto-detected if model too large)")
    parser.add_argument("--experts", type=int, default=0,
                       help="MoE expert count (0 for dense models)")
    parser.add_argument("--status-json", action="store_true",
                       help="Write status.json for real-time monitoring")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # If --alloy provided, read params from alloy file
    alloy_data = None
    if args.alloy:
        alloy_path = Path(args.alloy)
        if not alloy_path.exists():
            print(f"ERROR: Alloy file not found: {alloy_path}")
            sys.exit(1)
        alloy_data = json.loads(alloy_path.read_text())
        # Extract params from alloy
        args.model = alloy_data["source"]["baseModel"]
        args.cycles = alloy_data.get("cycles", 3)
        # Find prune and train stages
        for stage in alloy_data.get("stages", []):
            if stage["type"] == "prune":
                args.prune_level = stage.get("level", 0.3)
                args.prune_strategy = stage.get("strategy", "entropy")
            elif stage["type"] == "train":
                args.domain = stage.get("domain", args.domain)
                args.steps = stage.get("steps", args.steps)
                args.lr = float(stage.get("learningRate", str(args.lr)))
        print(f"  Loaded alloy: {alloy_data['name']} v{alloy_data['version']}")

    if not args.model:
        parser.error("model is required (either as positional arg or via --alloy)")
    if not args.domain:
        parser.error("--domain is required (either as CLI arg or via alloy train stage)")

    slug = args.model.split("/")[-1].lower()
    out = Path(args.output_dir or f"output/forged/{slug}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "benchmark").mkdir(exist_ok=True)

    # --- Pre-flight checks ---
    info = get_model_info(args.model)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cfg = ForgeConfig.auto(info["fp16_gb"], vram_gb, args.load_in_4bit)

    # HARD CHECK: pad-mode defrag requires fp16 weights for in-place tensor
    # zeroing. bnb 4-bit/8-bit weights are stored in packed format and cannot
    # be mutated through the standard tensor API. Auto-pick must NOT silently
    # override pad mode by selecting 4-bit. The right move is to fail loud and
    # let the user pick a smaller model, a bigger GPU, or --defrag-mode none.
    if args.defrag_mode == "pad" and cfg.load_4bit:
        sys.exit(
            f"FATAL: --defrag-mode pad requires fp16 weights, but auto-config "
            f"selected 4-bit (model fp16 = {info['fp16_gb']:.1f} GB, VRAM = "
            f"{vram_gb:.0f} GB, ratio = {info['fp16_gb']/vram_gb:.2f}). Pad-mode "
            f"defrag mutates weight tensors in place; bnb 4-bit storage is "
            f"packed and cannot be mutated through the standard tensor API. "
            f"Options: (1) use a GPU with more VRAM, (2) use a smaller base "
            f"model, (3) implement bnb-aware pad defrag (Path B in the v2 "
            f"forge plan), or (4) --defrag-mode slice (which violates llama.cpp "
            f"runtime portability per Finding 6 — not recommended)."
        )

    # Disk space check
    import shutil
    disk = shutil.disk_usage(out)
    disk_free_gb = disk.free / 1e9
    disk_pct = (disk.used / disk.total) * 100
    if disk_free_gb < 20:
        print(f"  ⚠️  WARNING: Only {disk_free_gb:.0f}GB disk free ({disk_pct:.0f}% used)")
        print(f"  Forge may fail if output + checkpoints exceed available space.")
    if disk_free_gb < 5:
        print(f"  ❌ ABORTING: Less than 5GB free. Clean up before forging.")
        sys.exit(1)

    # GPU compute test
    try:
        test_t = torch.randn(100, 100, device="cuda")
        _ = test_t @ test_t
        gpu_ok = "CUDA compute verified"
    except Exception as e:
        gpu_ok = f"CUDA FAILED: {e}"
        print(f"  ❌ {gpu_ok}")

    print(f"\n{'='*60}")
    print(f"  FORGING: {args.model}")
    print(f"  Params: {info['total_params']/1e9:.1f}B, fp16: {info['fp16_gb']:.1f}GB, 4-bit: {info['q4_gb']:.1f}GB")
    print(f"  Hardware: {torch.cuda.get_device_name(0)}, {vram_gb:.0f}GB VRAM, {gpu_ok}")
    print(f"  Disk: {disk_free_gb:.0f}GB free ({disk_pct:.0f}% used)")
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
        heads, hooks = prune(
            model, cycle_prune, info, cfg.pruning_method,
            metric=args.prune_metric,
            tokenizer=tokenizer,
            distribution=args.prune_distribution,
            calibration_source=args.calibration_source,
        )
        all_hooks.extend(hooks)

        write_status(out, "post_prune_eval", f"Cycle {cycle}: evaluating after prune",
                    cycle=cycle)
        post_prune = evaluate(model, eval_loader, out, f"post-prune-c{cycle}")
        imp = (baseline["perplexity"] - post_prune["perplexity"]) / baseline["perplexity"] * 100
        print(f"  After prune: perplexity={post_prune['perplexity']:.2f} ({imp:+.1f}% vs baseline)")
        check_vram("post-prune")

        # DEFRAG — structurally remove pruned heads on LAST cycle only
        # Defrag changes tensor dimensions which breaks subsequent cycles.
        # Future: proper per-cycle recalibration (issue #101)
        is_last_cycle = (cycle == args.cycles) or (args.early_stop and cycle >= 2 and
            abs(cycle_results[-2]["post_train_ppl"] - post_train["perplexity"]) / cycle_results[-2]["post_train_ppl"] * 100 < args.early_stop if len(cycle_results) >= 2 else False)

        if is_last_cycle and heads and args.defrag_mode != "none":
            write_status(out, "defrag", f"Cycle {cycle}: defragging pruned heads (final cycle, mode={args.defrag_mode})",
                        cycle=cycle)
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from defrag_inline import defrag_live_model
            # Remove hooks — defrag makes them unnecessary
            for h in all_hooks:
                h.remove()
            all_hooks.clear()

            # NO try/except. Defrag failure HALTS the forge. The whole point of
            # the v2 path is that defrag must actually happen — silently skipping
            # it produces a v1-class artifact mislabeled as v2, which is exactly
            # the bug class VALIDATED-TENSOR-SURGERY documents.
            freed = defrag_live_model(model, dead_heads=heads, mode=args.defrag_mode)
            freed_mb = freed / 1e6
            new_params = sum(p.numel() for p in model.parameters()) / 1e9
            print(f"  Defragged: freed {freed_mb:.0f}MB, model now {new_params:.1f}B params")
            write_status(out, "defrag_done", f"Freed {freed_mb:.0f}MB",
                        cycle=cycle, freed_mb=round(freed_mb))
            check_vram("post-defrag")
        elif heads:
            print(f"  Defrag deferred to final cycle (structural changes break mid-training)")

        write_status(out, "cycle_done", f"Cycle {cycle}: {imp:+.1f}% vs baseline",
                    cycle=cycle, perplexity=round(post_prune['perplexity'], 2),
                    improvement_pct=round(imp, 2))

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

    # --- 4. Final ---
    # Evaluate WITH pruning hooks still active — removing hooks before eval
    # causes pruned heads to output garbage (their weights were never zeroed,
    # only masked by the hooks during forward pass).
    final = evaluate(model, eval_loader)

    # Now remove hooks after eval
    for h in all_hooks:
        h.remove()
    total_imp = (baseline["perplexity"] - final["perplexity"]) / baseline["perplexity"] * 100

    print(f"\n{'='*60}")
    print(f"  {args.model}: {baseline['perplexity']:.2f} → {final['perplexity']:.2f} ({total_imp:+.1f}%)")
    print(f"{'='*60}")

    # --- 5. Generate samples + sanity check ---
    print("\n[4] Generating output samples...")
    samples = generate_samples(model, tokenizer, args.domain)
    for name, text in samples.items():
        (out / "benchmark" / f"{name}.txt").write_text(text)

    # Inference sanity check — catch broken models before declaring success
    print("\n[4b] Inference sanity check...")
    sanity_prompts = [
        ("fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"),
        ("hello", "def hello(name):\n    return"),
    ]
    sanity_passed = True
    for name, prompt in sanity_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        completion = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Check for degenerate output: repetition, empty, or just whitespace
        is_repetitive = len(set(completion.split())) < 3 and len(completion) > 20
        is_empty = len(completion.strip()) < 5
        is_garbled = completion.count("\n    return\n") > 2  # bare return loop = broken
        ok = not (is_repetitive or is_empty or is_garbled)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {completion[:80]!r}")
        if not ok:
            sanity_passed = False

    if not sanity_passed:
        print("\n  ⚠️  SANITY CHECK FAILED — model may generate garbage.")
        print("  The model will still be saved but should NOT be published without review.")

    # --- 6. Save ---
    print("\n[5] Saving model...")
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"  Saved to {model_dir}")

    # SAVE-THEN-RELOAD SMOKE TEST: load the just-saved model from disk
    # to verify the shapes in config.json match the actual safetensors.
    # This catches the entire class of bugs where defrag/prune mutates
    # tensor shapes but model.config doesn't get updated to match
    # (the qwen2-5-7b 2026-04-09 incident — model published, but
    # AutoModelForCausalLM.from_pretrained failed with size mismatch
    # errors because the slice-mode defrag produced per-layer shape
    # divergence that the single config.num_attention_heads couldn't
    # represent). Failing here at FORGE time means we never publish
    # an artifact that downstream users can't load.
    print("  [smoke] save-then-reload check...")
    try:
        from transformers import AutoModelForCausalLM as _ReloadCls
        _check = _ReloadCls.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        del _check
        print(f"  [smoke] OK — saved model loads cleanly via from_pretrained")
    except Exception as _e:
        # Loud failure — never silently let a non-loadable artifact land.
        raise RuntimeError(
            f"SAVE-THEN-RELOAD SMOKE TEST FAILED: the model saved to "
            f"{model_dir} cannot be loaded via from_pretrained. "
            f"This means the saved config.json doesn't match the actual "
            f"safetensors shapes (typically caused by defrag mutating "
            f"tensors without updating model.config). Original error:\n"
            f"  {type(_e).__name__}: {_e}\n"
            f"FIX: ensure defrag_live_model updates model.config to "
            f"reflect post-defrag dimensions, OR use defrag mode 'pad' "
            f"which preserves the wire shape."
        ) from _e

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

    # --- 8. Write executed alloy ---
    write_executed_alloy(args, alloy_data, results, samples, out)


def write_executed_alloy(args, alloy_data, results, samples, out: Path):
    """Write an executed .alloy.json with results, benchmarks, and samples."""
    import hashlib

    base = args.model.split("/")[-1].lower()
    domain = args.domain or "general"

    # Start from input alloy or build a new one
    if alloy_data:
        alloy = alloy_data.copy()
    else:
        alloy = {
            "name": f"{base}-{domain}-forged",
            "version": "1.0.0",
            "description": f"Forged {args.model} for {domain} domain",
            "author": "continuum-ai",
            "tags": [domain, "forged", "experiential-plasticity", "forge-alloy"],
            "license": "apache-2.0",
            "source": {
                "baseModel": args.model,
                "architecture": "qwen3_5" if "qwen3.5" in base else "qwen2" if "qwen2" in base else "llama",
            },
            "stages": [
                {
                    "type": "prune",
                    "strategy": getattr(args, "prune_strategy", "entropy"),
                    "level": args.prune_level,
                },
                {
                    "type": "train",
                    "domain": domain,
                    "dataset": results.get("training_data", ""),
                    "steps": args.steps,
                    "learningRate": str(args.lr),
                },
            ],
            "cycles": args.cycles,
        }

    # Ensure forge-alloy tag
    if "forge-alloy" not in alloy.get("tags", []):
        alloy.setdefault("tags", []).append("forge-alloy")

    # Build generation samples for alloy
    alloy_samples = []
    for name, text in samples.items():
        label = name.replace(".txt", "").replace("_", " ").title()
        alloy_samples.append({
            "label": label,
            "prompt": f"({domain} generation sample)",
            "completion": text.strip()[:2000],  # Cap at 2K chars
        })

    # Hash model weights — FULL file hashing, not partial.
    # 4KB prefix hashing is trivially bypassed (swap tensor data after header).
    # A 10GB model takes ~15s to hash — negligible vs a 42-minute forge.
    model_hash = ""
    model_dir = out / "model"
    if model_dir.exists():
        safetensors = sorted(model_dir.glob("*.safetensors"))
        if safetensors:
            h = hashlib.sha256()
            for sf in safetensors:
                with open(sf, 'rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        h.update(chunk)
            model_hash = f"sha256:{h.hexdigest()}"

    # Hash this script (code attestation)
    script_path = Path(__file__).resolve()
    script_hash = f"sha256:{hashlib.sha256(script_path.read_bytes()).hexdigest()}"

    # Populate results
    alloy["results"] = {
        "completedAt": results.get("forged_at", ""),
        "baselinePerplexity": results.get("baseline_ppl"),
        "finalPerplexity": results.get("final_ppl"),
        "improvementPct": results.get("improvement_pct"),
        "benchmarks": [
            {
                "name": "perplexity",
                "metrics": {
                    "baseline": results.get("baseline_ppl", 0),
                    "final": results.get("final_ppl", 0),
                    "improvement": results.get("improvement_pct", 0),
                },
            }
        ],
        "hardwareVerified": [
            {
                "device": results.get("device", "unknown"),
                "format": "fp16" if not results.get("load_4bit") else "4-bit",
                "verified": True,
            }
        ],
        "samples": alloy_samples,
        "integrity": {
            "trustLevel": "self-attested",
            "code": {
                "runner": "sentinel-ai/forge_model",
                "version": "3.0.0",
                "binaryHash": script_hash,
            },
            "modelHash": model_hash,
            "datasets": [],
            "attestedAt": results.get("forged_at", ""),
        },
    }

    alloy_path = out / f"{alloy['name']}.alloy.json"
    alloy_path.write_text(json.dumps(alloy, indent=2))
    print(f"  Alloy: {alloy_path}")


if __name__ == "__main__":
    main()
