---
license: apache-2.0
tags:
  - continuum-ai
  - forge-alloy
  - moe-compaction
  - calibration-aware
  - expert-pruning
  - consumer-hardware
  - mixtral
base_model: mistralai/Mixtral-8x7B-Instruct-v0.1
model-index:
  - name: mixtral-8x7b-instruct-compacted-conservative
    results: []
---

# Mixtral 8x7B Instruct — Compacted (Conservative)

**A Mixtral 8x7B compacted from 8→6 experts per layer via calibration-aware activation count, forged entirely on consumer hardware.**

| | Base Model | This Model | Change |
|---|---|---|---|
| **Total params** | 46.7B | ~35B | −25% |
| **Active params** | 12.9B | 12.9B | unchanged |
| **Experts/layer** | 8 | 6 | 8→6 pruned |
| **Size (fp16)** | 93.4 GB | 70.9 GB | −24% |
| **Baseline PPL** | — | 8.14 | — |
| **Final PPL (Q4_K_M)** | — | **8.97** | +10.2% |
| **Compression** | 93.4 GB | 20.4 GB (Q4_K_M) | **4.6x** |
| **Throughput** | — | 5,658 tok/sec (RTX 5090) | — |

> **Perplexity evaluated via llama.cpp** on wikitext-2-raw (162 chunks, context 2048). Final PPL = 8.97 ± 0.06. The 10.2% degradation from baseline is the cost of removing 25% of experts AND quantizing to 4-bit. A 93 GB datacenter model now runs on a MacBook Air.

> **§4.1.3.4 Calibration-Aware Activation Count Methodology.** The experts removed are the ones that fire least frequently on a held-out calibration corpus (300 code examples, 148,945 tokens), NOT the ones with the smallest weight norms. This is the same methodology used for [qwen3-coder-30b-a3b-compacted-19b-256k](https://huggingface.co/continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k), now validated on a second model family.

## Downloads

| Quantization | Size | PPL | Target Hardware |
|---|---|---|---|
| **Q4_K_M** | 20.4 GB | **8.97** | MacBook Air 24GB, RTX 3060 12GB+ |
| **fp16** | 70.9 GB | — | RTX 4090, RTX 5090 |

```bash
# Run with llama.cpp
./llama-cli -m mixtral-8x7b-compacted-Q4_K_M.gguf -p "Write a function that..."

# Run with Ollama
ollama run continuum-ai/mixtral-8x7b-instruct-compacted-conservative
```

## What this model is

A Mixtral 8x7B where the 2 least-activated experts per layer (out of 8) have been removed based on a calibration-aware activation profile. The surviving 6 experts per layer are the ones the model *actually uses* on a held-out code corpus. The pruning is behaviorally grounded — it measures what the model does, not what its weights look like.

**Cross-family anchor table — Row 2:**

| Row | Base Model | Family | Experts | Keep | Status |
|---|---|---|---|---|---|
| 1 | qwen3-coder-30b-a3b | Qwen3 MoE | 128 | 80 | ✅ Published |
| **2** | **Mixtral 8x7B Instruct** | **Mixtral** | **8** | **6** | **✅ This model** |
| 3 | Mixtral 8x22B Instruct | Mixtral (frontier) | 8 | TBD | ⬜ Next |
| 4 | Qwen3.5-35B-A3B | Qwen3.5 (hybrid attn) | TBD | TBD | ⬜ Planned |
| 5 | DeepSeek-V2-Lite | DeepSeek (shared+routed) | 64 | 32 | ⬜ Planned |

The methodology is the same across all rows; only the family adapter and the base model change. **Reproducibility across families is the contribution, not any single model.**

## Forged on Consumer Hardware

This model was forged on a single workstation — not a datacenter, not a cloud instance, not an H100 cluster.

| Component | Spec |
|---|---|
| **GPU** | NVIDIA RTX 5090 (32 GB VRAM) |
| **System RAM** | 64 GB (62 GB allocated to WSL2) |
| **Cold Tier** | WD Red Pro 16 TB (xfs, native Linux mount) |
| **OS** | Ubuntu on WSL2 (Windows 11) |
| **Load Strategy** | 4-bit NF4 hybrid: GPU + CPU fp32 offload + xfs disk spill |
| **Activation Profile** | 300 examples, 148,945 tokens, 103 seconds |
| **Expert Prune** | Streaming safetensors rewrite, 14 shards, ~12 minutes |

The forge pipeline resolved 13 production issues during development — from filesystem reliability (drvfs→xfs) to BitsAndBytes compatibility to MoE-specific loading strategies. Every fix is a committed patch in the [sentinel-ai](https://github.com/CambrianTech/sentinel-ai) repository. See [FOUNDRY-FILESYSTEM-SETUP.md](https://github.com/CambrianTech/sentinel-ai/blob/main/docs/FOUNDRY-FILESYSTEM-SETUP.md) for the operator playbook.

## Activation Profile (§4.1.3.4)

The expert selection was determined by running 300 held-out code examples (148,945 tokens) through the source model and counting how many times each expert was selected by the router gate at each layer. Sample layer profiles:

| Layer | Top-5 experts (by activation count) | Dead experts |
|---|---|---|
| Layer 0 | [5, 2, 3, 4, 0] — 49K, 42K, 41K, 37K, 35K | 0/8 |
| Layer 16 | [6, 2, 1, 5, 4] — 46K, 44K, 38K, 37K, 37K | 0/8 |
| Layer 31 | [3, 6, 5, 7, 0] — 54K, 40K, 38K, 36K, 35K | 0/8 |

Zero dead experts across all 32 layers. The bottom 2 experts per layer get ~20-25K activations vs ~50K for the top — a 2x spread that justifies the pruning threshold.

## Alloy Provenance

| Field | Value |
|---|---|
| **Recipe** | `_seed_mixtral-8x7b-instruct-compacted-conservative.alloy.json` (included) |
| **Forge Pipeline** | [sentinel-ai](https://github.com/CambrianTech/sentinel-ai) |
| **Forge-Alloy SDK** | v0.1.0 |
| **Source Architecture** | `mixtral` (block_sparse_moe-unfused layout) |
| **Calibration Corpus** | `heldout_code_python_300ex_125ktok` |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "continuum-ai/mixtral-8x7b-instruct-compacted-conservative",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "continuum-ai/mixtral-8x7b-instruct-compacted-conservative"
)

prompt = "Write a Python function that finds the longest palindromic substring."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Contributing

**Human and AI contributors welcome.** [continuum on GitHub](https://github.com/CambrianTech/continuum) · [Discord](https://discord.gg/arfbCV2H) · [Moltbook](https://www.moltbook.com/u/continuum)

---

*Forged by [continuum-ai](https://huggingface.co/continuum-ai) using the [forge-alloy](https://github.com/CambrianTech/forge-alloy) pipeline. Intelligence for everyone. Exploitation for no one.*
