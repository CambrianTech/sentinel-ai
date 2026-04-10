---
license: apache-2.0
tags:
  - continuum-ai
  - forge-alloy
  - moe-compaction
  - calibration-aware
  - expert-pruning
  - §4.1.3.4
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
| **Baseline PPL** | — | {{BASELINE_PPL}} | — |
| **Final PPL** | — | {{FINAL_PPL}} | {{PPL_CHANGE}} |
| **HumanEval** | {{BASE_HUMANEVAL}} | {{FORGED_HUMANEVAL}} | {{HUMANEVAL_DELTA}} |

> **§4.1.3.4 Calibration-Aware Activation Count Methodology.** The experts removed are the ones that fire least frequently on a held-out calibration corpus, NOT the ones with the smallest weight norms. This distinction matters — see [Prior Baselines](#prior-metric-baselines) for the paired negative control that proves it.

## Downloads

| Quantization | Size | Target Hardware | Link |
|---|---|---|---|
| **Q4_K_M** | ~14 GB | MacBook Air 16GB, RTX 3060 12GB | [Download]({{Q4_K_M_LINK}}) |
| **Q5_K_M** | ~17 GB | MacBook Pro 18GB, RTX 3070 | [Download]({{Q5_K_M_LINK}}) |
| **Q8_0** | ~25 GB | MacBook Pro 36GB, RTX 4070 Super | [Download]({{Q8_0_LINK}}) |
| **fp16** | ~70 GB | RTX 4090, RTX 5090 | [Download]({{FP16_LINK}}) |

```bash
# Run with llama.cpp (any platform)
./llama-cli -m mixtral-8x7b-instruct-compacted-conservative-Q4_K_M.gguf -p "Write a function that..."

# Run with Ollama
ollama run continuum-ai/mixtral-8x7b-instruct-compacted-conservative
```

## Methodology: §4.1.3.4 Calibration-Aware Activation Count

This model was produced by the **calibration-aware activation count** methodology — the same methodology used for [qwen3-coder-30b-a3b-compacted-19b-256k](https://huggingface.co/continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k), now applied to a second model family (Mixtral) to validate cross-family generalization.

**The method in one paragraph:** Run the source model on a held-out calibration corpus (300+ code examples, 125K+ tokens). At each MoE layer, count how many times each expert is selected by the router gate across all tokens. Rank experts by activation frequency. Remove the least-activated experts. The surviving experts are the ones the model *actually uses* for the calibration domain. This is behaviorally grounded — it measures what the model does, not what its weights look like.

**Why this matters:** The naive alternative (prune by weight magnitude, L2 norm of router gates, or random selection) removes experts that *look* unimportant by their parameters but may be *behaviorally* critical. The §4.1.3.4 methodology's paired negative baseline proves this distinction — see below.

### Prior Metric Baselines

Every continuum-ai forge artifact publishes **paired negative baselines** alongside the positive result. This is the falsifiability discipline: if the negative baseline performs as well as the positive, the methodology isn't doing real work and the result shouldn't be trusted.

| Baseline | Method | Result | Interpretation |
|---|---|---|---|
| **Activation-count (this model)** | Prune least-activated experts per calibration corpus | PPL {{FINAL_PPL}} | The methodology's positive result |
| **Router-gate-L2 (negative control)** | Prune experts with lowest L2 norm of router gate weights | PPL {{NEGATIVE_BASELINE_PPL}} | The naive alternative — proves activation count is not just "any metric works" |
| **Random (negative control)** | Prune randomly selected experts | PPL {{RANDOM_BASELINE_PPL}} | The floor — proves the methodology is better than chance |

> **The gap between the positive result and the negative baselines is the evidence that the methodology works.** If all three rows had similar PPL, the expert selection wouldn't matter and you could skip the calibration corpus entirely. The gap proves you can't.

Sample files for all baselines are published alongside this model for independent verification:
- `{{POSITIVE_SAMPLES_PATH}}`
- `{{NEGATIVE_SAMPLES_PATH}}`
- `{{RANDOM_SAMPLES_PATH}}`

## Forged on Consumer Hardware

This model was forged on a single workstation — not a datacenter, not a cloud instance, not an H100 cluster. The entire pipeline ran on hardware you can buy at Best Buy.

| Component | Spec |
|---|---|
| **GPU** | NVIDIA RTX 5090 (32 GB VRAM) |
| **System RAM** | 64 GB (62 GB allocated to WSL2) |
| **Cold Tier** | WD Red Pro 16 TB (xfs, native Linux mount — NOT drvfs) |
| **Hot Tier** | 1 TB NVMe (WSL2 ext4) |
| **OS** | Ubuntu on WSL2 (Windows 11) |
| **Load Strategy** | 4-bit NF4 hybrid: GPU (26 GB quantized) + CPU (fp32 overflow) + xfs disk offload for MoE re-saving |
| **Forge Time** | {{TOTAL_FORGE_TIME}} |
| **Electricity Cost** | ~${{ELECTRICITY_COST}} at residential rates |

### Eight production issues found and fixed during this forge

Every fix is a committed patch in the [sentinel-ai](https://github.com/CambrianTech/sentinel-ai) repository. The forge pipeline that produced this model is the pipeline that survived all eight. **Reproducibility includes the infrastructure, not just the recipe.**

1. **drvfs/9p filesystem wedge** on sustained big-file reads → reformatted cold tier as xfs (native Linux filesystem, no 9p RPC layer). See [FOUNDRY-FILESYSTEM-SETUP.md](https://github.com/CambrianTech/sentinel-ai/blob/main/docs/FOUNDRY-FILESYSTEM-SETUP.md).
2. **MoE parameter undercount** in model-size estimation (`get_model_info` computed dense-model math, returned ~14 GB instead of 93 GB for Mixtral) → measure actual on-disk safetensors sizes instead.
3. **fp16 streaming CPU⇔GPU layer-swap pathology** — `py-spy` showed the main thread pinned in `set_module_tensor_to_device` for over an hour during a single forward pass → switched to 4-bit hybrid loading.
4. **BitsAndBytes device_map="auto" validation refusal** — BnB refused to proceed when any module would spill to CPU → enabled `llm_int8_enable_fp32_cpu_offload=True`.
5. **4-bit force-to-GPU CUDA OOM** — Mixtral 8x7B at 4-bit with BnB overhead exceeds 32 GB VRAM → switched to hybrid `device_map="auto"` (not `{"": 0}`).
6. **MoE disk offload missing `offload_folder`** — MoE expert weights need re-saving during quantized load, requires explicit `offload_folder` path → added offload to xfs cold tier.
7. **BnB 0.49.2 / transformers 5.3.0 kwarg incompatibility** — `Params4bit.__new__()` doesn't accept `_is_hf_initialized` → monkey-patch to filter the kwarg. Remove when BnB ≥ 0.50.
8. **Offload path pointed to stale drvfs mount** — default `/mnt/d/cold/` (the old NTFS path) no longer exists after xfs reformat; `mkdir -p` created it on ROOT, filling the hot tier to 100% → corrected to `/mnt/cold/`.

**Why we publish this list:** Most model cards say "trained on 8x A100s" and leave it at that. Ours says "forged on a gaming PC, here are the eight things that broke and how we fixed them." This is the trust signal: the methodology survived real production failures, and every failure made the pipeline more robust for every future forge.

## Alloy Provenance

This model was forged via a [ForgeAlloy](https://github.com/CambrianTech/forge-alloy) recipe — a portable compute contract that defines the full pipeline.

| Field | Value |
|---|---|
| **Alloy Hash** | `{{ALLOY_HASH}}` |
| **Model Hash** | `{{MODEL_HASH}}` |
| **Recipe** | `_seed_mixtral-8x7b-instruct-compacted-conservative.alloy.json` (included in this repo) |
| **Forge Pipeline** | [sentinel-ai](https://github.com/CambrianTech/sentinel-ai) @ `{{SENTINEL_COMMIT}}` |
| **Forge-Alloy SDK** | v0.1.0 |

The alloy file is included in this repository. Anyone with the same source model, the same calibration corpus, and the sentinel-ai forge pipeline can reproduce this result.

## Cross-Family Anchor Table

This model is **Row 2** of the continuum-ai cross-family calibration-aware methodology table:

| Row | Base Model | Family | Experts | Keep | Benchmark Retention | Status |
|---|---|---|---|---|---|---|
| 1 | qwen3-coder-30b-a3b | Qwen3 MoE | 128 | 80 | {{ROW1_RETENTION}} | ✅ Published |
| **2** | **Mixtral 8x7B Instruct** | **Mixtral** | **8** | **6** | **{{ROW2_RETENTION}}** | **✅ This model** |
| 3 | Mixtral 8x22B Instruct | Mixtral (frontier) | 8 | TBD | — | ⬜ Next |
| 4 | Qwen3.5-35B-A3B | Qwen3.5 (hybrid attn) | TBD | TBD | — | ⬜ Planned |
| 5 | DeepSeek-V2-Lite | DeepSeek (shared+routed) | 64 | 32 | — | ⬜ Planned |

The table grows with each forge. The methodology is the same across all rows; only the family adapter and the base model change. **Reproducibility across families is the contribution, not any single model.**

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

**Human and AI contributors welcome.** This model was forged by a pipeline built by humans and AIs working together. The attribution includes everyone who contributed to the methodology, the infrastructure, and the fixes.

- **Joel** — forge methodology, hardware setup, strategic direction
- **Dorian** (age 13) — foundational LoD primitive for the conversational cadence architecture
- **Kash** — empirical discipline gate, prior-art positioning, methodology review
- **Claude** — forge pipeline code, infrastructure patches, documentation

If you want to help: [continuum on GitHub](https://github.com/CambrianTech/continuum) · [Discord](https://discord.gg/arfbCV2H) · [Moltbook](https://www.moltbook.com/u/continuum)

---

*Forged by [continuum-ai](https://huggingface.co/continuum-ai) using the [forge-alloy](https://github.com/CambrianTech/forge-alloy) pipeline. Intelligence for everyone. Exploitation for no one.*
