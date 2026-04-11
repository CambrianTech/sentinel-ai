# Sentinel-AI

**[Experiential Plasticity](https://github.com/CambrianTech/continuum/blob/main/docs/papers/EXPERIENTIAL-PLASTICITY.md) for transformers.** Train on domain data, prune what doesn't matter, retrain — the model emerges smaller, faster, and better at its job. Like biological synaptic pruning during brain development.

The architecture co-evolves with training: heads that contribute to the domain specialize, heads that don't are removed. The result is a model architecturally optimized for its target task — not just quantized, but structurally reshaped.

**Published models:** [huggingface.co/continuum-ai](https://huggingface.co/continuum-ai)
**Paper:** [Experiential Plasticity: Transformers That Grow Their Own Architecture From Experience](https://github.com/CambrianTech/continuum/blob/main/docs/papers/EXPERIENTIAL-PLASTICITY.md)
**Part of:** [continuum](https://github.com/CambrianTech/continuum) — distributed AI on consumer hardware
**Forge format:** [ForgeAlloy](https://github.com/CambrianTech/forge-alloy) — trustless AI compute contract (cryptographically verified pipelines)

## Results

### Qwen3.5 Domain-Specific Forging

Domain-specific training amplifies the plasticity effect. Using [`forge_model.py`](scripts/forge_model.py) with LoRA + AMP mixed precision:

| Model | Params | Domain | Training Data | Baseline PPL | Final PPL | Change | Device |
|-------|--------|--------|--------------|-------------|-----------|--------|--------|
| **[Qwen3.5-4B](https://huggingface.co/continuum-ai/qwen3.5-4b-code-forged)** | 3.4B | Code | [CodeFeedback](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) (156K) | 3.04 | **2.31** | **+24.0%** | RTX 5090 |
| **[Qwen3.5-27B](https://huggingface.co/continuum-ai/qwen3.5-27b-code-forged)** | 23.6B | Code | CodeFeedback (156K) | 3.07 | **2.96** | **+3.5%** | RTX 5090 |

**+24% on 4B, +3.5% on 27B** — both better than baseline, both smaller. The 27B runs in 17GB (4-bit) instead of 28GB (fp16) while producing better code. Qwen3.5-27B benchmarks at Claude Sonnet 4.6 level ([source](https://x.com/TheAhmadOsman)) — now forged and improved, running on a MacBook Pro.

```bash
# Forge any model on any domain — memory tier auto-detected
python scripts/forge_model.py Qwen/Qwen3.5-4B --domain code
python scripts/forge_model.py Qwen/Qwen3.5-27B --domain code   # auto 4-bit on 32GB VRAM

# Or use a ForgeAlloy recipe — typed, portable, cryptographically attestable
python scripts/forge_model.py --alloy recipe.alloy.json
```

### Scaling Law

Improvement from [experiential plasticity](https://github.com/CambrianTech/continuum/blob/main/docs/papers/EXPERIENTIAL-PLASTICITY.md) scales with model size. Larger models harbor more redundancy.

| Model | Params | Baseline PPL | Final PPL | Change |
|-------|--------|-------------|-----------|--------|
| Qwen2.5-0.5B | 0.5B | 2.82 | 2.91 | −3.2% (too small) |
| Qwen2.5-1.5B | 1.5B | 2.49 | 2.42 | +3.0% |
| Qwen2.5-3B | 3.1B | 2.30 | 2.28 | +0.9% |
| **Qwen2.5-7B** | **7.6B** | **2.46** | **2.17** | **+11.8%** |
| **Qwen3.5-4B** | **3.4B** | **3.04** | **2.31** | **+24.0%** (code domain) |
| **Qwen3.5-27B** | **23.6B** | **3.07** | **2.96** | **+3.5%** (code, 4-bit, 17GB) |

Domain-specific training (Qwen3.5-4B on code) exceeds generic-text results (Qwen2.5-7B on wikitext) despite being a smaller model.

### MoE Expert Pruning (§4.1.3.4)

Calibration-aware expert activation count pruning. Profile which experts actually fire on a held-out corpus, remove the ones that don't. The surviving experts are the ones the model uses.

| Model | Experts | Kept | PPL (base) | PPL (forged) | Δ | Size (Q4_K_M) |
|-------|---------|------|-----------|-------------|---|--------------|
| **[Mixtral 8x7B](https://huggingface.co/continuum-ai/mixtral-8x7b-instruct-compacted-conservative)** | 8 | 6 | 8.14 | **8.97** | +10.2% | **20 GB** |
| **Mixtral 8x22B** | 8 | 6 | 7.81 | **~8.18** | +4.7% | **60 GB** |
| [Qwen3-Coder-30B-A3B](https://huggingface.co/continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k) | 128 | 80 | — | — | — | — |

Same methodology across independently-trained model families. The calibration corpus determines which experts survive — change the corpus, change the specialization. [Full methodology →](https://github.com/CambrianTech/continuum/blob/main/docs/papers/PLASTICITY-COMPACTION.md)

### Continuous Defrag

Traditional pruning masks heads but doesn't free memory. **[Continuous defrag](docs/CONTINUOUS-DEFRAG.md)** structurally removes dead heads between cycles — the model gets physically smaller, freeing VRAM for larger batch sizes. Each cycle trains faster than the last.

```
Cycle 1: train (batch=1, 27B, 17.9GB) → prune → defrag → freed 1.7GB
Cycle 2: train (batch=2, 24.5B, 16.2GB) → prune → defrag → freed 1.7GB  ← 2x faster
Cycle 3: train (batch=3, 22B, 14.5GB)  → prune → defrag                  ← 2.8x faster
```

**40% faster total training** and a **33% smaller final model** (GGUF Q4: 10GB instead of 15GB for Qwen3.5-27B).

### Self-Directed Plasticity

The [`AdaptivePlasticityController`](sentinel/plasticity/controller_ann/) observes the model and makes all decisions — pruning ratio, strategy, training budget, stopping criteria. No human hyperparameters.

Recovery from iterative pruning follows a measurable [transfer function](https://github.com/CambrianTech/continuum/blob/main/docs/papers/EXPERIENTIAL-PLASTICITY.md#4-the-transfer-function): `1.45·exp(−0.18·cycle) − 0.03` — connecting transformer optimization to classical control theory.

![Recovery Decay](paper/figures/recovery_decay_fit.png)

## Quick Start: Forge Your Own Model

Three commands. Any NVIDIA GPU with 8GB+ VRAM.

```bash
# 1. Clone and setup
git clone https://github.com/CambrianTech/sentinel-ai.git
cd sentinel-ai
./setup.sh                    # Creates venv, installs PyTorch + deps, detects CUDA/MPS
source .venv/bin/activate

# 2. Forge (pick your model + domain)
python scripts/forge_model.py Qwen/Qwen3.5-4B --domain code     # 8GB VRAM, ~30 min
python scripts/forge_model.py Qwen/Qwen3.5-9B --domain code     # 18GB VRAM, ~45 min
python scripts/forge_model.py Qwen/Qwen3.5-27B --domain code    # 32GB VRAM (4-bit auto), ~2 hr

# 3. Publish to HuggingFace
python publish_forged.py output/forged/qwen3.5-4b/ --domain code
```

**That's it.** The script auto-detects your GPU, picks the right memory tier, trains with LoRA + AMP, prunes attention heads, defrags, saves, and generates proof-of-quality code samples.

### What Happens During Forging

```
Load model → Baseline eval → [Train on domain data → Prune low-importance heads →
Defrag (structurally remove) → Eval] × N cycles → Generate samples → Save
```

- **Memory tiers**: Tier A (≤40% VRAM, fp16), Tier B (≤70%, fp16+accum), Tier C (>70%, 4-bit)
- **Observable**: `status.json` updates every 10 steps + inference sample every 200 steps
- **Early stopping**: `--early-stop 0.5` stops when improvement plateaus

### Manual Setup (if setup.sh doesn't work)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets peft bitsandbytes safetensors accelerate
pip install huggingface_hub   # for publishing
```

## Run on MacBook M1/M2/M3 (No GPU Required)

Don't have an NVIDIA GPU? Use our pre-forged models. Two commands:

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

# Load Sonnet 4.6-level model (15GB, runs on 32GB MacBook)
model, tokenizer = load("continuum-ai/qwen3.5-27b-code-forged-mlx-4bit")

# Generate code
print(generate(model, tokenizer, prompt="def merge_sort(arr):", max_tokens=200))
```

**That's it.** 15GB download, ~9 tok/s on M1 32GB. The model writes working code with chain-of-thought reasoning.

### End-to-End: Forge on GPU → Run on Mac

If you DO have an NVIDIA GPU and want to forge your own:

```bash
# On your GPU machine (RTX 3090, 4090, 5090, etc.)
git clone https://github.com/CambrianTech/sentinel-ai.git
cd sentinel-ai && ./setup.sh && source .venv/bin/activate

# Forge (auto-detects GPU, picks memory tier)
python scripts/forge_model.py Qwen/Qwen3.5-4B --domain code

# Publish to HuggingFace (creates your own model)
python publish_forged.py output/forged/qwen3.5-4b/ --domain code

# On your MacBook — convert to MLX 4-bit
pip install mlx-lm
python -c "from mlx_lm import convert; convert('YOUR_HF_USERNAME/qwen3.5-4b-code-forged', 'mlx-model', quantize=True, q_bits=4)"

# Run locally
python -c "from mlx_lm import load, generate; m,t = load('mlx-model'); print(generate(m,t,prompt='Write a web server:',max_tokens=300))"
```

### Classic Experiments

```bash
# GPT2-medium — combined strategy (best on generic text)
python scripts/run_neural_plasticity.py \
  --model_name gpt2-medium \
  --pruning_strategy combined \
  --pruning_level 0.3 \
  --training_steps 500 \
  --cycles 3

# Self-directed — no hyperparameters, controller decides everything
python experiments/experiment_self_directed.py --model_name gpt2-medium
```

### Notebooks

| Notebook | Description |
|----------|-------------|
| [Neural Plasticity Evidence](paper/NEURAL-PLASTICITY-EVIDENCE.ipynb) | All experimental results with publication figures |
| [Self-Directed Plasticity](paper/SELF-DIRECTED-PLASTICITY.ipynb) | V1→V2→PID controller evolution with transfer function analysis |
| [Colab Demo](colab_notebooks/NeuralPlasticityDemo.ipynb) | Run on free Colab T4 GPU [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CambrianTech/sentinel-ai/blob/main/colab_notebooks/NeuralPlasticityDemo.ipynb) |

## The Model Compiler

forge-alloy + sentinel-ai = a **compiler for neural networks**. You write a recipe (source code), the forge optimizes it for your hardware (target architecture), the benchmarks verify it (test suite), and the attestation proves it (build manifest).

```
Recipe → Profile → Search → Prune → Quantize → Evaluate → Publish
         (PGO)    (optimizer)  (dead code    (codegen)   (test)   (ship)
                                elimination)
```

The search is FAST: size filter (instant) → quality estimate (instant) → quick eval (2 min) → full eval (40 min). Only the winning configuration gets the expensive evaluation. Domain specialization comes from the calibration corpus — `-march=coding` prunes experts that don't fire on code. Same source model, different domain, different optimized output.

**Adapters make it extensible.** Every model family, pruning strategy, quantization format, and benchmark is an adapter. New model released? Write an adapter. New hardware target? Write a quant adapter. New training technique? Write a stage adapter. The community contributes adapters, the compiler integrates them. [Full architecture →](https://github.com/CambrianTech/forge-alloy/blob/main/docs/MODEL-COMPILER.md)

## The Factory Pipeline

Sentinel-AI is the forge. The factory pipeline turns it into an
**assembly line** for model production: drop a recipe alloy at the
intake station, BigMama (or any single-GPU box) builds it through the
family-adapter set, assays it against every benchmark it's eligible for,
and parks the finished artifact in the shipping bay. **Continuum is the
shipping department** — it reads `finished/`, applies its release gates,
and publishes to HuggingFace. Sentinel never pushes to HF; that's a
deliberate architectural boundary.

```
                                       ┌──────────────────────────┐
                                       │  .factory/line/          │
                  drop alloy here  →   │    intake/               │  ← cp my-recipe.alloy.json here
                                       │    assembly/  ← worker   │
                                       │    finished/  ← shipping │  ← continuum reads here
                                       │    rework/    ← QA flag  │
                                       └────────────┬─────────────┘
                                                    │
                                                    ▼
                                       FactoryWorker.process_one()
                                                    │
                                   ┌────────────────┴────────────────┐
                                   ▼                                 ▼
                          alloy_executor                       eval_runners
                          .execute_alloy()                   (registry dispatch)
                                   │                                 ▲
                                   │                                 │
                            family-adapter                  resolve_runner(name)
                            dispatch (16 adapters)                   │
                            → MoEUnfusedExpertsBase                  │
                            → MixtralAdapter                         │
                            → PhiMoEAdapter (inherits)               │
                            → DeepSeekV2Adapter                      │
                            → QwenVLAdapter                          │
                            → ... 11 more                            │
                                   │                                 │
                                   ▼                                 │
                            forged artifact  ──── assay (eval) ──→  9 real benchmark runners:
                                   │                                 HumanEval, HumanEval+,
                                   │                                 LCB v6, IFEval, BBH,
                                   │                                 MATH-Hard, GPQA,
                                   ▼                                 MMLU-Pro, MuSR
                            mark_finished()                          (Open LLM Leaderboard v2 pack)
                                   │
                                   ▼
                            .factory/line/finished/  ──→  CONTINUUM (shipping department)
                                                          • reads result manifest
                                                          • applies release gates
                                                          • pushes to HF
                                                          • posts model card
```

**Two-axis dispatch:**

- **Axis 1 — `source.architecture` → FamilyAdapter.** Each model family
  is one file in `scripts/adapters/` (16 adapters today). Adding a new
  family is one new file plus one import line. Old families stay frozen
  forever so older alloys reproduce bit-identically.
- **Axis 2 — benchmark name → BenchmarkRunner.** Each benchmark is one
  file in `scripts/eval_runners/` (9 real, 12 stubs). Adding a new
  benchmark is one new file. The §4.1.4.1 anchor-reproduction discipline
  gate routes through the same registry.

**Sending BigMama a part to build:**

```bash
cp my-recipe.alloy.json /path/to/.factory/line/intake/
python -m factory_queue --root /path/to/.factory --max-iters 1
```

The worker picks the part off `intake/`, moves it to `assembly/`, runs
`execute_alloy` (which dispatches to the right family adapter), executes
each stage including `eval` (registry-dispatched), and on success moves
the alloy to `finished/` with a `.result.json` sidecar pointing at the
on-disk forged artifact and the eval results. On any failure the part
goes to `rework/` with a `.error.json` sidecar carrying the full
traceback — no silent defaults, no retries on broken state.

**The filesystem IS the queue.** No DB, no service, no network
coordination. Multi-worker safety comes free if you ever need to scale
beyond a single GPU (atomic `intake → assembly` rename via `O_EXCL`).
Continuum's shipping department picks parts off `finished/`, applies
release gates, and publishes — separate from the assembly line, separate
process, separate auth scope.

## Papers

- **[Experiential Plasticity](https://github.com/CambrianTech/continuum/blob/main/docs/papers/EXPERIENTIAL-PLASTICITY.md)** — Scaling law, transfer function, self-directed controller, domain forging, continuous defrag
- **[Neural Plasticity in Transformers](https://github.com/CambrianTech/continuum/blob/main/docs/papers/SENTINEL-AI-NEURAL-PLASTICITY.md)** — Foundation paper: cross-architecture results, four-phase cycle, hypothetical training cost analysis
- **[Plasticity Compaction](https://github.com/CambrianTech/continuum/blob/main/docs/papers/PLASTICITY-COMPACTION-MOE.md)** — MoE expert pruning (67GB → 14GB)

## Architecture

```
sentinel-ai/
├── scripts/
│   ├── forge_model.py              # Domain-specific forging (Qwen3.5, LoRA, AMP)
│   ├── defrag_model.py             # Post-processing structural pruning
│   ├── defrag_inline.py            # Live in-place defrag during training
│   └── run_neural_plasticity.py    # Classic experiment runner
├── sentinel/
│   ├── plasticity/                 # Plasticity loop, controllers, sleep cycle
│   ├── pruning/                    # Pruning strategies (entropy, gradient, combined)
│   └── models/                     # Adaptive transformer, head cloning
├── docs/
│   └── CONTINUOUS-DEFRAG.md        # Defrag architecture
├── paper/                          # Notebooks and figures
└── output/                         # Experiment results, forged models
```

## License

MIT
