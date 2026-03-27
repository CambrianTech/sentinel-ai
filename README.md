# Sentinel-AI

**Neural plasticity for transformers.** Prune, measure, grow, learn — a biologically inspired cycle that makes models smaller and better.

Sentinel-AI dynamically rewires transformer attention heads during training. It removes underutilized heads (pruning), measures the impact, regrows capacity where needed (mitosis), and retrains — producing models that are both smaller and more capable than the original.

## Results

Experiments run on RTX 5090 (32GB), March 2026. All models from HuggingFace, dataset: wikitext-2.

| Model | Params | Pruning | Heads Pruned | Baseline PPL | Final PPL | Change | Time |
|-------|--------|---------|-------------|-------------|-----------|--------|------|
| distilgpt2 | 82M | 30% entropy, 3 cycles | 3/72 + 1 mitosis | 474.24 | **3.08** | -99.4% | ~1 min |
| gpt2-medium | 355M | 30% entropy, 3 cycles | 115/384 | 3.34 | **3.25** | +2.7% | 3 min |
| gpt2-large | 774M | 30% entropy, 3 cycles | 216/720 | 3.05 | **3.17** | -4.0% | 10 min |
| gpt2-large | 774M | 40% entropy, 3 cycles | 288/720 | 3.03 | **3.27** | -8.1% | 6 min |
| Qwen2.5-3B | 3B | 30% entropy, 3 cycles | 30% sparsity | 2.30 | **2.29** | +0.45% | 19 min |

Key findings:
- **30% pruning consistently recovers or improves** over baseline after retraining
- **40% pruning recovers most quality** but benefits from more training steps
- **Head mitosis** (cloning overutilized heads) produces specialized copies that diverge
- **Larger models are more pruning-tolerant** — they have more redundancy to exploit

## Quick Start

```bash
git clone https://github.com/CambrianTech/sentinel-ai.git
cd sentinel-ai

# One-command setup (auto-detects CUDA/MPS/CPU, creates venv, installs deps)
./setup.sh

# Activate the environment
source .venv/bin/activate
```

## Run Experiments

### Full Plasticity Cycle (recommended)

The experiment runner handles warmup, attention analysis, multi-cycle pruning, retraining, evaluation, and text generation:

```bash
# GPT2-medium — good balance of speed and substance
python scripts/run_neural_plasticity.py \
  --model_name gpt2-medium \
  --pruning_strategy entropy \
  --pruning_level 0.3 \
  --training_steps 500 \
  --cycles 3

# GPT2-large — bigger model, more headroom
python scripts/run_neural_plasticity.py \
  --model_name gpt2-large \
  --pruning_strategy entropy \
  --pruning_level 0.3 \
  --training_steps 1000 \
  --cycles 3

# Qwen2.5-3B — modern architecture with GQA
python scripts/run_neural_plasticity.py \
  --model_name Qwen/Qwen2.5-3B \
  --pruning_strategy entropy \
  --pruning_level 0.3 \
  --training_steps 500 \
  --cycles 3
```

Results are saved to `output/neural_plasticity_<timestamp>/` with:
- Training metrics (CSV)
- Attention heatmaps and pruning decision visualizations (PNG)
- Generated text samples (baseline vs pruned)
- Saved model checkpoint

### Adaptive Architecture Experiment

Demonstrates the full biological cycle with gate-based pruning and head mitosis (cloning):

```bash
python experiment_plasticity.py
```

### Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CambrianTech/sentinel-ai/blob/main/colab_notebooks/NeuralPlasticityDemo.ipynb)

The notebook runs end-to-end on a free Colab T4 GPU.

## How It Works

### The Four-Phase Plasticity Cycle

```
PRUNE → MEASURE → GROW → LEARN
  │         │        │       │
  │         │        │       └─ Retrain with differential learning rates
  │         │        └─ Clone overutilized heads (mitosis)
  │         └─ Evaluate quality loss, identify gaps
  └─ Remove low-entropy heads (least informative)
```

1. **Prune** — Identify and remove attention heads with low information content (high entropy = uniform attention = not useful). Strategies: entropy, magnitude, random.

2. **Measure** — Evaluate perplexity impact. Track which layers lost capacity, which compensated.

3. **Grow** — Clone high-utilization heads into freed slots. Each clone starts at 50% capacity, maintaining output continuity (0.5 + 0.5 = 1.0). Clones then diverge and specialize.

4. **Learn** — Retrain with the pruned/grown architecture. Remaining heads adapt to cover lost capacity. New heads specialize into new roles.

### Why It Works

Transformers have significant redundancy in their attention heads. Many heads learn similar patterns. Pruning forces the remaining heads to specialize, and the retraining phase lets them adapt. The result is a model with fewer parameters that attends more efficiently.

This mirrors biological neural plasticity — the brain continuously prunes synapses during development and sleep, yet cognitive capability improves because remaining connections specialize.

## Project Structure

```
sentinel-ai/
├── setup.sh                          # One-command setup
├── experiment_plasticity.py           # Adaptive architecture demo
├── scripts/
│   └── run_neural_plasticity.py       # Full experiment runner
├── utils/neural_plasticity/
│   ├── experiment.py                  # NeuralPlasticityExperiment class
│   ├── visualization.py              # Attention heatmaps, dashboards
│   ├── core.py                       # Head metrics, pruning masks
│   └── training.py                   # Training loops
├── sentinel/
│   ├── models/                       # Adaptive transformer, head cloning
│   ├── plasticity/                   # Plasticity loop, sleep cycle
│   └── pruning/                      # Pruning strategies, fine-tuning
├── models/loaders/                   # Model-specific loaders (GPT2, OPT, etc.)
├── colab_notebooks/                  # Colab-ready notebooks
└── output/                           # Experiment results
```

## Related Work

- **Paper**: [Neural Plasticity in Transformers](../continuum/docs/papers/SENTINEL-AI-NEURAL-PLASTICITY.md) — the theoretical foundation
- **Paper**: [Plasticity Compaction](../continuum/docs/papers/PLASTICITY-COMPACTION.md) — applying plasticity to MoE models (67GB → 14GB)
- **Published models**: [continuum-ai on HuggingFace](https://huggingface.co/continuum-ai) — compacted models ready to use

## License

**AGPL-3.0** — use freely for research or commercial purposes. Modifications must remain open source. See [LICENSE](LICENSE).
