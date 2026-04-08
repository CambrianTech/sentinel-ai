# Compensation LoRA via Teacher Distillation — Design

> **Status**: Design document. Prototype implemented at `scripts/compensation_lora.py` for small-scale math/stability validation. Production scale-up gated on the small-scale validation passing.

## Problem

Section §4.1.3.2 of [PLASTICITY-COMPACTION](../../continuum/docs/papers/PLASTICITY-COMPACTION.md) documents a structural limitation of the activation-magnitude head-importance metric, even with the per-layer normalization fix from §4.1.3.1: the metric is computed against a calibration distribution that is necessarily close to the student's fine-tuning corpus, which means it identifies as "low importance" the heads whose contribution to *local* fine-tuning loss is small. Those heads turn out to be load-bearing for *held-out* task generalization (HumanEval, GSM8K, MMLU, etc.). The disconnect was empirically reproduced on Qwen2.5-Coder-7B base across three runs of the same forge methodology with the only variable being the number of training cycles:

| Run | Configuration | HumanEval pass@1 | Notes |
|---|---|---|---|
| 1 | broken global activation rank, 1 cycle, 500 steps | 50.0% | depth-biased prune |
| 2 | per-layer activation rank (§4.1.3.1 fix), 1 cycle, 500 steps | 54.9% | best to date |
| 3 | per-layer activation rank, 3 cycles, 1000 steps each | 46.3% | worse than 1-cycle |

**More training did not close the gap to the base 7B anchor (62.2 / 53.7); 6× more training made the gap *worse*.** The methodology preserves perplexity (Run 3 internal PPL is 1.77 vs base 2.10, a 16% improvement) but degrades HumanEval, exactly the disconnect originally documented in §3.3 of EXPERIENTIAL-PLASTICITY on Qwen3.5-4B. The disconnect is reproducible across model families and across cycle counts; it is a structural limitation of the metric, not a one-shot anomaly.

## Hypothesis

Adding a small **compensation structure** that is trained against the *unpruned teacher* (rather than against task loss on a fine-tuning corpus) can recover the held-out task capability that the surviving heads alone cannot absorb. The compensation structure is much smaller than what was pruned (LoRA adapter, ~few hundred MB) compared to the pruned model (~14 GB for a 7B base), so the net compaction is preserved.

The intuition is that **fine-tuning recovery** asks the surviving heads to do double duty (their own work + the work of removed heads), which they cannot fully do because they have finite capacity and because the calibration distribution biases them toward locally-rewarded patterns. **Distillation recovery** asks a *new* small structure (the LoRA adapter) to learn the *delta* between the pruned model's output and the teacher's output — i.e., to learn specifically what the pruned heads were contributing — without re-tasking the surviving heads. The new structure has its own dedicated capacity for the compensation work, and the training signal (teacher vs student hidden states) is held-out-aware by construction because the teacher's outputs reflect the full distribution of held-out task behavior rather than just the local fine-tuning loss.

## Architectural origin

The pattern is from `models/unet_transformer.py`'s `BaselineIntegratedBlock`, which already implements baseline integration + skip connections for GPT-2-era models in the sentinel-ai codebase. The key components:

```python
self.baseline_adapter = nn.Linear(embed_dim, embed_dim)   # learnable adapter
self.baseline_gate = nn.Parameter(torch.ones(1) * 0.3)    # learnable fusion gate
self.ln_baseline = nn.LayerNorm(embed_dim)
# In forward:
adapted_baseline = self.baseline_adapter(self.ln_baseline(baseline_states))
gate_value = torch.sigmoid(self.baseline_gate)
hidden_states = hidden_states * (1 - gate_value) + adapted_baseline * gate_value
```

This is structurally a *learned correction* applied to the student's hidden states, with a learnable gate controlling fusion strength, trained to make the student behave more like the baseline (teacher) at each layer. Mapping to a model-agnostic LoRA adapter:

| `BaselineIntegratedBlock` component | LoRA-pattern equivalent |
|---|---|
| `baseline_adapter` (Linear, embed_dim → embed_dim) | LoRA's low-rank adapter on attention/FFN projections (Linear, embed_dim → embed_dim, rank-r decomposed) |
| `baseline_gate` (sigmoid-gated learnable scalar) | LoRA's `alpha / rank` scaling factor |
| `ln_baseline` (LayerNorm before adapter) | implicit; the LoRA lives inside the existing layer norms of the student |
| `BaselineIntegratedBlock`'s baseline-state input from the teacher | the distillation training loop — teacher forward passes feed the LoRA's training signal, not the inference path |

The LoRA-pattern version trades the explicit per-block baseline integration (which requires both models running in lockstep at inference time) for a *trained* low-rank correction that's baked into the student's weights at training time and runs with no inference overhead. The teacher is only needed during the compensation training phase.

## Math

Let:
- $T(x)$ = unpruned teacher model
- $S(x)$ = pruned student model
- $L = \{l_1, l_2, \ldots, l_n\}$ = set of LoRA adapter parameters (trainable)
- $S_L(x)$ = student with LoRA adapters applied: at each target projection, $W_{\text{eff}} = W + (B A) \cdot (\alpha / r)$, where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ are the LoRA matrices, $r$ is the rank, $\alpha$ is the scaling factor

The training objective is the **distillation loss** between $T(x)$ and $S_L(x)$ on a calibration dataset $D$ drawn from a *held-out* task distribution:

$$
\mathcal{L}_{\text{distill}}(L) = \mathbb{E}_{x \sim D}\left[ \mathcal{L}_{\text{KD}}(T(x), S_L(x)) \right]
$$

Three candidate forms for $\mathcal{L}_{\text{KD}}$:

**Form A — MSE on per-layer hidden states.** Most directly U-Net-analogous; matches the `BaselineIntegratedBlock` pattern of "learn a residual correction at each layer."

$$
\mathcal{L}_{\text{KD}}^{\text{MSE}} = \frac{1}{n_{\text{layers}}} \sum_{i=1}^{n_{\text{layers}}} \| T_{\text{hidden}}^{(i)}(x) - S_{L,\text{hidden}}^{(i)}(x) \|_2^2
$$

**Form B — KL divergence on output logits.** Standard knowledge distillation (Hinton 2015).

$$
\mathcal{L}_{\text{KD}}^{\text{KL}} = T^2 \cdot D_{\text{KL}}\left( \text{softmax}(T_{\text{logits}}(x) / T) \| \text{softmax}(S_{L,\text{logits}}(x) / T) \right)
$$

where $T$ is the temperature (default 2.0).

**Form C — Sum of A and B**, equally weighted. Targets both intermediate-layer alignment and output-distribution alignment simultaneously; empirically more robust in the KD literature than either alone.

The optimization updates only the LoRA parameters $L$; the student base weights and the teacher are both frozen:

$$
L^* = \arg\min_L \mathcal{L}_{\text{distill}}(L)
$$

## Constraints from the rest of the substrate

The compensation LoRA must be compatible with the existing forge pipeline's invariants:

1. **`q_proj.shape == [hidden_size, hidden_size]`** (Finding 6 from VALIDATED-TENSOR-SURGERY): pad-mode defrag preserves this; the LoRA adapters must not break it. LoRA adapters are additive corrections to the existing projection weights, so they preserve the shape automatically.
2. **Teacher and student must have matching `hidden_size` and `num_hidden_layers`**: required for hidden-state distillation. Pad-mode defrag preserves both, so this works for any pad-mode student. Slice-mode students (which shrink `hidden_size`) are not compatible with this distillation script and would require a projection adapter (not yet implemented).
3. **`head_dim` explicit on both configs**: the `harness_checks.py` no-fallback discipline enforces this at load time; computing `head_dim = hidden_size // num_heads` post-prune is the v1 bug.
4. **No modifications to existing forge pipeline files**: this is a pure addition. The compensation LoRA runs as a standalone post-forge stage; the existing `forge_v2_pipeline.sh` is not touched.

## Stability concerns to validate at small scale before scaling up

Five things that can go wrong, with the validation method for each:

### 1. Hidden state magnitude mismatch
Pad-mode student has zero rows in `q_proj` and `o_proj` columns at the dead head positions. The model's hidden states might therefore have systematically different magnitudes than the teacher's, even at the un-pruned positions, because the surviving heads' attention patterns are biased by the absence of the dead heads' contributions. If the magnitude mismatch is large, the MSE loss will be dominated by the magnitude term rather than by the meaningful signal-direction term, and the LoRA will learn to scale rather than to compensate.

**Validation**: before training, run one forward pass each on teacher and student against a sample input. Print the per-layer hidden-state L2 norms for both. If they differ by more than ~2× at any layer, normalize the hidden states before computing the MSE (LayerNorm-style), or switch to a cosine-similarity loss (which is scale-invariant).

### 2. Gradient interaction between gradient checkpointing + LoRA + bnb-8bit teacher
A stack of memory-saving primitives that can interact in subtle ways: bnb-8bit changes how the teacher's forward pass produces fp16 outputs (cast on the fly), gradient checkpointing recomputes activations during backprop, and LoRA's parameter wrapping changes which tensors get gradients. Any one of these can fail silently (NaN/inf in gradients) without erroring out.

**Validation**: train for 5 steps on a tiny calibration set (10 examples). After each step, check the LoRA parameter gradients for NaN/inf and check that the loss decreases monotonically. If either check fails, identify which primitive is responsible by toggling them off one at a time.

### 3. Distillation loss exploding early
At step 0, the LoRA produces zero output (the LoRA's `B` matrix is initialized to zero by convention, so `B @ A = 0`), so the student is identical to the un-compensated student at step 0. The loss therefore starts at the post-prune gap. If the loss *diverges* in early training instead of decreasing, the LoRA is too aggressive (alpha too high relative to rank) or the learning rate is too high.

**Validation**: print the loss at step 0 and step 1. Step 0 should equal the gap at the un-compensated student's hidden states; step 1 should be less than or equal to step 0. If step 1 is greater, lower the learning rate by 10× and retry.

### 4. Per-layer loss imbalance
Different layers may have different hidden-state distributions (early layers have smaller residual norms; late layers have larger ones, per §4.1.3.1's observation about the depth-bias). The unweighted average MSE across layers may therefore be dominated by late-layer differences and ignore early-layer compensation needs.

**Validation**: print the per-layer MSE losses individually at step 1. If any single layer's loss is more than 10× the median, normalize per layer (divide each layer's MSE by the layer's hidden-state variance) before averaging.

### 5. Token alignment between teacher and student
Teacher and student must produce hidden states for the same input tokens. If their tokenizers differ (e.g., the student's tokenizer was modified during forge), the hidden-state distillation is comparing apples to oranges.

**Validation**: tokenize one example with both tokenizers and assert that the resulting `input_ids` are identical. If they're not, the student's tokenizer is incompatible with the teacher's and compensation cannot be performed without aligning them first.

## Small-scale validation plan

The script `scripts/compensation_lora.py` is the **prototype** for validating these stability concerns. It is intentionally model-agnostic so the same script runs on:

- **distilgpt2 as both teacher and student** (6 layers, 82M params) for the small-scale validation pass
- **Qwen2.5-Coder-7B base + v2-7B as student** for the production run (gated on small-scale passing)

The small-scale validation procedure:

1. Load distilgpt2 as the teacher (no quantization needed at this scale)
2. Manually prune ~50% of attention heads in a copy of distilgpt2 via pad-mode (preserves hidden_size)
3. Run `compensation_lora.py` with `--teacher distilgpt2 --student <pruned distilgpt2> --steps 50 --calibration-data <tiny mixture>`
4. Verify each of the 5 stability checks above
5. Compare distilgpt2-compensated generations vs distilgpt2-pruned-no-compensation generations on a held-out smoke test (some HumanEval problems and some general text). The compensated model should produce *meaningfully different* (and at least directionally better) outputs than the un-compensated pruned model.

If all 5 stability checks pass and the small-scale generation comparison shows the compensation is doing meaningful work, the design is validated and we scale up to the 7B production run. If any check fails, the design is refined (loss function, normalization, learning rate, LoRA rank) and the small-scale test re-run.

## Production scale-up plan (gated on small-scale validation)

Once the small-scale validation passes:

1. Load Qwen2.5-Coder-7B base as teacher (bnb-8bit, ~7 GB)
2. Load existing v2-7B (the 54.9 HumanEval pad-mode pruned artifact) as student (fp16, ~14 GB, gradient checkpointed)
3. Run `compensation_lora.py` with held-out calibration data (HumanEval-format problems mixed with GSM8K, MMLU, etc., drawn from sources that do NOT overlap the v2-7B forge's fine-tuning corpus)
4. Train for 500 steps initially (the same budget as the v2-7B forge itself; longer if budget allows)
5. Save the compensated model
6. Run the calibrated EvalPlus pipeline (`eval_with_calibration.py`) against the compensated model
7. Compare HumanEval pass@1 to:
   - the un-compensated v2-7B baseline (54.9) — improvement here validates the compensation strategy
   - the base 7B anchor (62.2) — closing the full gap validates the strategy as a complete fix for §4.1.3.2

## Success criteria

The compensation strategy is validated at production scale if:

- HumanEval pass@1 of the compensated v2-7B exceeds the un-compensated v2-7B baseline (54.9) by **at least 3 points**, which is outside the calibration tolerance band (±3 pt) defined in §4.1.4.1 of PLASTICITY-COMPACTION
- The compensated model still passes Layer 7 deployment-runtime gate (loads in llama.cpp at production speed, produces coherent code on the smoke test)
- The compensated model file size is not meaningfully larger than the un-compensated v2-7B (the LoRA is merged into the base weights, so the file size stays at ~5.3 GB; the compensation is essentially free at inference time)

If the compensated v2-7B closes most of the gap to the base anchor (lands in [60, 65] HumanEval), the strategy is fully validated and §4.1.3.3 of the methodology paper writes itself as "compensation LoRA via distillation is the structural fix for the §4.1.3.2 disconnect." If the compensated model improves but doesn't close the full gap, the strategy is partially validated and the next iteration is the cross-layer skip path from `BaselineIntegratedBlock` (more invasive, requires modifying the HF model class to expose intermediate residual streams). If the compensated model does not improve at all, the design is refuted and the next iteration is to explore loss formulations beyond hidden-state MSE and logit KL.

## Failure mode escalation

If any step fails, the escalation path is:

1. **Small-scale stability check fails** → fix the loss formulation / hyperparameters / dtype handling and re-run small-scale
2. **Small-scale passes but production scale-up fails** (NaN, OOM, divergence) → debug the memory/dtype interaction at 7B scale, possibly reduce LoRA rank or batch size, possibly use gradient accumulation
3. **Production training completes but compensated model doesn't beat un-compensated baseline** → the LoRA-pattern compensation is insufficient; escalate to cross-layer skip path implementation (implement a small wrapper around the HF model class that exposes the residual stream for skip-fuse)
4. **Cross-layer skip path also doesn't help** → the compensation approach is wrong for this regime; escalate to the next experimental wave (held-out-aware metric reformulation, learned routing-aware compensation, or both)

Each escalation step is more invasive than the last, and each step has its own design doc + small-scale validation before scale-up. We don't skip steps.

## Authors

- Joel Teply (Cambrian AI)
- with assistance from Kash (KashCompiler-side Claude, Anthropic)

## License

CC-BY 4.0 (design doc text). Code (`scripts/compensation_lora.py`) under the parent project license.
