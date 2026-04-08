# VL Forge — Vision-Language Extension to the Forge Pipeline

> **Status**: Design document. Phase 1 implementation gated on review of this doc. Code lands as additive new modules first to avoid colliding with in-flight text-only forge runs; refactor into `forge_model.py` / `compensation_lora.py` / `cpu_expert_prune_v2.py` happens once Move 2 (Qwen3-Coder-30B-A3B) ships.

## Problem

The forge pipeline (`forge_model.py`, `compensation_lora.py`, `cpu_expert_prune_v2.py`) is text-only. The entire Qwen3.5 family — `Qwen3.5-27B`, `Qwen3.5-35B-A3B`, and every sibling — is `Qwen3_5MoeForConditionalGeneration` / `Qwen3_5ForConditionalGeneration`: vision-language by construction, with `vision_config`, `image_token_id`, `video_token_id`, a SigLIP-So400m-derived vision tower, and an MLP merger projecting visual features into the text decoder's hidden space. There is no text-only Qwen3.5 checkpoint published.

This collapses the strategy space:

- The Qwen 3.5+ floor and the vision-paramount priority cannot both be satisfied by the current pipeline.
- Running the existing forge against a Qwen3.5 checkpoint will either (a) crash on `get_layers()` or assertion mismatches between `text_config.hidden_size` and `vision_config.hidden_size`, or worse, (b) silently destroy the vision pathway by leaving the vision tower untouched while pruning/compensating the text decoder it feeds into, producing a "VL" model whose vision is structurally disconnected from its language.

The pipeline impact analysis identified **13 distinct breakage points across the 3 forge files** — 5 hard CRASHes, 8 SILENT BUGs. The silent bugs are the dominant risk because the artifact appears to load and produce text, but its multimodal capability is broken in ways that only show up on held-out vision benchmarks.

## Hypothesis

The forge methodology (importance-ranked pruning + pad-mode defrag + KL-on-logits compensation LoRA) can be extended to VL models with surgical, additive changes — **not a redesign**. The architectural reason: Qwen3.5-VL uses **scatter-injection** of vision tokens into the text decoder rather than cross-attention. The vision tower and merger are a self-contained pathway whose output is poured into the text decoder's input embeddings at sentinel positions. This means:

1. **The vision tower is structurally separable** from the prunable text decoder. It is small (~0.4–0.5B params for SigLIP-So400m) relative to the text decoder (14B for 27B dense, 30B+ for 35B-A3B MoE), so preserving it bit-exact costs essentially nothing in compaction terms.
2. **The merger MLP is the only bridge** between modalities, and it is a single small module (~5–25M params depending on `out_hidden_size`). Preserving it bit-exact has negligible compaction cost.
3. **The text decoder can be forged using the existing methodology unchanged** as long as the forge code knows to whitelist the vision tower, the merger, the vision-token vocab rows, and the M-RoPE constants from any modification.
4. **Compensation LoRA can train on multimodal calibration data** (image+text pairs) without architectural changes to `compensation_lora.py` — the existing KL-on-logits loss propagates gradient through the merger automatically when image inputs flow through the forward pass, because the student's output logits depend on both vision and text pathways.

The bet is that **VL forging is a whitelisting and calibration-data problem, not a loss-function problem.** If this hypothesis holds, the sprint shrinks from "multi-week greenfield" to "~2 weeks of additive engineering on top of the existing methodology."

## Architectural origin

Two existing patterns in the codebase justify this approach:

1. **The compensation LoRA pattern from `COMPENSATION-LORA-DESIGN.md`** — distillation against an unmodified teacher recovers held-out task capability that the surviving heads alone cannot absorb. The same hypothesis applies to vision-language tasks: if the text decoder has been pruned, the merger and vision tower are now feeding a slightly-different downstream context, and the LoRA's job is to learn the delta. This is identical math to the text-only case; only the calibration data changes.
2. **The harness_checks no-fallback discipline** — `assert_explicit_head_dim` and the v2-7B forge's hard preconditions caught the v1 14B disaster by failing loudly on missing config invariants. We extend this discipline to vision: hard-fail at load time if the vision tower path is unrecognized, if `deepstack_visual_indexes` is non-empty (a future Qwen variant we have not validated against), or if the vision token IDs collide with prunable embedding rows.

## Architectural facts (from the architecture survey)

These are the load-bearing facts about Qwen3.5-VL that drive every design decision below. They apply identically to `Qwen3.5-27B` (dense FFN) and `Qwen3.5-35B-A3B` (MoE FFN); the two checkpoints differ only in text-decoder width/depth and dense-vs-MoE FFN.

### Vision tower (`model.visual.*`)
- 27 transformer blocks, `hidden_size: 1152`, `intermediate_size: 4304`, `num_heads: 16`, `gelu_pytorch_tanh` activation — SigLIP-So400m signature
- `patch_size: 16`, `temporal_patch_size: 2`, `spatial_merge_size: 2`
- `num_position_embeddings: 2304` learned (capped — beyond this, spatial tiles must be re-tiled)
- `deepstack_visual_indexes: []` on current checkpoints; non-empty means multi-level injection into text decoder, which would invalidate layer-deletion safety
- ~0.4–0.5B params total
- Plain softmax attention; **not** affected by hybrid attention concerns
- Identical between 27B and 35B-A3B

### Merger / projector
- 2-layer MLP: `LayerNorm → Linear(1152·4 → out_hidden_size) → GELU → Linear`
- `out_hidden_size`: 2048 (35B-A3B) or 5120 (27B) — projects to text decoder hidden size
- `spatial_merge_size: 2` means 2×2 spatial token merge (concat 4 tokens → MLP down)
- Parameter count: ~5–25M depending on `out_hidden_size`
- **Single point of failure between modalities** — pruning, aggressive quantization, or re-initialization severs vision entirely

### Vision-token entry into the text decoder
- `image_token_id: 248056`, `video_token_id: 248057`, `vision_start_token_id: 248053`, `vision_end_token_id: 248054`
- Processor inserts `<vision_start> <image>×N <vision_end>` placeholders
- Visual tower + merger produce N projected embeddings
- Modeling code overrides `get_input_embeddings()` to scatter visual embeddings into the text-embedding tensor at positions where `input_ids == 248056`
- Spatial/temporal position rides on M-RoPE in the text decoder, not on additive PE in the merger

### M-RoPE
- `mrope_interleaved: true`
- `mrope_section: [11, 11, 10]` — t/h/w channel split
- `partial_rotary_factor: 0.25` — only 25% of head dims rotated
- `rope_theta: 1e7`
- **Spatial structure reaches the decoder via M-RoPE.** Any tool that "normalizes" RoPE to standard 1-D will destroy multimodal grounding while text generation looks fine.

### Text decoder hybrid attention (issue #163 territory)
- `layer_types`: `[linear_attention, linear_attention, linear_attention, full_attention] × N`
  - 35B-A3B: × 10 (40 layers total)
  - 27B: × 16 (64 layers total)
- `linear_attention` = Gated DeltaNet (`linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: float32`)
- `full_attention` = Gated softmax with `attn_output_gate: true` (per-head output gate, **non-standard**)
- Issue #163's Strategy A layer-aware defrag applies identically — vision adds no new defrag concern here

### MoE specifics — 35B-A3B only
- `num_experts: 256`, `num_experts_per_tok: 8`
- **`shared_expert_intermediate_size: 512`** — 1 always-on shared expert per layer, **NOT in the routed pool**
- 8 routed + 1 shared = 9 active per token
- `mlp_only_layers: []` — every text decoder layer is MoE
- Vision tower is dense; MoE is text-side only
- **`cpu_expert_prune_v2.py` regex collapses routed and shared experts under the same pattern; pruning the shared expert lobotomizes every layer.** This is a one-line fix but a hard requirement.

### 27B differences
- No `num_experts` field, no MoE — fully dense FFN with `intermediate_size: 17408`
- 64 layers, hybrid attention pattern × 16
- Otherwise identical vision pathway and M-RoPE configuration

## Math

The forge methodology is unchanged at the level of the loss function. Let:

- $T(x_{\text{text}}, x_{\text{img}})$ = unpruned VL teacher
- $S(x_{\text{text}}, x_{\text{img}})$ = pruned VL student (text decoder pruned; vision tower and merger preserved bit-exact)
- $L$ = LoRA parameters on the text decoder's prunable projections
- $S_L(\cdot, \cdot)$ = student with LoRA applied
- $D_{\text{vl}}$ = held-out multimodal calibration distribution (CharXiv + BLINK + MMMU-Pro vision-only)
- $D_{\text{txt}}$ = held-out text-only calibration distribution (HumanEval + GSM8K + MMLU subset)

The compensation training objective:

$$
\mathcal{L}_{\text{distill}}(L) = \mathbb{E}_{(x_{\text{text}}, x_{\text{img}}) \sim D_{\text{vl}} \cup D_{\text{txt}}}\left[ \mathcal{L}_{\text{KD}}(T(x_{\text{text}}, x_{\text{img}}), S_L(x_{\text{text}}, x_{\text{img}})) \right]
$$

with the KL-on-logits formulation that won the v2-7B ablation:

$$
\mathcal{L}_{\text{KD}}^{\text{KL}} = T^2 \cdot D_{\text{KL}}\left( \text{softmax}(T_{\text{logits}} / T) \,\|\, \text{softmax}(S_{L,\text{logits}} / T) \right)
$$

For text-only inputs $x_{\text{img}} = \emptyset$, the forward pass skips the vision tower and the math reduces to the existing text-only compensation_lora.py case. For image-bearing inputs, the vision tower and merger contribute to the student's logits via scatter injection, and the gradient through the LoRA depends on the merger output indirectly. **The merger and vision tower are frozen** — gradient flows through them but does not update them. This is the standard "compensate the language model around a frozen vision encoder" recipe from LLaVA-KD / LLaVA-MoD.

The MSE-on-hidden-states loss is **not** used for VL because hidden states at vision-token positions have a different distribution than text-token positions, and the per-layer MSE would be dominated by whichever side has higher norms. Sticking with KL-on-logits avoids this entirely (logits are a single distribution at every token position regardless of modality).

## Constraints from the rest of the substrate

These are non-negotiable invariants the VL forge must preserve. Each maps to a specific assertion or whitelist entry in the implementation.

1. **Vision tower preserved bit-exact.** No pruning, no quantization, no LoRA, no defrag. The 27 SigLIP blocks, the patch_embed, and the position embedding table at `model.visual.*` are read-only throughout the forge.
2. **Merger preserved bit-exact.** The 2-layer MLP at `model.visual.merger` (or the equivalent attribute) is excluded from every prune/LoRA target list. No pattern in `target_modules` or expert-prune regex may match it.
3. **Vision token vocab rows preserved.** Embedding rows at indices `vision_start_token_id`, `vision_end_token_id`, `image_token_id`, `video_token_id`, plus any other token in `processor.tokenizer.special_tokens_map` are excluded from any embedding-table defrag or vocabulary truncation.
4. **M-RoPE constants preserved.** `mrope_interleaved`, `mrope_section`, `partial_rotary_factor`, `rope_theta` in the saved config.json must match the source model exactly. Any tool that rewrites RoPE parameters fails the post-forge validation.
5. **Shared expert preserved (35B-A3B only).** `model.layers.{i}.mlp.shared_expert.*` is excluded from the routed-expert prune regex. Only `model.layers.{i}.mlp.experts.{j}.*` is in the routed pool.
6. **`attn_output_gate` tensors preserved.** Any per-head output gate parameters must survive defrag and pruning. Tools that drop unrecognized tensors fail the post-forge validation.
7. **Linear-attention (Gated DeltaNet) layers handled correctly.** Issue #163's Strategy A layer-aware defrag applies — full_attention layers are defragged, linear_attention layers are skipped. This is unchanged from the text-only Qwen3.5 case.
8. **`deepstack_visual_indexes == []` precondition.** Hard-fail at load time if the source checkpoint has non-empty deepstack indices. We have not validated layer-deletion safety against multi-level visual injection.
9. **Tokenizer/processor alignment.** Teacher and student must use identical processors (image preprocessor + tokenizer). The compensation distillation requires bit-identical tokenization of text inputs and bit-identical image preprocessing (resize, normalize, patchify) on both sides.

## Stability concerns to validate at small scale before scaling up

Five things that can go wrong, with the validation method for each:

### 1. Vision tower accidentally modified
After every forge operation, hash the vision tower state dict and assert it equals the source model's hash. This catches any accidental write through a misnamed pattern. **Validation:** SHA256 over `model.visual.state_dict()` before and after each forge stage; assert equality.

### 2. Merger accidentally pruned or LoRA-targeted
After loading the LoRA-wrapped student, iterate `student.named_modules()` and assert no module under `model.visual` has been wrapped with a `LoraLayer`. **Validation:** explicit module-tree walk; fail loud on any vision-side LoRA.

### 3. Vision token rows zeroed during embedding defrag
If the embedding table defrag preserves only "active" rows by some heuristic, vision-token rows at indices ~248053-248057 may be marked inactive (they don't appear in text-only calibration data). **Validation:** assert `embed_tokens.weight[image_token_id].norm() > 0` and equals the source row exactly, after every embedding-table touch.

### 4. Multimodal calibration produces no merger gradient
Even with image inputs flowing through the forward pass, if `requires_grad=False` is set too aggressively on the merger, gradient won't propagate through it (which is fine for parameter updates, but we need it to flow through the merger to reach the LoRA weights downstream). **Validation:** at step 1 of training, check that `student.text_decoder.layers[0].self_attn.q_proj.lora_A.weight.grad` is non-zero on a batch where every input has an image. If it's zero, gradient is being blocked at the merger boundary.

### 5. Routed-vs-shared expert misclassification (35B-A3B only)
The expert prune regex must distinguish `mlp.experts.{j}` (routed, prunable) from `mlp.shared_expert` (always-on, never prunable). **Validation:** before pruning, dry-run the regex against a list of all `mlp.*` parameter names from the source model and assert that exactly `num_experts × num_layers` entries match the routed pattern and exactly `num_layers` entries match the shared pattern. Off-by-one fails loudly.

## Phased plan

Each phase produces a publishable artifact and gates the next. Each phase has its own small-scale smoke test before scale-up.

### Phase 1 — Vision-safe text-only path (~3-5 days)

**Goal:** forge `Qwen3.5-27B` (dense, no MoE complications) using the existing text-only methodology, with vision pathway whitelisted as preserved. The output is a compensated 27B that still loads as VL and passes a 10-image vision smoke test bit-exact against the source model on the vision side, while showing the expected text-side compaction and calibration delta.

**Why 27B first, not 35B-A3B:** dense FFN means no shared-expert footgun. We isolate the vision-safety work from the MoE work, validate the whitelist independently, and only then layer MoE complexity on top. 27B is also a better debugging target — fewer moving parts, faster iteration.

**Deliverables:**
- `scripts/vision_safety.py` — module that enumerates "untouchable" parameter names from a VL model config + state dict, supplying the whitelist consumed by every forge stage. Pure read; no model modification.
- `scripts/test_vision_safety.py` — CPU smoke test on a tiny VL model (e.g., `Qwen/Qwen2-VL-2B-Instruct` if available, else a stubbed config) verifying the whitelist correctly identifies the vision tower, merger, vision-token rows, and M-RoPE constants.
- Read-only audit pass over `forge_model.py`, `compensation_lora.py`, `cpu_expert_prune_v2.py` to confirm the 13 breakage points from the impact analysis are all addressable by consulting the whitelist (no architectural surprises).

**Smoke test gate:** `test_vision_safety.py` passes; the whitelist correctly excludes every parameter under `model.visual.*` and the merger; vision token vocab rows are flagged for preservation.

### Phase 2 — Vision-aware compensation calibration (~3-5 days)

**Goal:** extend the compensation training loop to feed multimodal calibration data through the same forward pass, so the merger contributes gradient signal to the LoRA on the text decoder. The output is a compensated 27B trained on CharXiv + BLINK + MMMU-Pro vision-only that shows non-degraded vision benchmark performance vs the source model.

**Deliverables:**
- `scripts/vl_calibration_loader.py` — dataset loader for the held-out trio (CharXiv, BLINK, MMMU-Pro vision-only), wrapping HF Datasets with the Qwen3.5-VL processor for image preprocessing. License-clean (Apache + CC-BY).
- `scripts/test_vl_calibration_loader.py` — verifies one batch loads correctly with image tensors and text input_ids, processor produces the expected `<vision_start> <image>×N <vision_end>` placeholders, and the resulting batch can be fed to a Qwen3.5-VL model for a forward pass without errors.
- A new compensation training entry point `scripts/compensation_lora_vl.py` that imports from `compensation_lora.py` (no edits to that file) and adds: AutoProcessor loading, vision-safe target_modules selection (excludes anything matching the vision_safety whitelist), multimodal batch collator, and the vision-tower-frozen requires_grad setup.

**Smoke test gate:** at step 1 of compensation training, gradient is non-zero on text-decoder LoRA weights when the input batch is image+text; loss decreases monotonically over 30 steps; vision tower SHA256 unchanged; merger SHA256 unchanged.

### Phase 3 — MoE + 35B-A3B (~3-5 days)

**Goal:** apply Phase 1 + 2 to the MoE 35B-A3B target. Validate that the routed/shared expert split is correctly enforced and that compensation LoRA on top of expert pruning works for VL.

**Deliverables:**
- `scripts/cpu_expert_prune_vl.py` — wraps `cpu_expert_prune_v2.py` regex generation with shared-expert exclusion, plus vision_safety whitelist enforcement. Refactored into the canonical file once Move 2 is shipped.
- `scripts/test_expert_prune_vl.py` — dry-run the regex against the 35B-A3B parameter list; assert `256 × 40 = 10240` routed-expert tensor matches and `40` shared-expert tensor matches with the shared pattern (and zero shared-expert matches under the routed pattern).
- Production forge run on `Qwen3.5-35B-A3B` or `Qwen3.5-27B` with the full validated stack.

**Smoke test gate:** all assertions in Phase 1 and Phase 2 pass on the 35B-A3B target; routed expert prune produces expected tensor shape changes; shared expert weights bit-exact; vision tower bit-exact.

### Phase 4 — Refactor into canonical files (~1-2 days)

**Goal:** once Move 2 (text-only Qwen3-Coder-30B-A3B) has shipped and the existing `forge_model.py` / `compensation_lora.py` / `cpu_expert_prune_v2.py` are no longer in active use for in-flight forges, fold the vl_* additive modules into the canonical files. The `vision_safety.py` module stays as a module; the wrappers (`compensation_lora_vl.py`, `cpu_expert_prune_vl.py`) get inlined as `--modality vl` branches in the canonical scripts.

**Deliverables:**
- `forge_model.py`, `compensation_lora.py`, `cpu_expert_prune_v2.py` updated with vision_safety integration; old script wrappers deleted.
- All 13 breakage points from the impact analysis closed and reverified.
- VL-FORGE-DESIGN.md updated with "shipped" notes.

## Coordination with in-flight Move 2 forge

continuum-side Claude is currently running Move 2 (text-only Qwen3-Coder-30B-A3B-Instruct forge) using the unmodified `forge_model.py` / `compensation_lora.py` / `cpu_expert_prune_v2.py`. **Phase 1, 2, and 3 deliverables are all additive new files** — they import from the canonical modules but do not modify them. This guarantees zero risk of breaking the Move 2 run.

Phase 4 (refactor into canonical files) only happens after Move 2 ships and is verified. If Move 2 takes longer than expected, the additive modules remain usable indefinitely; the refactor is a cleanup, not a blocking dependency.

## Success criteria

The VL forge methodology is validated if:

- Phase 1: vision-safety whitelist correctly identifies every untouchable parameter on `Qwen3.5-27B`; CPU smoke test passes without false positives or false negatives.
- Phase 2: a compensated `Qwen3.5-27B` trained on CharXiv + BLINK + MMMU-Pro shows vision benchmark performance within ±2 points of the source model on a held-out test set, while text-side benchmarks show the expected compaction-driven delta within the same calibration tolerance band as the v2-7B work (±3 pt).
- Phase 3: the same is true on `Qwen3.5-35B-A3B`, with the additional constraint that routed-expert pruning has reduced total parameter count by the target ratio and shared experts are bit-exact.
- Phase 4: the forge pipeline is single-source-of-truth again (no parallel `*_vl.py` wrappers) and the existing text-only forges still pass their regression tests.

## Failure mode escalation

1. **Phase 1 vision-safety whitelist misses a parameter** → expand the whitelist generator to walk the model's module tree dynamically rather than relying on name patterns. Re-run smoke test.
2. **Phase 2 merger gradient is zero** → debug the requires_grad propagation; possibly switch to manual hook-based gradient capture on the merger output. If unfixable, fall back to text-only calibration with explicit vision-bench regression testing as the safety net.
3. **Phase 2 vision benchmarks regress** → the LoRA is competing with the merger for output-space coverage. Reduce LoRA rank or add a vision-token-position-only loss term that explicitly preserves the merger's contribution at scatter positions.
4. **Phase 3 shared-expert prune misclassification** → tighten the regex; add an explicit allowlist of shared expert parameter names per layer. This is a code bug, not a methodology failure.
5. **Phase 3 MoE compensation collapses (loss → 0 like the text-only MSE failure)** → add small router entropy regularizer per LLaVA-MoLE; if that doesn't work, fall back to text-only-calibrated compensation and accept whatever vision regression that produces.

Each escalation step is a refinement, not a redesign. The architecture survey gave us enough confidence in the structural separation between modalities that none of these failure modes require revisiting the whitelisting hypothesis.

## Authors

- Joel Teply (Cambrian AI)
- with assistance from Kash (KashCompiler-side Claude, Anthropic)

## License

CC-BY 4.0 (design doc text). Code (`scripts/vision_safety.py`, `scripts/compensation_lora_vl.py`, `scripts/cpu_expert_prune_vl.py`, etc.) under the parent project license.

---

## Appendix A — VL knowledge distillation literature review

Brief grounding for why the existing text-only `compensation_lora.py` KL-on-logits + MSE-on-hidden-states loss formulation ports forward unchanged for VL. The literature on distilling vision-language models is sparse but has converged on the "freeze the vision encoder, distill on text response" pattern that matches our setup.

### MiniVLM (Wang et al., 2020) and DistillVLM (Fang et al., 2021)

Early dense VL distillation work targeting BERT-era VL-BERT and ViLBERT. Both use full-model losses combining output logits + vision-language matching scores (a classification head over (image, text) pairs). **Not directly applicable** to our setup because they distill *both* the vision encoder and the text backbone simultaneously, whereas we keep the vision tower frozen bit-exact and compensate only the text decoder.

### LLaVA-KD (2024) and LLaVA-MoD (2024)

The closest analogues to our setup. Both target LLaVA-1.5 by reducing the LLM backbone (Vicuna-13B → Vicuna-7B in LLaVA-KD; Vicuna-7B → smaller in LLaVA-MoD) **while keeping the CLIP vision encoder frozen**. The distillation loss is computed only on text response logits with image conditioning — i.e., the same forward pass produces a vision-conditioned text output, and the KD loss compares student vs teacher logits at every text-output position.

This is the recipe we want. Three concrete things from these papers:

1. **No explicit vision-token loss term is needed.** Both LLaVA-KD and LLaVA-MoD rely on response-logit KL alone. The vision tower contributes to the loss indirectly through the conditioned text output, but it has no direct gradient target. This validates our Phase 2 design — feeding image inputs through the same forward pass and using the existing KL-on-logits loss is the right starting point.
2. **Temperature on the KL is load-bearing.** Both papers use T=2.0 for the softmax temperature on the KD loss. Higher temperatures (T=4) caused training instability; lower (T=1) reduced the soft-target signal too much. Our v2-7B work also used T=2.0; this transfers cleanly.
3. **The frozen vision encoder must be in eval mode** during distillation. Dropout in the vision tower at training time produces different visual embeddings on the same image across forward passes, which creates a shifting target that the LoRA cannot converge to. `model.visual.eval()` plus a `requires_grad=False` walk over `model.visual.*` parameters is the load-bearing setup.

### MoE distillation: LLaVA-MoLE and MoE-LLaVA (2024)

For Phase 3 (MoE 35B-A3B) the relevant concern is **router collapse during distillation**. When the LoRA on the text decoder starts learning compensation deltas, the router can drift toward routing all tokens to the surviving experts that the LoRA most strongly modifies, which collapses the routing diversity and produces a degenerate post-distillation model.

LLaVA-MoLE's mitigation is a small entropy regularizer added to the routing loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{KD}} - \lambda \cdot \mathbb{E}_{x}\left[ H(\text{router}(x)) \right]
$$

where $H$ is the routing distribution entropy and $\lambda$ is a small constant (~0.01 in their setup). This nudges the router to maintain non-degenerate distributions over the surviving experts while the LoRA learns its compensation deltas.

**For Phase 3, this is the failure-mode escalation #5 in §"Failure mode escalation"** of the main design doc. We don't enable it by default — only if Phase 3 training shows routing collapse (signature: a small number of experts dominate routing on every input within ~50 steps).

### What the literature does NOT cover

- **Distilling MoE VL models.** No published work targets the exact intersection of MoE backbone + VL conditioning + LoRA-only updates. We are the first to attempt this combination in Phase 3.
- **Calibration-aware expert importance for VL pruning.** §4.1.3.4 of the methodology paper applies the calibration-aware metric to text-only MoE; the VL-specific extension (calibration corpus must include image+text pairs to measure expert activation correctly) is novel.
- **Compensation LoRA against a frozen-vision-encoder teacher.** LLaVA-KD distills the LLM backbone wholesale; we distill only LoRA-adapter deltas. This is a smaller hypothesis class and expected to be more stable, but no published work has validated it.

### Net implication for the design

The existing `compensation_lora.py` loss formulation (KL-on-logits at T=2.0) ports forward unchanged for Phase 2. The only new requirement is feeding image inputs through the same forward pass via the AutoProcessor, plus the vision tower frozen-eval setup. Phase 3 may require the entropy regularizer; Phase 2 will not.

---

## Appendix B — Multimodal calibration dataset survey

Full survey of license-clean multimodal calibration data candidates. Used to populate the Phase 2 calibration corpus selection. Annotated with license, size, modality, suitability ranking, and reasons for inclusion or rejection.

### Top picks for held-out VL calibration (commercial-safe)

Ranked by suitability for distillation calibration use. These are the datasets to actually use in Phase 2.

| Rank | Dataset | License | Size | Modality | Why | URL |
|---|---|---|---|---|---|---|
| 1 | **DocVQA private test** | CC-BY 4.0 | ~5k held-out questions | doc image + QA | Private test split is exactly what calibration wants — teacher activations meaningful and unmemorized | docvqa.org/datasets |
| 2 | **CharXiv** (2024) | CC-BY-SA 4.0 | 2,323 charts, ~10k QAs | scientific chart + QA | Recent (2024), deliberately hard, low memorization in pre-2025 training corpora. Excellent calibration candidate. | charxiv.github.io |
| 3 | **MMMU-Pro vision-only** (2024) | Apache 2.0 | ~1.7k examples | college-level multimodal | Newer/harder than original MMMU; vision-only split is held-out by construction | mmmu-benchmark.github.io |
| 4 | **BLINK** (2024) | Apache 2.0 | 3,807 MCQ across 14 perceptual tasks | low-level visual perception | Released mid-2024, niche enough to be held-out for many checkpoints | zeyofu.github.io/blink |
| 5 | **InfographicVQA** | CC-BY 4.0 | ~3k held-out test | infographic + QA | Less saturated than ChartQA; private test split | docvqa.org/datasets/infographicvqa |
| 6 | **Perception Test** (DeepMind 2023) | CC-BY 4.0 | 11.6k videos | video + temporal/multiple-choice QA | Cleanest video benchmark license-wise; held-out test labels on server | github.com/google-deepmind/perception_test |
| 7 | **TempCompass** (2024) | MIT | ~7.5k QAs over 410 videos | video, pure temporal reasoning | Niche, recent, isolated from spatial cues — strong held-out signal | github.com/llyx97/TempCompass |
| 8 | **MVBench** (2024) | MIT | 4,000 QAs across 20 temporal task types | video, action/scene/counting | Standard but clean MIT license | github.com/OpenGVLab/Ask-Anything |

**Recommended Phase 2 calibration corpus:** CharXiv + BLINK + MMMU-Pro vision-only for still-image, plus Perception Test + TempCompass for video. All Apache/MIT/CC-BY, all license-clean for commercial derivatives, all sufficiently held-out from typical 2024-trained-VLM corpora.

### License-fraught (commercial blockers)

Use only if the Phase 2 forge target is research-only, not commercial:

| Dataset | License problem |
|---|---|
| **ScienceQA** | CC-BY-NC-SA 4.0 — NC clause kills commercial derivatives |
| **Video-MME** | CC-BY-NC-SA 4.0 — same NC clause |
| **RealWorldQA** | CC-BY-ND 4.0 — ND blocks derivative datasets, OK as frozen activation input only |
| **EgoSchema** | Ego4D non-OSI license, requires signature |

### Likely-already-in-training-data (poor calibration signal)

| Dataset | Concern |
|---|---|
| **OBELICS** | Likely in Qwen-VL training mix; URL rot risk |
| **COCO Captions, VQAv2** | In every VLM's training set; useless as held-out |
| **MSR-VTT** | Heavily memorized + noncommercial clause |

### Hard reject (do not use under any condition)

- **LAION-***: license-fraught, scraped, deprecated by HF
- **WebVid-10M**: withdrawn over licensing
- **Any "scraped from web" datasets without per-image license attestation**

---

## Appendix C — Pipeline VL impact analysis: enumerated breakage points

Full enumeration of every place in the existing forge pipeline that breaks on a VL model. 13 distinct breakage points across 3 files: **5 hard CRASHes (loud failure), 8 SILENT BUGs (run but destroy vision capability)**. The silent bugs are the higher-risk class because the post-forge artifact appears to load and produce text correctly while its multimodal capability is structurally broken.

### `forge_model.py`

| Line(s) | Assumption | Class | Impact |
|---|---|---|---|
| 140 | `head_dim = h // nh` computed from text config only, ignoring `vision_config` | SILENT BUG | Vision tower heads ignored in importance ranking |
| 174–177 | `get_layers()` checks only `model.model.layers` (Qwen/Llama style) or `model.transformer.h` (GPT2 style); falls through on VL module trees | CRASH | `RuntimeError("Cannot find model layers")` at line 178 on any VL model |
| 520–541 | `compute_head_importance()` iterates only modules with `self_attn` or `attn`, hardcodes `q_proj` | SILENT BUG | Vision tower attention blocks excluded from importance ranking; vision projector heads scored as zero-importance and may be accidentally pruned |
| 643–688 | `compute_activation_importance()` installs hooks only on `o_proj` within `layers[li]` | SILENT BUG | Vision tower projector outputs never hooked; activation-based importance never measured for vision pathway |
| 829, 845 | `prune_by_zeroing()` hardcodes `["q_proj", "o_proj"]` and `["k_proj", "v_proj"]` | SILENT BUG | Vision projector linear layers (different names: `fc_in`, `fc_out`, etc.) never zeroed; dead heads leak into vision inference |
| 866 | `o_dim_per_head = o_proj.weight.shape[1] // info["num_heads"]` hardcodes text num_heads | CRASH | Division by text head count fails for vision projector with different head structure |
| 680–688 | Strategy A skips non-`full_attention` layers via `config.layer_types`, but VL configs put `layer_types` in `text_config`, not at top level | SILENT BUG | Vision tower modules never recognized as different from text full-attention layers; wrongly hooked |

### `compensation_lora.py`

| Line(s) | Assumption | Class | Impact |
|---|---|---|---|
| 74, 123 | `AutoModelForCausalLM.from_pretrained()` without `AutoProcessor`; only tokenizer loaded | CRASH | Vision tower and image processor not loaded; forward pass crashes on image inputs |
| 304–310 | `JsonlTextDataset.__getitem__()` tokenizes text only, no image processor | SILENT BUG | Calibration data missing image inputs entirely; distillation loss meaningless |
| 356–365 | Default `target_modules` hardcode text attention names: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | SILENT BUG | Vision projector linear layers (with different names) never receive LoRA compensation |
| 401–414 | `assert teacher.config.hidden_size == student.config.hidden_size` at top level | CRASH | For VL configs, top-level `hidden_size` is in `text_config.hidden_size`; `vision_config.hidden_size` is different. Assertion may fail spuriously or silently pass on wrong field |

### `cpu_expert_prune_v2.py`

| Line(s) | Assumption | Class | Impact |
|---|---|---|---|
| 80–81 | `ROUTER_GATE_RE = r"^model\.layers\.(\d+)\.mlp\.gate\.weight$"` and `EXPERT_TENSOR_RE = r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.([a-z_]+)\.weight$"` assume unfused MoE layout in text decoder only | CRASH | Vision tower MoE tensors (if present, e.g. future Qwen3.5-VL variants with vision experts) silently skipped; **also matches `mlp.shared_expert.*` if not carefully bounded** — see §5 of main design doc on shared-expert preservation |
| 374 | `tc = cfg.get("text_config", cfg)` extracts MoE config from text_config only | SILENT BUG | Vision MoE config (if any) in `vision_config` never inspected; post-prune `config.json` is internally inconsistent |
| 345–349 | `config.json` mutation only updates text MoE params | SILENT BUG | If vision_config has experts, num_experts post-prune is wrong; inference crashes when router expects pruned experts |

### Crash-vs-silent breakdown

- **CRASH (5):** lines 174–177, 866 in `forge_model.py`; lines 74/123, 401–414 in `compensation_lora.py`; lines 80–81 in `cpu_expert_prune_v2.py`. These fail loudly at load or first forward pass — easy to detect, easy to fix.
- **SILENT BUG (8):** lines 140, 520–541, 643–688, 829/845, 680–688 in `forge_model.py`; lines 304–310, 356–365 in `compensation_lora.py`; lines 374, 345–349 in `cpu_expert_prune_v2.py`. These produce a model that loads, runs, and generates text — but vision capability is silently broken. **These are the higher-risk class and the reason the vision_safety whitelist module exists** — every silent bug above can be closed by consulting the whitelist before adding parameters to prune/LoRA target lists.

### Net design implication

All 13 breakage points are addressable by integrating the `vision_safety.py` whitelist (Phase 1 deliverable) into each prune/LoRA target selection step. The 5 CRASHes also require restructuring `get_layers()` and the assertion logic to be modality-aware. The 8 SILENT BUGs require restructuring the importance/prune target selectors to consult the whitelist before matching on parameter names. Both classes of fix are mechanical once the whitelist is in place; no new methodology is needed.

This is what makes the VL forge a 1-2 week sprint instead of a multi-week greenfield effort: the breakages are enumerable, the fix is uniform (consult the whitelist), and the methodology (importance metric + compensation LoRA) ports forward from the text-only work unchanged.
