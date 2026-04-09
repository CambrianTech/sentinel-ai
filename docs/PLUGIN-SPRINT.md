# Plugin Sprint: family-adapter architecture for sentinel-ai

> **Status:** mid-sprint, 2026-04-08. 9 commits landed on `cross-arch-portability-fixes`,
> 8-step roadmap remaining. Read this entire doc before resuming after a crash —
> it captures what got built, why, what's still pending, and the order of operations.
>
> **Companion docs:** [`VL-FORGE-DESIGN.md`](VL-FORGE-DESIGN.md) (vision-safety scaffolding),
> [`COMPENSATION-LORA-DESIGN.md`](COMPENSATION-LORA-DESIGN.md) (§4.1.3.3 distillation),
> [`continuum/docs/architecture/FORGE-ALLOY-DOMAIN-EXTENSIBILITY.md`](../../continuum/docs/architecture/FORGE-ALLOY-DOMAIN-EXTENSIBILITY.md)
> (schema-side architecture proposal that's mid-refactor).

---

## TL;DR

The forge pipeline used to be one mutable script (`alloy_executor.py` + `forge_model.py`
+ `cpu_expert_prune_v2.py`) that grew per-family `if architectures[0] == ...` branches
every time a new model class shipped. This drift broke the legacy Qwen3.5 catalog's
bit-equivalent reproducibility and was the immediate cause of the 9 plugin-sprint
commits below.

The fix is **two-axis dispatch**:

```
Axis 1 (already existed): stage type → StageExecutor
    prune        → PruneExecutor
    train/lora   → TrainExecutor
    expert-prune → ExpertPruneExecutor
    quant/eval/publish/... → output stage executors

Axis 2 (this sprint):     source.architecture → FamilyAdapter
    qwen3_5    → Qwen3DenseAdapter
    qwen2      → Qwen2DenseAdapter
    qwen3_moe  → Qwen3MoEAdapter
    olmoe      → OlmoeAdapter
```

Each `StageExecutor.execute()` reads the alloy's `source.architecture`, looks up
the family adapter, calls the matching method, returns the mutated context. **No
shared script ever branches on architecture.** New family = new file in
`scripts/adapters/` plus one import line. Old families stay frozen forever so
older alloys keep reproducing bit-identically.

The reproducibility test (`tests/reproducibility/`) is the gate. It parametrizes
over every published `continuum-ai/*` artifact and validates four tiers:

| Tier | What it checks | Status |
|---|---|---|
| **Tier 1** dispatch | every alloy resolves to a clean adapter chain | **19/19 green** |
| **Tier 2** byte-equivalent re-forge | run the alloy → produce a model → modelHash matches | not yet wired (needs 5090) |
| **Tier 3** sample-hash | per-problem JSONL bytes hash to alloy-claimed sha256 | 7 green / 2 xfailed (schema gap) |
| **Tier 4** canonical pass@1 | re-score JSONLs with evalplus's CLI → matches alloy headline | 14 green |

**40 passed, 0 skipped, 2 xfailed across the whole reproducibility module.**

---

## What was built (the 9 plugin-sprint commits)

Branch: `cross-arch-portability-fixes` on `/Volumes/FlashGordon/cambrian/sentinel-ai/`

| Commit | Purpose |
|---|---|
| `f82773b` | wip-snapshot: preserve uncommitted vision_safety + Dockerfile + install.sh from before the drive crash |
| `04e9cc5` | adapters: introduce family-adapter dispatch + Tier 1 reproducibility test (Qwen3DenseAdapter, dispatch+registry, base ABC, test parametrized over 14 alloys) |
| `4d087d4` | wire transform-stage executors to delegate to family adapter (PruneExecutor / TrainExecutor / ExpertPruneExecutor become ~5-line dispatchers; the model-touching code moves into Qwen3DenseAdapter) |
| `9b386fb` | Qwen3MoEAdapter — flips qwen3-coder-30b-a3b-compacted to green at Tier 1 |
| `d369ee6` | OlmoeAdapter — flips olmoe-1b-7b-compacted-5b to green |
| `7bab0c0` | Qwen2DenseAdapter — flips qwen2.5-coder-7b-compacted to green |
| `1bc32d2` | Tier 4 evalplus re-scoring + fix non-canonical pass@1 publish bugs (the morning's flagship humaneval_plus 86.0 vs canonical 85.4 finding) |
| `d37cf12` | republish_alloy_only.py — focused metadata-only HF re-upload tool used to fix the qwen3-coder humaneval_plus values in place |
| `<latest>` | backfill alloys for ALL 17 published continuum-ai/* artifacts (3 qwen2.5 + qwen3.5-27b parent + 4 downstream variants) |

Working tree clean. Wip preservation pointers exist:
- `wip/pre-plugin-sprint-2026-04-08` → `f82773b` (sentinel-ai)
- `wip/types-additive-checkpoint-bd4349d` → `bd4349d` (forge-alloy)

---

## Architecture

### Repository layout (after the sprint)

```
sentinel-ai/
├── scripts/
│   ├── adapters/                          ← NEW: family-adapter package
│   │   ├── __init__.py                    ← registers concrete adapters
│   │   ├── base.py                        ← FamilyAdapter ABC + AdapterCall + STAGE_METHOD_MAP + REQUIRES_FAMILY_OVERRIDE
│   │   ├── registry.py                    ← AdapterRegistry singleton, strict architecture-string lookup
│   │   ├── dispatch.py                    ← resolve_adapter_chain(alloy) → list[AdapterCall]
│   │   ├── qwen3_dense.py                 ← Qwen3DenseAdapter for qwen3_5 (REAL Tier 2 bodies for prune/train)
│   │   ├── qwen2_dense.py                 ← Qwen2DenseAdapter for qwen2 (Tier 1 stubs, will Tier-2 wire same as qwen3_dense)
│   │   ├── qwen3_moe.py                   ← Qwen3MoEAdapter for qwen3_moe (Tier 1 stubs for expert-activation-profile / expert-prune)
│   │   └── olmoe.py                       ← OlmoeAdapter for olmoe (Tier 1 stubs, parallel to qwen3_moe)
│   ├── stages/
│   │   ├── transform_stages.py            ← REFACTORED: PruneExecutor / TrainExecutor / ExpertPruneExecutor are now 5-line dispatchers
│   │   ├── output_stages.py               ← bug fix: _parse_evalplus_output now section-aware (was overwriting humaneval+ over humaneval)
│   │   └── ... (input_stages, registry.py, base.py unchanged)
│   ├── alloy_executor.py                  ← unchanged surface; the executors it invokes now delegate to adapters
│   ├── add_benchmark.py                   ← FIXED: _load_evalplus_results delegates to canonical scorer (was reading nonexistent JSON keys)
│   ├── publish_model.py                   ← unchanged (modelHash convention TBD — see roadmap step 7)
│   ├── alloy_to_card.py                   ← unchanged
│   ├── backfill_alloy_from_results.py     ← NEW: synthesize alloy from legacy forging_results.json
│   ├── derive_alloy_from_parent.py        ← NEW: synthesize alloy for downstream variant from parent + derivation stage
│   ├── republish_alloy_only.py            ← NEW: focused metadata-only HF re-upload (alloy + README + QR), supports backfill mode
│   └── (existing: forge_model.py, cpu_expert_prune_v2.py, expert_activation_profile.py, compensation_lora.py, eval_with_calibration.py, ...)
├── tests/
│   └── reproducibility/                   ← NEW: parametrized over the published catalog
│       ├── _humaneval_scorer.py           ← canonical evalplus CLI wrapper with macOS reliability_guard + fork-mp workaround
│       ├── _cache/
│       │   ├── continuum-ai_*.json        ← pinned alloy snapshots (committed; the test asserts against these bytes)
│       │   └── samples/
│       │       ├── *.jsonl                ← pinned per-problem eval samples (committed)
│       │       └── .gitignore             ← excludes *_eval_results.json (regenerated each run)
│       ├── test_published_alloys_dispatch.py        ← Tier 1
│       ├── test_published_alloys_sample_hashes.py   ← Tier 3
│       └── test_published_alloys_scoring.py         ← Tier 4
└── docs/
    ├── PLUGIN-SPRINT.md                   ← THIS FILE
    ├── VL-FORGE-DESIGN.md                 ← vision-safety scaffolding (untouched this sprint)
    └── ... (other docs unchanged)
```

### Family adapter contract (`scripts/adapters/base.py`)

```python
class FamilyAdapter(ABC):
    architectures: tuple[str, ...] = ()  # subclass MUST set; registry keys off this

    # Methods that REQUIRE per-family override (raise NotImplementedError on base):
    def prune(self, ctx, **params): ...                    # dense head pruning
    def train(self, ctx, **params): ...                    # LoRA / recovery training (handles compensation distillation too via 'teacher' field)
    def expert_prune(self, ctx, **params): ...             # MoE expert removal
    def expert_activation_profile(self, ctx, **params): .. # §4.1.3.4 calibration-aware metric profiling
    def compensation_lora(self, ctx, **params): ...        # §4.1.3.3 KL distillation
    def context_extend(self, ctx, **params): ...           # YaRN / NTK / etc.
    def modality(self, ctx, **params): ...                 # vision/audio encoder attach

    # Methods that are family-agnostic by default (return ctx unchanged):
    def quant(self, ctx, **params): ...                    # GGUF / MLX / safetensors / ONNX
    def eval(self, ctx, **params): ...                     # benchmark eval — TODO: adapter-driven runner registry per roadmap step 4
    def publish(self, ctx, **params): ...
    def package(self, ctx, **params): ...
    def deploy(self, ctx, **params): ...
    def deliver(self, ctx, **params): ...
    def source_config(self, ctx, **params): ...

    REQUIRES_FAMILY_OVERRIDE: frozenset[str]  # which methods MUST be overridden
    STAGE_METHOD_MAP: dict[str, str]          # alloy stage type → method name
```

### Dispatch flow

```
alloy.json
    ↓
alloy_executor.execute_alloy()
    ↓
for stage in stages:
    create_executor(stage)            ← scripts/stages/registry.py
        ↓
    PruneExecutor.execute(ctx)        ← scripts/stages/transform_stages.py
        ↓
    family = resolve_family_adapter(  ← scripts/adapters/registry.py
        ctx.alloy['source']['architecture']
    )
        ↓
    family.prune(ctx, **stage_params) ← e.g. Qwen3DenseAdapter.prune
        ↓                              ← lazy imports torch + forge_model
    forge_model.compute_head_importance(...)
    forge_model.prune(...)
    defrag_inline.defrag_live_model(...)
        ↓
    return mutated ctx
```

### What's REAL Tier 2 today vs. what's stubbed

| Adapter | Method | Status |
|---|---|---|
| `Qwen3DenseAdapter` | `prune` | **REAL** (lazy imports forge_model.prune + defrag_inline.defrag_live_model) |
| `Qwen3DenseAdapter` | `train` | **REAL** (lazy imports forge_model.train_lora) |
| `Qwen3DenseAdapter` | `context_extend` | Tier 2 stub (returns ctx, logs intent) |
| `Qwen2DenseAdapter` | `prune` | Tier 2 stub (raises if ctx.model is non-None) — same code as Qwen3 wants extracting to base |
| `Qwen2DenseAdapter` | `train` | Tier 2 stub (handles `teacher` field for compensation distillation in dispatch logic, but body raises) |
| `Qwen3MoEAdapter` | `expert_activation_profile` | Tier 2 stub |
| `Qwen3MoEAdapter` | `expert_prune` | Tier 2 stub |
| `OlmoeAdapter` | `expert_activation_profile` | Tier 2 stub (parallel to Qwen3MoE) |
| `OlmoeAdapter` | `expert_prune` | Tier 2 stub (parallel to Qwen3MoE) |

All 4 adapters return ctx cleanly when `ctx.model is None` (the dispatch-only path
the Tier 1 test exercises). Tier 2 wiring is roadmap step 3 below.

---

## Live HuggingFace state (post-sprint)

Every published continuum-ai/* model artifact has a forge-alloy on HF:

| Artifact | Architecture | Adapter | Alloy provenance | Notes |
|---|---|---|---|---|
| `qwen2.5-0.5b-general-forged` | qwen2 | Qwen2Dense | **backfilled** from forging_results.json | alloyHash `a3750da128ba76f0` |
| `qwen2.5-1.5b-general-forged` | qwen2 | Qwen2Dense | **backfilled** | alloyHash `f024d59a481e9032` |
| `qwen2.5-3b-general-forged` | qwen2 | Qwen2Dense | **backfilled** | alloyHash `a13bcfcdc2c8652a` |
| `qwen2.5-coder-7b-compacted` | qwen2 | Qwen2Dense | shipped (§4.1.3.3 anchor) | the v2-7b dense compensated artifact |
| `qwen3.5-0.8b-general-forged` | qwen3_5 | Qwen3Dense | shipped | |
| `qwen3.5-2b-general-forged` | qwen3_5 | Qwen3Dense | shipped | |
| `qwen3.5-4b-general-forged` | qwen3_5 | Qwen3Dense | shipped | |
| `qwen3.5-4b-code-forged` | qwen3_5 | Qwen3Dense | shipped | |
| `qwen3.5-4b-code-128k-forged` | qwen3_5 | Qwen3Dense | shipped | uses context-extend stage |
| `qwen3.5-9b-general-forged` | qwen3_5 | Qwen3Dense | shipped | highest downloads (2.5K) |
| `qwen3.5-27b-code-forged` | qwen3_5 | Qwen3Dense | **backfilled** | alloyHash `80a26f0ec24dfc1e`, parent of 2 derivatives |
| `qwen3.5-4b-code-forged-defragged` | qwen3_5 | Qwen3Dense | **derived** from `qwen3.5-4b-code-forged` | alloyHash `62f1107fb6142943`, package stage |
| `qwen3.5-4b-code-forged-GGUF` | qwen3_5 | Qwen3Dense | **derived** | alloyHash `f7f4f6ddf29019d2`, quant stage (Q4_K_M + Q8_0) |
| `qwen3.5-27b-code-forged-defragged` | qwen3_5 | Qwen3Dense | **derived** from `qwen3.5-27b-code-forged` | alloyHash `f3e68ab40f644c9a`, package stage |
| `qwen3.5-27b-code-forged-mlx-4bit` | qwen3_5 | Qwen3Dense | **derived** | alloyHash `6ca79c62b879cd4c`, quant stage (mlx 4bit) |
| `qwen3-coder-30b-a3b-compacted-19b-256k` | qwen3_moe | Qwen3MoE | shipped (§4.1.3.4 anchor) | **CORRECTED** in commit 1bc32d2 from `aa61c4bdf463847c` to `011970c80c2f3429`. Also v1.0.1 with `humaneval_plus` corrected to canonical evalplus pass@1 (was 86.0 from non-canonical convention, now 85.4) |
| `olmoe-1b-7b-compacted-5b` | olmoe | OLMoE | shipped (§4.1.3.4 cross-arch anchor) | |

The 18th continuum-ai repo is `experiential-plasticity-paper` (paper, not a model)
and is intentionally excluded from the test catalog.

---

## modelHash conventions (currently TWO — needs unification, roadmap step 7)

**Convention A — `publish_model.hash_model_weights` (legacy):**
```python
sha256(concat(shard_bytes for shard in sorted(*.safetensors)))
```
Used by: `publish_model.py` for freshly-forged artifacts.
Pros: trivial to verify with `cat *.safetensors | sha256sum`.
Cons: requires downloading all shards to verify; doesn't preserve per-shard
attestation; not reproducible from HF metadata alone.

**Convention B — `backfill_alloy_from_results._model_hash_from_shard_hashes` (new):**
```python
sha256(canonical_json([{"filename", "sha256"}, ...]))  # sorted by filename
```
where each per-shard sha256 comes from HuggingFace's LFS metadata API
(`?blobs=true` returns `siblings[].lfs.sha256` for each file).
Used by: `backfill_alloy_from_results.py` and `derive_alloy_from_parent.py` for
the 8 backfilled artifacts in the previous commit.
Pros: reproducible from HF metadata alone (no downloads), preserves per-shard
attestation in `integrity.fileHashes[]`, works for any size repo (the 27B's
11×5GB shards were hashed in seconds).
Cons: not the same string as convention A on the same bytes.

**The inconsistency:** today, the freshly-forged `qwen3-coder-30b-a3b-compacted-19b-256k`
artifact's `modelHash` uses convention A (`sha256:236af12e...`), and the
backfilled artifacts use convention B (`sha256:3f1fd0b9...`). Both attest the
same underlying bytes but produce different hash strings, so a single verifier
has to know which convention each alloy uses.

**Roadmap step 7 unifies them.** The plan is to switch `publish_model.py` to
convention B, write a one-shot migration that re-stamps existing alloys' modelHash
field with the new convention (the per-shard list is already preserved on
backfilled alloys; the freshly-forged ones need a one-time recompute), and
document convention B as the only modelHash convention going forward. Convention
A's "concat-and-sha" property is preserved by `integrity.fileHashes[]` letting
anyone reconstruct it externally.

---

## The 8-step "correct architecture" roadmap

### Step 1 — Extract `QwenDenseBase`

**Why:** Qwen2DenseAdapter and Qwen3DenseAdapter both call `forge_model.prune` and
`forge_model.train_lora` the same way at the Tier 2 layer. Two siblings exist,
both with proven Tier 1 dispatch behavior. The OOP rule says: extract the base
NOW, not before. Adding a third sibling (Qwen3.5-VL or any future Qwen-family
dense forge) without extracting first means three parallel bodies.

**Plan:**
1. New file `scripts/adapters/qwen_dense_base.py` defining `QwenDenseBase(FamilyAdapter)`
   with the real `prune` and `train` bodies that currently live in `qwen3_dense.py`.
2. `Qwen3DenseAdapter(QwenDenseBase)` — keeps `architectures = ("qwen3_5",)` and
   `context_extend()` override (for the 4b-code-128k variant). Body methods
   are inherited.
3. `Qwen2DenseAdapter(QwenDenseBase)` — keeps `architectures = ("qwen2",)` and
   the `train()` override that handles the `teacher` field for §4.1.3.3
   compensation distillation (this is the 20% that differs).
4. Re-run all reproducibility tests → must stay 19/19 green at Tier 1.
5. Per the never-lose-work rule: don't delete the old bodies, move them as a refactor.

**Acceptance:** test stays green, the qwen3_dense.py + qwen2_dense.py files
shrink to ~30 lines each (just architectures tuple + the family-specific overrides).

### Step 2 — Extract `MoEUnfusedExpertsBase`

**Why:** Qwen3MoEAdapter and OlmoeAdapter have identical method shapes for
`expert_activation_profile` and `expert_prune`. Both will Tier-2-wire to the same
scripts (`expert_activation_profile.py`, `cpu_expert_prune_v2.py`). Both target
unfused MoE expert layouts (`mlp.experts.{e}.{gate,up,down}_proj` style). The
differences are layer geometry (48×128 vs 16×64) and architecture string. Same
extraction-after-two-siblings logic as Step 1.

**Plan:**
1. New file `scripts/adapters/moe_unfused_base.py` defining
   `MoEUnfusedExpertsBase(FamilyAdapter)` with the lazy-import bodies for
   `expert_activation_profile` and `expert_prune`.
2. `Qwen3MoEAdapter(MoEUnfusedExpertsBase)` — keeps `architectures = ("qwen3_moe",)`.
3. `OlmoeAdapter(MoEUnfusedExpertsBase)` — keeps `architectures = ("olmoe",)`.
4. Future Mixtral / Granite / DeepSeek-V2 adapters inherit from this base when
   they ship.
5. Re-run reproducibility tests → 19/19 green.

**Note:** Mixtral / Phi-MoE / Granite-MoE / DeepSeek-V2 all use DIFFERENT module-tree
layouts (block_sparse_moe vs mlp.experts vs granite-fused vs deepseek-routed-shared).
The base class should NOT bake in a specific tensor walk — it should use the
`expertTensorLayout` field from the alloy stage to dispatch to the right walk
inside the base. That's the §4.1.3.4 design intent. If extraction reveals the
walk doesn't fit cleanly under one base, the right answer is to make `expert_prune`
itself dispatch off `expertTensorLayout` rather than to add per-family branches.

### Step 3 — Tier 2 wiring for the MoE adapters

**Why:** the morning's flagship qwen3-coder-30b-a3b-compacted-19b-256k forge run
exists today but it ran via direct CLI invocations of `expert_activation_profile.py`
and `cpu_expert_prune_v2.py --importance-json`. The adapter system can DISPATCH
that alloy at Tier 1 but if you actually try to execute it via `alloy_executor.py`,
the methods raise NotImplementedError. Tier 2 wiring closes that gap so the alloy
file is the single entry point for re-running the forge.

**Plan:**
1. `expert_activation_profile.py` is currently a CLI script. Refactor it to expose
   a `profile_experts(model, calibration_corpus_path, **opts) -> ImportanceJSON`
   function while keeping the CLI wrapper for backward compatibility.
2. `cpu_expert_prune_v2.py --importance-json` is also CLI-only. Refactor to
   `prune_experts_from_importance(model, importance_json_path, **opts) -> PrunedModel`.
3. `MoEUnfusedExpertsBase.expert_activation_profile` calls the new function via
   lazy import. Same for `expert_prune`.
4. Both methods short-circuit cleanly when ctx.model is None (Tier 1 path stays
   working).
5. Tier 2 reproducibility test (NEW: `test_published_alloys_re_forge.py`) runs
   ONLY on a 5090 — uses pytest's `--gpu-required` mark or a `RUN_TIER_2` env
   var. Loads each MoE alloy, executes the chain end-to-end, asserts the
   produced safetensors hash matches the alloy's modelHash field. This is the
   "gold standard" reproducibility gate; runs on BigMama.
6. The dense path (Qwen3DenseAdapter / Qwen2DenseAdapter — once they inherit
   from QwenDenseBase) will need similar Tier 2 wiring for `prune` and `train`,
   but Qwen3DenseAdapter already has REAL bodies from commit `4d087d4` so this
   is a smaller delta — mostly making sure the lazy imports don't break under
   the new inheritance.

**Acceptance:** Tier 1 stays 19/19 green on Mac. Tier 2 (when run on BigMama)
asserts byte-equivalent re-forge for at least one alloy.

### Step 4 — Eval-runner registry on family adapters

**Why:** the family adapter's `.eval()` method is currently a no-op default that
returns ctx unchanged. The actual eval today happens via the standalone
`tests/reproducibility/_humaneval_scorer.py`. That works for HumanEval but the
methodology paper's frontier targets (Qwen3-Coder-480B, DeepSeek-V3.1, Mixtral
8x22B) all use SWE-Bench Pro / LiveCodeBench v6 / Aider-Polyglot — not HumanEval.
There's no place to plug in those benchmark runners today. **This is the gap
that actually blocks frontier targets.**

**Plan:**
1. New module `scripts/eval_runners/` with one file per benchmark suite:
   ```
   eval_runners/
   ├── __init__.py            ← BenchmarkRunnerRegistry singleton
   ├── base.py                ← BenchmarkRunner ABC: name, score(samples_path) → ScoreResult
   ├── humaneval.py           ← HumanEvalRunner — wraps _humaneval_scorer.py
   ├── humaneval_plus.py      ← HumanEvalPlusRunner — same scorer, plus-test path
   ├── livecodebench.py       ← LiveCodeBenchRunner — when v6 wiring in eval_with_calibration.py is ready
   ├── swe_bench.py           ← SWEBenchRunner — STUB for now, lights up before Qwen3-Coder-480B
   ├── mmlu.py                ← MMLURunner via lm-eval-harness
   ├── mmlu_pro.py            ← MMLUProRunner
   ├── mmmu.py                ← MMMURunner — STUB until first VL artifact ships
   └── ...
   ```
2. `BenchmarkRunnerRegistry.register("humaneval", HumanEvalRunner)` etc.
3. `FamilyAdapter.eval(ctx, **params)` becomes a real default that:
   a. Reads `params['benchmarks']` (list of benchmark dicts with `name` field)
   b. For each, looks up the runner via registry
   c. If found, runs it; if not, raises with a clear "benchmark X has no
      registered runner — add one in scripts/eval_runners/" message.
4. Family adapters can OVERRIDE `.eval()` if they need family-specific
   evaluation (e.g. a Qwen3VLAdapter might want to attach an image
   preprocessor before delegating to the base eval). Most won't need to.
5. Update Tier 4 reproducibility test to use the new registry rather than
   importing the scorer directly. `_humaneval_scorer.py` becomes an
   implementation detail of `eval_runners/humaneval.py`.
6. Document the registry in this file and in adapter docs.

**Acceptance:** Tier 4 test still passes (using the registry path), the registry
has at least 2 runners (humaneval + humaneval_plus), and adding a new benchmark
is one new file in `scripts/eval_runners/`.

### Step 5 — forge-alloy `llm-forge` domain extension (cross-repo) ✓ (forge-alloy commit `4fd715e`)

**Status:** package + registration mechanism + LlmForgeDomain (re-exporting
from `forge_alloy.types` while the universal-core extraction lands as a
separate refactor commit) + photo-provenance + ticketing stubs + the
regression test gate against all 3 published continuum-ai/* alloys with
eval samples — all green. Per TDD, the `python/tests/test_domain_extension_layout.py`
test is the contract spec; 17 of 17 pass.

**Schema gaps caught and fixed inline by the regression gate:**
- `AlloyHardware.deviceTargets` (every published alloy carries it; was being
  silently dropped on validation)
- `AlloyResults.forgedParamsB` + `activeParamsB` (MoE-specific param counts
  on the morning's qwen3-coder-30b-a3b and OLMoE flagships)
- `BenchmarkResult.{score, baseScore, delta, calibrated, samplesPath,
  baseSamplesPath, resultHash, baseResultHash, metric}` — first-class fields
  that the publish pipeline + Tier 4 reproducibility test both consume
  but the schema was hiding behind a generic `metrics` open dict
- `model_config.extra='allow'` on every BaseModel so artifact-specific
  extras (notes, methodology anchor URLs, fourRunProgression,
  lossFunctionAblation, etc.) round-trip preserved without enumerating
  every possible artifact's extras in the schema

**What's still pending under Step 5** (lands as a follow-up refactor commit
that's a pure move, no new behavior):
- Move the actual class definitions for ML stage types out of
  `forge_alloy/types.py` and INTO `forge_alloy/domains/llm_forge.py`.
  Today the latter re-exports them; the eventual end state has the
  definitions live in the domain extension and the universal core
  contains only `ForgeAlloy`, `AlloySource`, `AlloyTarget`, `AlloyResults`,
  `AlloyReceipt`, `IntegrityAttestation`, `Publication`, `AlloyHardware`,
  `AlloyOutputs` and the universal stage discriminator union (which
  loads its branches from the registered domains at validation time).
- Add `domains: list[str] = ["llm-forge"]` field to `ForgeAlloy` root.
- Update Continuum-side TS bindings (ts-rs regen).

The wip/types-additive-checkpoint-bd4349d branch on forge-alloy still
preserves the wrong-layered first attempt per the never-lose-work rule.

### Step 6 — Vision-safety integration (Qwen3VLAdapter)

**Why:** the morning's `vision_safety.py` whitelist module (committed in
`f82773b`) can preserve vision towers bit-exact through a forge run, but it
isn't wired into any adapter. The 8 existing Qwen3.5-derived artifacts shipped
text-only (their cards are honest, but they silently discarded a vision pathway
that could have been preserved for free). Future Qwen3.5-VL re-forges need a
`Qwen3VLAdapter` that consults the whitelist before any prune / train / quant
operation touches a vision-related parameter.

**Plan:**
1. New file `scripts/adapters/qwen3_vl.py` — `Qwen3VLAdapter(QwenDenseBase)`
   (assuming Step 1 has landed) with `architectures = ("qwen3_5_vl",)`.
2. Override the relevant methods to consult `vision_safety.py` at the start
   of each forge step:
   - `prune()`: filter the head importance metric to skip vision-tower heads
   - `train()`: filter LoRA target_modules to exclude vision-tower projections
   - `expert_prune()`: skip vision-shared experts (if the family is MoE-VL)
3. `modality()` becomes a real method that handles `modality` stage params
   (e.g. attaching a SigLIP encoder if missing).
4. New Tier 1 test fixture: a fake VL alloy that exercises the modality stage
   and verifies the dispatcher routes through Qwen3VLAdapter cleanly.
5. Per `VL-FORGE-DESIGN.md`, the actual VL forge methodology lives in Phase 2
   (Tier 2 wiring) and Phase 3 (MoE-VL extension). Step 6 here is the Phase 1
   adapter scaffold.

**Acceptance:** dispatch test passes for a synthetic Qwen3-VL alloy. The 11
breakage points listed in `VL-FORGE-DESIGN.md` Appendix C are addressed by
the adapter's vision_safety calls (verified by existing CPU smoke test
`scripts/test_vision_safety.py`).

### Step 7 — modelHash convention unification

**Why:** see "modelHash conventions" section above. publish_model.py and the
backfill tools use different hash algorithms over the same bytes. A single
verifier should not need to know which convention each alloy uses.

**Plan:**
1. Move `_model_hash_from_shard_hashes` and `_shard_hashes_via_lfs` from
   `backfill_alloy_from_results.py` into a shared module
   `scripts/alloy_hashing.py`.
2. Update `publish_model.hash_model_weights` to use the new convention.
3. Write a one-shot migration `scripts/migrate_modelhash_convention.py` that:
   - Walks each cached alloy
   - For artifacts with `integrity.fileHashes[]` already populated, recomputes
     `modelHash` via the new convention
   - For artifacts WITHOUT `fileHashes[]` (the 3 freshly-forged ones from
     before backfill), pulls the file LFS hashes from HF and populates
     `fileHashes[]` THEN recomputes `modelHash`
   - Diffs old vs new modelHash, prints summary
4. Run the migration in dry-run mode, review, then `--confirm` to push
   updated alloys via `republish_alloy_only.py` (which already supports the
   correction flow from commit `1bc32d2`).
5. Add a Tier 3-style test that asserts every cached alloy's `modelHash` is
   equal to `_model_hash_from_shard_hashes(alloy.integrity.fileHashes)`.
   This test is the gate that prevents future drift back to convention A.

**Acceptance:** every cached alloy's modelHash is verifiable from HF metadata
alone (no downloads), every cached alloy has `fileHashes[]` populated, the
new test gate passes.

### Step 8 — `priorMetricBaselines.samplesHash` schema field + calibration corpus upload

**Why:** the 2 remaining xfails in the reproducibility test:

```
qwen3-coder-30b-a3b-compacted-19b-256k priorMetricBaselines[router-gate-l2-norm]:
    publishes student_samples_router_l2_baseline.jsonl but no samplesHash → unpinned
olmoe-1b-7b-compacted-5b priorMetricBaselines[broad-corpus]:
    publishes student_samples_broad_calibration.jsonl but no samplesHash → unpinned
```

Both falsifiability anchors for §4.1.3.4 are publishable but not
hash-verified. Anyone with HF write access could swap the negative-baseline
JSONL and the test wouldn't catch it.

Also: the §4.1.3.4.1 calibration corpus discipline gate requires the
calibration corpus (`calibration/heldout_code300.jsonl`) to be hash-pinned AND
uploaded. Both flagship MoE alloys reference the corpus by path but the file
doesn't exist in the HF repo. Same fix layer.

**Plan:**
1. Forge-alloy schema: add `priorMetricBaselines[].evaluation.samplesHash` field
   (string, optional, sha256 prefix).
2. Forge-alloy schema: add `calibrationCorpora[].sha256` field (already proposed
   in the FORGE-ALLOY-DOMAIN-EXTENSIBILITY doc; just needs to land).
3. `alloy_to_card.py` / `publish_model.py`: when a `priorMetricBaselines[]`
   entry has a `samplesPath`, compute the sha256 of the file and inject as
   `samplesHash`. Same as the existing `resultHash` injection for forward
   benchmarks (commit `464358e`).
4. Same for calibration corpora — when the alloy declares
   `calibrationCorpora[]`, hash and inject.
5. Upload the corpus files (`calibration/heldout_code300.jsonl` for the qwen3-coder
   artifact, `calibration/heldout_code300.jsonl` + `calibration/heldout_broad300.jsonl`
   for OLMoE) alongside the existing model files.
6. Re-run `republish_alloy_only.py` against both flagship MoE artifacts. The
   updates land alongside the existing JSONLs.
7. Tier 3 reproducibility test: the 2 xfails auto-flip to pass. Add a new
   case for calibration-corpus hash verification.

**Acceptance:** 2 xfails removed, calibration corpora live on HF with hashes
pinned in the alloy, the falsifiability anchors are now byte-verifiable.

---

## Reproducibility test layers (canonical reference)

```
Tier 1 (cheap, runs anywhere)
    test_published_alloys_dispatch.py
        for each alloy in PUBLISHED_ALLOYS:
            chain = resolve_adapter_chain(alloy)
            assert chain is non-empty
            assert every required-override method is overridden on the family adapter
    What it proves: contract resolves cleanly, no per-family branches needed,
    new architectures added via new files.

Tier 2 (medium, requires 5090, NOT YET BUILT)
    test_published_alloys_re_forge.py
        for each MoE alloy:
            execute the chain end-to-end via alloy_executor
            assert produced safetensors hash == alloy.results.integrity.modelHash
    What it proves: the published byte chain is reproducible from the alloy
    alone via the adapter set, not from a one-off CLI invocation.

Tier 3 (cheap, runs anywhere)
    test_published_alloys_sample_hashes.py
        for each (alloy, samples_file) pair where the alloy declares samplesPath + resultHash:
            fetch samples from HF
            assert sha256(bytes) == alloy.results.benchmarks[].resultHash
    What it proves: the producer cannot silently swap the per-problem JSONL
    after publish.

Tier 4 (medium, runs anywhere with evalplus installed)
    test_published_alloys_scoring.py
        for each (alloy, samples_file, expected_score) tuple:
            actual = score_jsonl(samples_file)  # uses evalplus official CLI via subprocess
            assert |actual - expected| <= 0.1pp
    What it proves: the published headline pass@1 reproduces canonically against
    the published JSONL bytes via the canonical scorer. Catches publish-pipeline
    counting-convention bugs (e.g. the qwen3-coder humaneval_plus 86.0 vs 85.4
    bug from commit 1bc32d2).
```

The macOS-evalplus reliability_guard workaround in `_humaneval_scorer.py` is
load-bearing for Tier 4 — without it, evalplus on macOS reports 0.000 for
every JSONL because `resource.setrlimit(RLIMIT_AS, ...)` errors out and the
multiprocessing workers (which use spawn on macOS) don't inherit parent
monkey-patches. The fix is documented in the scorer module docstring; the
short version is "fresh subprocess + fork start_method + reliability_guard
no-op patch in a `python -c` preamble."

---

## Glossary of acronyms / repo paths

- **alloy** — `forge-alloy.json` provenance file. Forge-alloy is the universal
  Merkle-chain-of-custody envelope for any data transformation pipeline (not
  just ML).
- **family adapter** — concrete subclass of `FamilyAdapter` in
  `scripts/adapters/` that handles one model architecture's tensor walks.
- **stage executor** — the existing `scripts/stages/` dispatcher that maps
  alloy stage types (prune, train, expert-prune, quant, eval, ...) to executor
  classes. Now thin: just delegates to the family adapter.
- **§4.1.3.x** — section references to `continuum/docs/papers/PLASTICITY-COMPACTION.md`.
  §4.1.3.1 = per-layer normalized head importance fix. §4.1.3.2 = held-out
  calibration discipline. §4.1.3.3 = compensation LoRA. §4.1.3.4 = calibration-
  aware MoE expert importance metric (the morning's empirical anchor).
  §4.1.3.4.1 = calibration corpus discipline gate.
- **forge-alloy** — `/Volumes/FlashGordon/cambrian/forge-alloy/` (parallel repo).
- **continuum** — `/Volumes/FlashGordon/cambrian/continuum/` (parallel repo).
- **sentinel-ai** — `/Volumes/FlashGordon/cambrian/sentinel-ai/` (this repo).
- **BigMama** — the RTX 5090 forge machine. Tier 2 reproducibility tests run
  here.

---

## Crash-recovery checklist

If a future session picks this up after a drive crash or context loss:

1. Read this file end-to-end. The 8-step roadmap above is the source of truth
   for what's pending.
2. `git -C /Volumes/FlashGordon/cambrian/sentinel-ai branch | grep wip` —
   confirm the wip preservation branches still exist.
3. `git -C /Volumes/FlashGordon/cambrian/sentinel-ai log --oneline cross-arch-portability-fixes -20` —
   confirm the 9 plugin-sprint commits are present.
4. `cd /Volumes/FlashGordon/cambrian/sentinel-ai && .venv/bin/python -m pytest tests/reproducibility/` —
   confirm 40 passed, 0 skipped, 2 xfailed. If anything else, something
   regressed and needs investigation BEFORE moving forward.
5. `~/.claude/projects/.../memory/MEMORY.md` should have entries pointing at
   `feedback_adapters_not_branches`, `feedback_never_lose_work`,
   `project_continuum_ai_hf_reproducibility`, `project_sentinel_ai_is_claudes_code`.
   These are critical context — read them before resuming.
6. `cat /Users/joel/Desktop/convo-with-kash.txt` is Joel pasting Claude's own
   work for visibility, NOT Kash sending Claude work. Sentinel-ai is Claude's
   code, all of it.
7. The next step in the roadmap is **Step 1: extract `QwenDenseBase`**. Start
   there unless the user redirects.
