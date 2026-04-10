# Frontier Deferred Catalog

Models that **clearly want to be forged** (huge MoE, no consumer-tier variants on HF, viral-headline potential) but require **new family adapters** before they can run through the factory pipeline. Listed here so the work isn't lost; each entry has the architecture details + what's needed to add the adapter.

Per Kash's frontier-target review (2026-04-09): these are the "yuge MoE for consumer 5090, viral if it works" candidates beyond the obvious ones (Mixtral 8x22B, DeepSeek-V3, Qwen3-Coder-480B, Qwen3-235B-A22B) that already have working adapters in the family-adapter set.

---

## 1. MiniMax-Text-01 (456B total / 45.9B active)

**Source**: [`MiniMaxAI/MiniMax-Text-01`](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)
**License**: MIT
**Paper**: [arxiv:2501.08313](https://arxiv.org/abs/2501.08313) — Hacker News attention earlier this year

**Architecture from config.json (verified 2026-04-09)**:
```
model_type:              minimax_text_01
architectures:           [MiniMaxText01ForCausalLM]
num_hidden_layers:       80
hidden_size:             6144
intermediate_size:       9216
num_local_experts:       32
num_experts_per_tok:     2
max_position_embeddings: 10,240,000  ← 10M context
```

**Why it's interesting**: hybrid Lightning Attention + MoE — first frontier model to combine softmax attention with linear-attention layers in a hybrid architecture. The 10M context window is also unprecedented.

**Forge target**: 32 → 12 experts (~62.5% prune, 2 active per token leaves 16.7% load — borderline aggressive). With recovery training: ~140B post-prune → ~45 GB Q4 → fits multi-GPU consumer grid (3-4 × 5090) OR 64 GB Mac.

**What's needed**:
- New family adapter: `scripts/adapters/minimax_text_01.py`
- Adapter must handle the **hybrid attention layer** (some layers are softmax, some are lightning-attention) — only the MoE feedforward expert pruning is family-standard. The attention surgery NEEDS to know which layers are which type and skip the lightning layers.
- Tensor name pattern: TBD — need to inspect the safetensors index. Probably `model.layers.{i}.block_sparse_moe.experts.{e}.{w1,w2,w3}` if it's Mixtral-shaped, or `mlp.experts.{e}.{gate,up,down}_proj` if it's Qwen3MoE-shaped, or something custom.
- `model_auto_class()` likely needs `AutoModel` with `trust_remote_code=True` (the model uses `custom_code` per the HF tags).

**Headline if it works**: "First consumer-accessible MiniMax-Text-01 with the lightning-attention hybrid intact." Per Kash: "we'd be the first lab to demonstrate that calibration-aware MoE pruning works on a hybrid attention architecture, which is itself a methodology paper section worth writing."

---

## 2. Hunyuan-Large (Tencent, 389B total / 52B active)

**Source**: [`tencent/Tencent-Hunyuan-Large`](https://huggingface.co/tencent/Tencent-Hunyuan-Large)
**License**: Tencent custom commercial license (gated download — config.json fetch returned empty, requires `huggingface-cli login` with accepted ToS)
**Paper**: [arxiv:2411.02265](https://arxiv.org/abs/2411.02265)

**Architecture (from Kash's review + paper)**:
```
total_params:        389B
active_params:       52B
num_local_experts:   ~16  (paper)
num_experts_per_tok: 1    (paper — single-expert routing)
num_hidden_layers:   ~80  (paper)
```

**Why it's interesting**: first frontier-class Chinese open MoE. Released late 2024 to significant community attention with cards reporting competitive benchmarks against Llama 3.1 405B at ~8× lower active params. **Almost no community quants exist beyond a handful; zero structurally pruned variants on HF.**

**Forge target**: 16 → 10 experts (~37.5% prune, 1 active per token leaves 10% load — same as morning's qwen3-coder-30b flagship ratio). ~150 GB post-prune → ~50 GB Q4 → similar consumer grid territory to Qwen3-Coder-480B (multi-GPU 5090 grid OR 96 GB Mac Studio).

**What's needed**:
- New family adapter: `scripts/adapters/hunyuan_large.py`
- HF auth + ToS acceptance to even fetch the config. Joel needs to manually `huggingface-cli login` and accept the Hunyuan ToS once. Then the daemon can auto-download.
- Tensor name pattern: TBD until we can read the config. Probably mixtral-shaped (`block_sparse_moe`) but unverified.
- Single-expert routing (`num_experts_per_tok: 1`) is unusual — most MoE families use 2-8 active. Verify the prune algorithm doesn't assume top-k > 1.

**Headline if it works**: "First consumer-accessible Hunyuan-Large." Per Kash: "Chinese-frontier-with-strong-benchmarks-but-nobody-can-actually-run-it. The exact 'people clearly want access to it but can't' pattern."

---

## 3. Snowflake Arctic (480B total / 17B active)

**Source**: `Snowflake/snowflake-arctic-instruct` (need to verify URL)
**License**: Apache 2.0

**Architecture (from public summary)**:
```
total_params:        480B
active_params:       17B
num_local_experts:   128
num_experts_per_tok: 2
```

**Why it's interesting**: 480 total experts is the most extreme MoE published. The "first Arctic that fits a single 5090" headline would be enormous.

**Forge target**: 128 → 80 experts (37.5% prune, identical ratio to morning's qwen3-coder-30b). 17B active stays. Post-prune ~300 GB → ~100 GB Q4 → still multi-GPU.

**What's needed**: New family adapter. Architecture verification (haven't fetched config yet).

---

## What "needs new family adapter" means concretely

For each entry above, the work is roughly:

1. **Read the config.json** from HF to verify `model_type`, expert count, expert routing dimensions, hidden/intermediate sizes
2. **Inspect the safetensors index** to find the tensor name patterns for router gates and expert projections
3. **Determine which `LayoutSpec`/`FusedLayoutSpec` matches** (or write a new one if neither does)
4. **Write a new adapter file** in `scripts/adapters/<family>.py`:
   - Subclass `MoEUnfusedExpertsBase` if the layout matches Qwen3MoE
   - Subclass `MixtralAdapter` if the layout matches block_sparse_moe-unfused
   - Subclass `FamilyAdapter` directly if neither (e.g. fused like GraniteMoE, or hybrid like MiniMax)
5. **Override `model_auto_class()`** if the family doesn't load via `AutoModelForCausalLM`
6. **Register the architecture string** in the adapter's `architectures` tuple
7. **Add a TDD test** in `tests/unit/adapters/test_<family>_adapter.py` against a synthetic safetensors fixture
8. **Add the recipe to `seed_factory_queue.py`** with verified geometry

Each new adapter is roughly **half a day to a day of work** depending on whether the layout matches an existing base.

---

## Suggested implementation order (per Kash 2026-04-09)

1. ✅ Mixtral 8x7B (8→6) — cold-tier pressure test (currently in flight)
2. Mixtral 8x22B (8→5 or 8→6 with recovery) — single-5090 headline
3. Qwen3-235B-A22B — already covered by Qwen3MoEAdapter; cold tier required (~720 GB peak)
4. **Hunyuan-Large or MiniMax-Text-01** — pick whichever has cleaner adapter requirements; both require new adapters but Hunyuan looks like a Mixtral-shaped layout (likely faster to add)
5. Qwen3-Coder-480B — moonshot, needs grid

**Headline rotation**: ship one model per week, write a paper section per shipped model, post on HN with the integrity audit story prominently. Each shipped frontier MoE is one HN post, one Twitter thread, one paper section, and one new entry on the §4.1.3.4 cross-family anchor list.

---

## See also

- `docs/PLUGIN-SPRINT.md` — the family-adapter dispatch architecture
- `docs/FACTORY-PROTOCOL.md` — the disk protocol that makes the grid layer possible
- `scripts/adapters/sota_moe.py` — existing SOTA family adapters (Mixtral, Phi-MoE, DeepSeek-V2, GraniteMoE)
- `scripts/seed_factory_queue.py` — current catalog of viable forge targets
