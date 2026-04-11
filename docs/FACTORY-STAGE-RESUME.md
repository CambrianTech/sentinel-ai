# Factory Stage Resume — Don't Throw The Car Away

**Status**: Critical fix needed in sentinel-ai alloy executor.

## The Problem

The alloy executor runs the entire pipeline as one monolithic stage:
`load → profile → prune → quant → eval → publish`

If ANY step fails (OOM at prune, network error at publish, disk full at quant), the executor starts over from scratch. On the 8x22B forge, this means:
- 80 minutes to reload the model
- 63 minutes to re-profile (148K tokens through 56 layers × 8 experts)
- 23 minutes to re-prune (59 shards × 50s each)
- All wasted because the post-prune model reload OOM'd

**This burned 4+ hours on 2026-04-10.** The profiling results and pruned shards were sitting on disk, perfectly intact. The executor threw them away and started over.

## The Fix

Each stage writes a completion checkpoint. On restart, the executor checks for existing checkpoints and skips completed stages.

### Checkpoint Files

```
.factory/work/{alloy-name}/
├── importance.activation_count.json    ← profiling complete
├── pruned/
│   ├── model-00001-of-00028.safetensors
│   ├── ...
│   ├── model-00028-of-00028.safetensors
│   └── expert_prune.metadata.v1.json   ← prune complete (has finished_at)
├── mixtral-8x22b-pruned-Q4_K_M.gguf   ← quant complete
├── eval-results.json                    ← eval complete
└── .stage-checkpoints.json             ← stage completion record
```

### `.stage-checkpoints.json`

```json
{
  "profile": {
    "completed_at": "2026-04-10T18:33:00Z",
    "output": "importance.activation_count.json",
    "hash": "sha256:2d2ae4f5..."
  },
  "prune": {
    "completed_at": "2026-04-11T00:17:02Z",
    "output": "pruned/",
    "shards": 28,
    "hash": "sha256:..."
  },
  "quant": null,
  "eval": null,
  "publish": null
}
```

### Resume Logic

```python
def execute_stage(stage_name, stage_fn, ctx):
    checkpoint = load_checkpoint(ctx.work_dir)
    
    if checkpoint.get(stage_name, {}).get("completed_at"):
        log(f"Stage '{stage_name}' already complete — skipping")
        # Restore output paths from checkpoint
        return restore_from_checkpoint(ctx, stage_name, checkpoint)
    
    # Run the stage
    result = stage_fn(ctx)
    
    # Write checkpoint
    save_checkpoint(ctx.work_dir, stage_name, result)
    
    return result
```

### What Each Stage Needs to Check

| Stage | Check | Skip if |
|---|---|---|
| **Load model** | Can't skip | Always runs (but freed before prune) |
| **Profile** | `importance.activation_count.json` exists | File exists + hash matches source model |
| **Prune** | `pruned/expert_prune.metadata.v1.json` has `finished_at` | Metadata shows complete + all shards exist |
| **Quant** | GGUF file exists with expected size | File exists + size > threshold |
| **Eval** | `eval-results.json` exists | File exists + has PPL value |
| **Publish** | HuggingFace repo exists | API check |

### The Model Load Optimization

Even with stage resume, the model load (80 min for 8x22B) happens every time. But if profiling is skipped (importance JSON exists), the model only needs to load for eval — and at that point we should use the llama.cpp eval path (safetensors → GGUF → llama-perplexity) which bypasses the BnB/transformers load entirely.

```
If profiling complete AND prune complete:
  → Skip transformers model load entirely
  → Convert pruned safetensors → GGUF (20 min)
  → Quantize GGUF (5 min)
  → llama-perplexity eval (10 min)
  → Total: 35 min instead of 80 min + 63 min + 23 min = 166 min
```

## Impact

| Scenario | Without resume | With resume |
|---|---|---|
| OOM during prune (8x22B) | 143 min wasted, start over | 0 min wasted, resume from pruned shards |
| Network error at publish | 200+ min wasted | 0 min, just retry publish |
| Disk full during quant | 170 min wasted | 0 min, free space and resume |
| Power loss at eval | Everything lost | Resume from GGUF |

## Implementation Priority

1. **Profiling cache** (DONE on BigMama — checks for existing importance JSON)
2. **Prune completion check** — verify metadata.finished_at + all shards exist
3. **GGUF-first eval path** — when profiling + prune are done, skip transformers entirely
4. **Stage checkpoint file** — formal .stage-checkpoints.json
5. **Executor resume logic** — read checkpoints at startup, skip completed stages
