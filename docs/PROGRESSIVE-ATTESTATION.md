# Progressive Attestation — Chain of Custody During Forge

**Status**: Design. One session of executor plumbing to implement.

## The Principle

The alloy is not written at the end. It's built up stage by stage. Each stage adds its attestation link. Crashes don't break the chain — they pause it. Resume continues from the last valid link.

Git IS the ledger. Each stage commit references artifacts by hash. No blockchain needed — git is already a Merkle tree.

---

## The Chain

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: RECIPE                                         │
│   input:  alloy.json (the recipe)                       │
│   output: recipe committed to git                       │
│   attest: commit_hash + recipe_hash                     │
│   ← chain starts here                                   │
├─────────────────────────────────────────────────────────┤
│ Stage 2: LOAD + PROFILE                                 │
│   input:  source_model (sha256 of safetensors)          │
│           calibration_corpus (sha256 of corpus file)    │
│   output: importance.activation_count.json              │
│   attest: source_hash + corpus_hash + importance_hash   │
│   checkpoint: importance.json exists + hash matches     │
│   ← attestable now, before prune                        │
├─────────────────────────────────────────────────────────┤
│ Stage 3: PRUNE                                          │
│   input:  importance_hash (chains to profile)           │
│           source_shards (sha256 per shard)              │
│   output: pruned shards + metadata.json                 │
│   attest: importance_hash + source_hashes + output_hash │
│   checkpoint: metadata.finished_at + all shards exist   │
│   ← attestable now, before quant                        │
├─────────────────────────────────────────────────────────┤
│ Stage 4: QUANTIZE                                       │
│   input:  pruned_shards_hash (chains to prune)          │
│   output: model-Q4_K_M.gguf                             │
│   attest: pruned_hash + gguf_hash + quant_config        │
│   checkpoint: gguf file exists + size matches            │
│   ← attestable now, before eval                         │
├─────────────────────────────────────────────────────────┤
│ Stage 5: EVAL                                           │
│   input:  gguf_hash (chains to quant)                   │
│           eval_dataset (sha256 of wikitext)              │
│   output: eval-results.json (PPL, benchmarks)           │
│   attest: gguf_hash + dataset_hash + results_hash       │
│   checkpoint: results.json exists + has PPL value        │
│   ← attestable now, before publish                      │
├─────────────────────────────────────────────────────────┤
│ Stage 6: PUBLISH                                        │
│   input:  all previous hashes                           │
│   output: HuggingFace repo URL + model card             │
│   attest: complete alloy with full hash chain            │
│   checkpoint: HF repo exists + alloy uploaded            │
│   ← final attestation, verify URL live                  │
└─────────────────────────────────────────────────────────┘
```

## The Alloy File — Progressive Construction

The alloy starts as a recipe (stages defined, results empty). After each stage, results are filled in:

```json
{
  "name": "mixtral-8x22b-compacted-70b",
  "stages": [...],
  "results": {
    "stageCheckpoints": {
      "profile": {
        "completedAt": "2026-04-10T18:33:00Z",
        "outputHash": "sha256:2d2ae4f5...",
        "inputHashes": {
          "sourceModel": "sha256:abc...",
          "calibrationCorpus": "sha256:def..."
        }
      },
      "prune": {
        "completedAt": "2026-04-11T00:17:02Z",
        "outputHash": "sha256:ghi...",
        "inputHashes": {
          "importanceJson": "sha256:2d2ae4f5..."
        },
        "metadata": {
          "shardsWritten": 28,
          "expertsDropped": 672,
          "layersPruned": 56
        }
      },
      "quant": null,
      "eval": null,
      "publish": null
    }
  }
}
```

After profile: alloy has 1 checkpoint. After prune: 2 checkpoints. After crash and resume: chain continues from the last checkpoint. The final alloy has all 5 checkpoints.

## Implementation

### 1. Checkpoint Writer (new module)

```python
# sentinel-ai/scripts/stages/checkpoint.py

class StageCheckpoint:
    def __init__(self, work_dir: Path):
        self.path = work_dir / ".stage-checkpoints.json"
        self.data = self._load()

    def is_complete(self, stage: str) -> bool:
        cp = self.data.get(stage)
        return cp is not None and cp.get("completedAt") is not None

    def verify(self, stage: str) -> bool:
        """Verify checkpoint hash matches artifact on disk."""
        cp = self.data.get(stage)
        if not cp:
            return False
        output_path = self.work_dir / cp["outputPath"]
        return sha256_file(output_path) == cp["outputHash"]

    def complete(self, stage: str, output_path: str, input_hashes: dict, metadata: dict = None):
        """Mark stage complete with attestation data. Atomic write."""
        self.data[stage] = {
            "completedAt": datetime.utcnow().isoformat() + "Z",
            "outputPath": output_path,
            "outputHash": sha256_file(self.work_dir / output_path),
            "inputHashes": input_hashes,
            "metadata": metadata or {},
        }
        self._save_atomic()

    def _save_atomic(self):
        """Write to temp file, then rename. Can't get half-written checkpoint."""
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.rename(self.path)
```

### 2. Executor Changes (alloy_executor.py)

```python
# Before each stage:
checkpoint = StageCheckpoint(work_dir)

if checkpoint.is_complete("profile") and checkpoint.verify("profile"):
    log("Profile already complete — skipping (saved 63 min)")
    ctx.importance_json_path = checkpoint.output_path("profile")
else:
    ctx = run_profile(ctx)
    checkpoint.complete("profile",
        output_path="importance.activation_count.json",
        input_hashes={
            "sourceModel": sha256_file(source_model_path),
            "calibrationCorpus": sha256_file(corpus_path),
        })

if checkpoint.is_complete("prune") and checkpoint.verify("prune"):
    log("Prune already complete — skipping (saved 23 min)")
    ctx.pruned_dir = checkpoint.output_path("prune")
else:
    ctx = run_prune(ctx)
    checkpoint.complete("prune",
        output_path="pruned/",
        input_hashes={
            "importanceJson": checkpoint.data["profile"]["outputHash"],
        },
        metadata={"shardsWritten": 28, "expertsDropped": 672})

# ... same pattern for quant, eval, publish
```

### 3. Alloy Progressive Write

After each stage checkpoint, update the alloy file:

```python
def update_alloy_with_checkpoint(alloy_path, stage_name, checkpoint_data):
    alloy = json.load(open(alloy_path))
    if "stageCheckpoints" not in alloy.get("results", {}):
        alloy.setdefault("results", {})["stageCheckpoints"] = {}
    alloy["results"]["stageCheckpoints"][stage_name] = checkpoint_data
    # Atomic write
    tmp = alloy_path + '.tmp'
    json.dump(alloy, open(tmp, 'w'), indent=2)
    os.rename(tmp, alloy_path)
```

### 4. Git Commits (optional, per-stage)

```python
def commit_stage(repo_dir, stage_name, alloy_path):
    """Commit the updated alloy after each stage. Optional but enables
    full git history = full attestation ledger."""
    subprocess.run([
        "git", "add", alloy_path,
        "git", "commit", "-m",
        f"forge: {stage_name} complete — {alloy_name}"
    ], cwd=repo_dir)
```

## What This Costs

- ~50 lines of checkpoint.py
- ~20 lines per stage in alloy_executor.py (the is_complete + verify + complete calls)
- Zero performance impact (sha256 hashing is <1s even for 5GB shards)
- Zero architectural change (same stages, same alloy schema, just write timing changes)

## What This Saves

The 8x22B forge on 2026-04-10:
- **Without checkpoints**: 4+ hours burned on 3 retries. OOM at prune (kink #14), meta tensor crash (kink #9), second meta tensor crash (daemon restart). Each retry reloaded 80 min + re-profiled 63 min.
- **With checkpoints**: First retry skips profile (63 min saved). Second retry skips profile + prune (86 min saved). OOM fix + resume from pruned shards → straight to GGUF quant + eval.

**Total time saved on this ONE forge: ~4 hours.**

On a factory running 24/7, this is the difference between forging 3 models per day and forging 1.
