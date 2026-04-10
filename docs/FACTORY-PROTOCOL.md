# Factory Protocol v0

**Status**: v0 — single-node and small-grid (≤10 nodes). Stable enough that any consumer can implement against it without importing sentinel-ai code.

**Why this document exists**: per Kash 2026-04-09 — "the disk protocol IS the API contract." Once `intake/`, `assembly/`, `finished/`, `rework/`, `.heartbeat.json`, `throughput.jsonl`, and `factory_node.toml` stabilize as a spec, sentinel-ai doesn't need to know continuum exists, and continuum doesn't need to know sentinel-ai's internals. They communicate via filesystem state — the same primitive Maildir uses for email and systemd uses for service state. **The right primitive for distributed AI work because the failure mode (node crash) leaves recoverable state on disk that another worker can pick up.**

This spec is what makes the eventual Rust port of the orchestrator possible without running Python.

---

## Directory layout

A "hive node" is one machine running the factory protocol. Each node owns a single queue root, conventionally:

```
~/sentinel-factory/.factory/
├── factory_node.toml          ← node config (this node's identity + storage tiers)
├── line/
│   ├── intake/                ← parts entering the factory
│   ├── assembly/              ← parts currently being built
│   ├── finished/              ← parts completed and awaiting release
│   ├── rework/                ← parts that failed QA, need triage
│   ├── deferred/              ← (optional) parts the foreman parked
│   ├── .heartbeat.json        ← daemon liveness state
│   ├── .worker.pid            ← single-worker lock
│   └── throughput.jsonl       ← append-only audit log
├── work/                      ← per-part forge intermediate state
│   └── <part_stem>/           ← work dir for a single in-flight or finished part
└── calibration/               ← held-out corpora referenced by alloys
    └── *.jsonl
```

The four "stations" in `line/` (`intake → assembly → finished | rework`) are the part lifecycle's state machine. Files move between them via **atomic `rename()`** which is the entire concurrency primitive — multi-worker safety comes free if you ever need it because two workers attempting the same `rename()` will see exactly one succeed.

---

## File schemas

### Part files (the alloys)

A part is a single file at `line/<station>/<filename>.alloy.json`. The filename is the part's identity within the queue. The file content is a JSON document conforming to the **forge-alloy** schema (separate spec, see `forge-alloy/python/forge_alloy/types.py`).

Filename conventions:
- `<basename>.alloy.json` — original recipe, no retries
- `<basename>.retry<N>.alloy.json` — recovered from a previous crash N times
- A `<basename>` MUST be unique per queue. Duplicates are a foreman error.

The retry counter is encoded in the filename so it survives across station moves without a separate state file.

### `.heartbeat.json` — daemon liveness

Atomic-write JSON file. Updated by the worker on every poll iteration (typically every 30 seconds). Consumers read it to know:
- Is this node alive?
- What is it building right now?

```json
{
  "pid": 12345,
  "state": "idle | building | offline",
  "current_part": "mixtral-8x7b-instruct-compacted-conservative.alloy.json",
  "last_beat_at": "2026-04-09T20:48:58Z",
  "host": "BigMama"
}
```

**Stale-detection**: a consumer reading `.heartbeat.json` MUST verify `pid` is still alive on the target host. A stale heartbeat (process gone but file remains) means the daemon crashed; the next worker startup runs the recovery pass and the file is overwritten.

### `.worker.pid` — single-worker lock

Plain text file containing the running daemon's PID. Used for soft-lock enforcement: only one daemon may own a queue root at a time. Stale PID files (process not alive) are cleaned by the next worker startup.

A consumer that wants to write to `intake/` does NOT need to acquire the worker lock — `intake/` is multi-writer safe because writes are atomic file creations.

### `throughput.jsonl` — append-only audit log

One JSON object per line, written by the worker on every state transition. **Append-only, never rewritten, never compacted in-place.** Consumers can stream-read the tail and the producer-consumer semantics are simple.

```jsonl
{"at": "2026-04-09T20:31:58Z", "outcome": "finished", "alloy": "qwen2-5-7b-instruct-compacted.retry1.alloy.json", "forged_dir": "/path/to/work/dir"}
{"at": "2026-04-09T20:35:12Z", "outcome": "rework", "alloy": "olmoe-1b-7b-0924-instruct-compacted.alloy.json", "error": "Layer 6 silent-regression invariant: cycle 1 PPL 11750 > previous 7734"}
{"at": "2026-04-09T20:36:01Z", "outcome": "evicted", "path": "/path/to/orphan/work/dir", "bytes": 11000000000, "action": "deleted"}
{"at": "2026-04-09T20:48:58Z", "outcome": "promoted_from_rework", "alloy": "mixtral-8x7b-instruct-compacted-conservative.alloy.json", "from": "mixtral-8x7b-instruct-compacted-conservative.retry1.alloy.json"}
```

Standard outcomes:
- `finished` — part completed, artifact in `finished/`
- `rework` — part failed, artifact in `rework/` with `.error.json` sidecar
- `recovered` — part was in `assembly/` on startup, moved back to `intake/` with retry counter incremented
- `evicted` — auto-cleanup pass removed an orphan work dir
- `promoted_from_rework` — foreman manually moved a rework part back to intake

### `.events.jsonl` — the forge event stream (v0.2)

An append-only JSON Lines file written by the daemon at every forge lifecycle transition. Each line is one event with a timestamp, the node hostname, an event `kind`, and a kind-specific payload. **This is the file-based v0 transport for forge events; continuum's `Events.emit()` pub/sub layer will eventually subscribe to this file and republish events as native pub/sub events.** Until then, operators and agents tail the file directly (`tail -F .events.jsonl`) or read it in batches via `FactoryQueue.read_events()`.

The events stream is **observability, not load-bearing state**. The canonical state of the line is still held in `.heartbeat.json` (current daemon liveness + current part) and the station directories (`intake/` / `assembly/` / `finished/` / `rework/` — where each alloy file physically sits). Events are the *history* of how state changed, not the state itself. This means event emission is best-effort and never blocks a forge; a lost event is a gap in observability but the state remains authoritative via the canonical sources.

**Event kinds (v0.2)**, with required payload fields beyond the base (`timestamp`, `host`, `kind`, `alloy`):

| Kind | When emitted | Required payload |
|---|---|---|
| `forge/started` | New part picked up from intake | `forged_dir`, optionally `alloy_name`, `source_model`, `stages[]` |
| `forge/stage/started` | Stage begins | `stage` (string name), optionally `stages[]` (full pipeline) |
| `forge/stage/progress` | Periodic progress within a stage (optional, fine-grained) | `stage`, `substage?`, `progress` (0.0-1.0), `samples_done?`, `samples_total?` |
| `forge/stage/completed` | Stage finishes cleanly | `stage`, `elapsed_s`, optionally `forged_dir` |
| `forge/model/load/*` | Optional fine-grained model-load lifecycle | `source_gb`, `streaming`, `max_gpu_gb`, `max_cpu_gb`, `peak_*_gb`, `elapsed_s` |
| `forge/completed` | Full forge lands in `finished/` | `forged_dir`, `modelHash`, `published`, `elapsed_s`, `priorMetricBaselines_count`, `hf_repo_url?` |
| `forge/rework` | Forge moves to `rework/` with error | `stage`, `error` (truncated to 500 chars), `elapsed_s` |

**Example event stream** (one forge from start to finish):

```jsonl
{"timestamp": "2026-04-10T01:09:00.000Z", "host": "BigMama", "kind": "forge/started", "alloy": "mixtral-8x7b-instruct-compacted.retry2.alloy.json", "forged_dir": "/home/joel/sentinel-factory/work/mixtral-8x7b...", "alloy_name": "mixtral-8x7b-instruct-compacted-conservative", "source_model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "stages": ["expert-activation-profile", "expert-prune", "quant", "eval", "publish"]}
{"timestamp": "2026-04-10T01:09:00.500Z", "host": "BigMama", "kind": "forge/stage/started", "alloy": "mixtral-8x7b-instruct-compacted.retry2.alloy.json", "stage": "executor", "stages": ["expert-activation-profile", "expert-prune", "quant", "eval", "publish"]}
{"timestamp": "2026-04-10T02:45:15.000Z", "host": "BigMama", "kind": "forge/stage/completed", "alloy": "mixtral-8x7b-instruct-compacted.retry2.alloy.json", "stage": "executor", "elapsed_s": 5774.5, "forged_dir": "/home/joel/sentinel-factory/work/mixtral-8x7b..."}
{"timestamp": "2026-04-10T02:45:16.000Z", "host": "BigMama", "kind": "forge/completed", "alloy": "mixtral-8x7b-instruct-compacted.retry2.alloy.json", "forged_dir": "/home/joel/sentinel-factory/work/mixtral-8x7b...", "modelHash": "sha256:...", "published": false, "elapsed_s": 5776.0, "priorMetricBaselines_count": 1}
```

**Schema compatibility**: consumers MUST tolerate unknown payload fields (forward compatibility). Consumers MUST NOT assume fields beyond the required ones for each kind. Adding new kinds or new optional fields is a minor version bump; removing fields or changing their semantics is a breaking change requiring a major version bump.

**Rotation**: the events file can be rotated by a log-rotator (for long-running nodes producing many events) but rotation must preserve the file's contents in order and never reorder events. A rotated file sequence (`.events.jsonl.1`, `.events.jsonl.2`, etc.) is a valid continuation of the stream.

**Subscriber patterns**:

1. **Tail and parse** — `tail -F .events.jsonl | jq` for a live feed on the operator's terminal
2. **Batch read with `since`** — `FactoryQueue.read_events(since_timestamp=...)` returns events newer than a given timestamp, for poller clients that catch up intermittently
3. **Republish to continuum `Events.emit()`** — a bridge process that tails the file and republishes each event to continuum's native pub/sub system as `data:forge:<kind>` events. This is the path that connects file-based v0 events to continuum's native event infrastructure without changing the daemon's side of the contract.

### Sidecar files

Each part in `finished/` and `rework/` has at least one sidecar with the same basename:

- `finished/<name>.alloy.json` + `finished/<name>.result.json`
- `rework/<name>.alloy.json` + `rework/<name>.error.json`

**Sidecar glob contract**: any file in `finished/` or `rework/` matching `<name>.*` is a sidecar belonging to that part. Consumers reading a station MUST treat all `<name>.*` files as one logical bundle. This includes evaluation sample files like `<name>.router-gate-l2-baseline.eval-samples.jsonl.gz` — those live in `finished/` next to the alloy, NOT in `work/`. The `work/` directory holds only intermediate state that gets evicted after the result is stable; anything that needs to outlive the work dir gets promoted to a sidecar.

#### `<name>.result.json`

```json
{
  "completed_at": "2026-04-09T20:31:58Z",
  "forged_dir": "/path/to/work/dir/_seed_qwen2-5-7b-instruct-compacted",
  "alloy_path": "/path/to/.factory/line/assembly/_seed_qwen2-5-7b-instruct-compacted.alloy.json",
  "published": false,
  "modelHash": "sha256:abc...",
  "fileHashes": [
    {"filename": "model-00001-of-00002.safetensors", "sha256": "..."},
    {"filename": "model-00002-of-00002.safetensors", "sha256": "..."}
  ],
  "hf_repo_url": "https://huggingface.co/continuum-ai/qwen2-5-7b-instruct-compacted",
  "alloyChainHash": "sha256:...",
  "signatureBundle": {"signer": "...", "signature": "...", "chain": ["sha256:..."]},
  "priorMetricBaselines": [
    {
      "name": "router-gate-l2-baseline",
      "metric": "humaneval-pass1",
      "value": 78.7,
      "samplesPath": "router-gate-l2-baseline.eval-samples.jsonl.gz",
      "note": "§4.1.3.4 negative-baseline: gate-magnitude pruning without activation profile"
    }
  ]
}
```

Required fields: `completed_at`, `forged_dir`. Optional: `modelHash`, `fileHashes`, `hf_repo_url`, `alloyChainHash`, `signatureBundle`, `priorMetricBaselines`, anything the publisher injected.

**Provenance fields** (`alloyChainHash`, `signatureBundle`) bridge "the forge ran" to "the published artifact is cryptographically chained back to its source." The `ship`-role node populates these when it uploads.

**`priorMetricBaselines`** captures the §4.1.3.4 negative-baseline runs as structured data so the protocol can index falsifiable evidence — not just opaque sample files in `work/`. Each entry's `samplesPath` is relative to the `finished/` station and resolves to a sidecar (see below).

#### `<name>.error.json`

```json
{
  "failed_at": "2026-04-09T17:35:19Z",
  "error": "Layer 6 silent-regression invariant: cycle 1 PPL 11750 > previous 7734",
  "traceback": "Traceback (most recent call last):\n  File ...",
  "retries": 1
}
```

### `factory_node.toml` — declarative node config

See the dedicated section in the schema below. The grid (continuum) reads this file across all nodes via the heartbeat protocol or via a remote `read` to make routing decisions.

---

## State machine (per part)

```
                  ┌──────────────────────────────────────┐
                  │                                      │
                  ▼                                      │
   intake/  ─→  assembly/  ─→  finished/                 │
                  │              (terminal until         │
                  │              continuum's shipping    │
                  │              flow advances them)     │
                  │                                      │
                  └──→ rework/  ─→  (foreman triage)    │
                       │             ─→  intake/ (retry)─┘
                       └──→  (deferred or scrapped)
```

**Transitions** (all via atomic `rename()`):

1. `intake → assembly`: worker pops oldest pending part. Atomic. Multi-worker safe.
2. `assembly → finished`: forge succeeded. Worker writes `.result.json` sidecar, then renames the alloy file.
3. `assembly → rework`: forge failed. Worker writes `.error.json` sidecar, then renames the alloy file.
4. `assembly → intake (recovered)`: on daemon startup, any part still in `assembly/` from a previous crash is moved back to `intake/` with the retry counter incremented. After `MAX_RETRIES=3` it goes to `rework/` instead with a "crash recovery exhausted" error sidecar.
5. `rework → intake (promoted)`: foreman manual action. Resets retry counter.

**Invariants**:
- A part filename is unique within `line/`. It exists in exactly one station at a time.
- `assembly/` may contain at most one part per worker (single-worker lock enforces this).
- A part in `finished/` is never moved back to earlier stations. Either it's released (e.g. published to HF) and the work dir gets evicted, or it stays as a historical artifact.

---

## `factory_node.toml` schema

Per-node declarative config. Lives at `<queue_root>/factory_node.toml`. Single source of truth for what tier of storage this node has.

```toml
[node]
name        = "bigmama"             # display name in heartbeats / grid view
hostname    = "BigMama"             # actual host (for grid coordination)
roles       = ["forge"]             # forge | ship | both (see "Node roles" below)
gpu_count   = 1
gpu_vram_gb = 32                    # 5090=32, 4090=24, 3090=24, 4060=8

[storage]
# Hot tier: fast SSD where the .factory/ queue dir lives.
hot = { path = "/home/joel/sentinel-factory/.factory", min_free_gb = 30 }

# Cold tiers in PRIORITY ORDER. Evictions fill tier 1 first, spill to
# tier 2 when tier 1 hits min_free, etc.
[[storage.cold]]
name             = "melmac"
path             = "/mnt/d/cold"
fs_type          = "drvfs"          # ext4 | drvfs | nfs | s3
write_mb_per_sec = 210              # informs the grid scheduler's wall-clock estimates
read_mb_per_sec  = 210
purpose          = ["work-archive", "published-backup"]

# [[storage.cold]]
# name = "secondary-archive"
# path = "/mnt/e/cold"

# Network tiers — remote sources the scheduler can stream from.
# At multi-Gbit symmetric residential (5-8 Gbit Google Fiber tier),
# HF itself becomes a viable storage tier: Mixtral 8x22B (~260 GB)
# pulls in ~5-7 min at 8 Gbit vs ~35 min at gigabit. Treat the local
# cold tier as forge-output archive, not as a mandatory source cache.
#
# Peer tiers describe other hive nodes on the mesh. The grid
# coordinator gossips part hashes; if a peer already has the source
# weights for a forge target, the scheduler routes the pull to the
# peer (LAN-speed) instead of re-fetching from HF (WAN-speed).
[[storage.network]]
name             = "huggingface"
kind             = "hf"
read_mb_per_sec  = 1000              # capped by HF egress, not your pipe
purpose          = ["source-weights"]

# [[storage.network]]
# name            = "peer-melmac-2"
# kind            = "peer"
# endpoint        = "tailscale://nerd-friend-node:7100"
# read_mb_per_sec = 950               # 8 Gbit LAN sustained
# purpose         = ["source-weights", "forged-artifact-mirror"]

[forge]
# Per-node retry policy. Contract constant — consumers MUST honor this
# when reading recovered parts. Different node types want different
# policies (a fast forge node retries more aggressively than a slow one).
max_retries = 3

[grid]
coordinator = "tailscale://continuum-coordinator:7100"
heartbeat_interval_seconds = 30
```

### Storage tiers (hot / cold / network)

The cache hierarchy is **L0 GPU VRAM → L1 system RAM → L2 hot SSD → L3 cold HDD → L4 network (peer LAN or HF WAN)**. `factory_node.toml` declares L2/L3/L4 explicitly so the grid scheduler can compute wall-clock estimates.

- **`[storage.hot]`** — single fast SSD where `.factory/` lives. Holds the queue, the in-flight work dir, and recently-finished artifacts. `min_free_gb` is a soft floor; auto-cleanup tries to keep this much free by spilling older artifacts to cold tiers.
- **`[[storage.cold]]`** — local mounted drives in priority order. Forged-artifact archive + (today) source-weight cache for big-MoE pulls. At gigabit-residential, the cold tier doubles as a source cache because re-pulling 260 GB from HF is slow. At multi-Gbit, the cold tier becomes pure archive.
- **`[[storage.network]]`** — remote sources the scheduler can stream from. Two `kind`s:
  - **`hf`** — HuggingFace Hub. Always available, no auth beyond `HF_TOKEN`. Read speed capped by HF egress.
  - **`peer`** — another hive node on the mesh (Tailscale endpoint). The grid coordinator gossips part hashes; if a peer already has the source weights, the scheduler routes the pull to the peer at LAN speed instead of re-fetching from HF at WAN speed.

**Multi-Gbit unlock**: at 5–8 Gbit symmetric, network tiers become first-class — the bottleneck moves from I/O back to GPU compute, which is the right place for it. The "node affinity" optimization (forge + ship on the same box to avoid 300 GB transfers) becomes irrelevant; nodes split freely.

### Node roles

The `roles` field declares what workload types this node accepts:

- **`forge`** — runs the prune/quant/eval pipeline. Needs GPU + hot/cold storage. Consumes parts via the `intake → assembly → finished | rework` lifecycle.
- **`ship`** — runs the publish step (HF upload, GGUF quantization, model card generation, signature bundle assembly). Does NOT need a GPU; needs network bandwidth + HF auth. Consumes parts that already have a `result.json` and emits the `hf_repo_url` + `signatureBundle` fields back into the sidecar.
- **`both`** — single-node deployments where the same machine forges and ships. Default for the BigMama-style setup.

The grid coordinator routes parts to nodes based on role: a `forge`-only node receives intake parts, hands them off to a `ship`-only node when forging completes (via grid event or by transferring the alloy + work dir to the ship node's `intake/`).

Per Kash's note: **read/write speed metadata per tier matters** for grid scheduling. Without it, the scheduler is flying blind on time-to-completion estimates. Always populate `write_mb_per_sec` and `read_mb_per_sec` if known.

---

## Consumer contract

**Any consumer** (continuum's grid layer, a CI runner, a status dashboard, a third-party integration) reading these directories MUST be able to operate **without importing any sentinel-ai code**. The protocol is the API; the Python implementation is one consumer of it.

**Required reads**:
- Walk `line/<station>/*.alloy.json` for parts in each station
- Read `.heartbeat.json` for daemon state
- Read `factory_node.toml` for node capabilities
- Tail `throughput.jsonl` for state transition history
- Read `<finished_or_rework>/<name>.{result,error}.json` for sidecar metadata

**Required writes** (for orchestrators that want to inject parts):
- Atomic write of new alloy files to `intake/` (write to `intake/.<name>.tmp` then rename to `intake/<name>.alloy.json`)
- DO NOT write directly to `assembly/`, `finished/`, or `rework/` — only the worker daemon transitions parts between stations.

**Required reads for the foreman / grid coordinator** to make scheduling decisions:
- `factory_node.toml` (storage tiers, GPU capacity)
- `.heartbeat.json` (current state, current part)
- `pressure(<root>)` style disk usage (free GB on hot, free GB on each cold tier)

---

## Extensibility — beyond forge

Per Kash's third addition: **the disk protocol expands beyond forge workloads naturally**. Today's alloy files are forge-shaped (`source.architecture → family adapter → forge stages → publish`). The v0.2 of the protocol should generalize so other workload types can use the same scaffold:

```
line/intake/
├── mixtral-8x7b.forge.alloy.json     ← forge workload (current)
├── helper-eval-2026-04.eval.alloy.json    ← (future) eval workload
├── qwen3-235b.train.alloy.json       ← (future) training workload
├── persona-helper.inference.alloy.json    ← (future) inference workload
```

Each consumer adapter handles its own workload type via the same `intake → assembly → finished | rework` lifecycle. **The grid layer doesn't care what's inside the alloy — it only cares about which node has the right capabilities to run it.**

Concretely, the v0.2 schema additions:
- Add a top-level `workloadType: "forge" | "eval" | "train" | "inference"` field on every alloy.
- Family adapters dispatch on `workloadType + source.architecture`.
- Default `workloadType: "forge"` for backwards compatibility with existing alloys.

This is **design only** for v0 — implementation lands when there's a second workload type to ship. But the spec should ALLOW it so we don't have to renegotiate the protocol when continuum starts running its own non-forge workloads on the same hive nodes.

---

## Risk register

### v0 disk-backed queue scaling ceiling

**Risk**: filesystem-based work queues hit walls at high node counts and high alloy throughput. Per Kash's note: inode pressure, fsync latency, lack of priority semantics, no atomic move across mount points. For a single-node or ≤10-node grid the disk protocol is fine indefinitely. For 100+ nodes handling thousands of alloys/day, the v1 architecture is PostgreSQL `LISTEN/NOTIFY`, NATS JetStream, or similar.

**Mitigation**: by the time the grid scales past what filesystems handle, continuum's grid layer will exist and the Python daemon retires. The disk protocol is a **v0 primitive, not the v1 architecture** — and that's the design intent.

**Contingency** (if continuum's grid layer slips): the Python daemon becomes load-bearing for production and needs hardening:
- Concurrent worker locking that survives stale-PID edge cases
- Partial-write recovery for `throughput.jsonl` and sidecar files
- Retry policy with exponential backoff
- Dead-letter handling for permanently broken parts
- Possibly: replace filesystem station moves with a SQLite WAL queue at `<root>/queue.db` while keeping the same logical state machine

The 12 tests on `factory_queue.py` cover the happy path; production failure happens in the edge cases. **Worth a follow-up audit if continuum grid layer doesn't ship by 2026-Q3.**

---

## Versioning

This spec is **v0**. Breaking changes to the directory layout or file schemas bump the major version. Additive changes (new optional fields, new file types) bump the minor version.

When a breaking change is needed, the spec adds a `protocol_version` field to `factory_node.toml` so consumers can detect and refuse incompatible nodes:

```toml
[protocol]
version = "0.1"
```

The current absence of `[protocol]` implies version `0.0` (this document).

---

## See also

- `scripts/factory_queue.py` — the Python reference implementation
- `scripts/factory_storage.py` — storage tier management + `factory_node.toml` loader
- `scripts/bootstrap-hive-node.sh` — fresh-node bootstrap (writes `factory_node.toml.example`)
- `docs/HIVE-NODE-OPERATOR.md` — operator playbook (manual run, foreman commands)
- `docs/PLUGIN-SPRINT.md` — the family-adapter dispatch architecture (axis 1: `source.architecture → FamilyAdapter`)
- `forge-alloy/python/forge_alloy/types.py` — the alloy schema this protocol carries
