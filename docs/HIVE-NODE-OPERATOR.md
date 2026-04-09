# Hive Node Operator's Playbook

This is the manual playbook for running ONE forge node (BigMama, or any
other consumer-GPU box) **without continuum**. Every command in here is
runnable today; you (the human foreman) operate the node directly via
`python -m factory_queue` and your own HF token.

When continuum's grid layer ships, the same node will be commandable
remotely from the grid. Until then, this playbook is the manual escape
hatch — and it's the same set of operations the eventual grid layer will
trigger, just typed by hand instead of dispatched by an agent.

## Prerequisites (BigMama, one-time)

```bash
# Repo + virtualenv
cd ~ && git clone https://github.com/CambrianTech/sentinel-ai.git
cd sentinel-ai && ./setup.sh && source .venv/bin/activate

# HF auth (only needed if you want --publish to push to your org)
export HF_TOKEN=hf_xxxxx     # add to ~/.bashrc to persist

# Calibration corpus — the §4.1.3.4 calibration-aware metric needs a
# held-out corpus on disk. Reuse the one from this morning's flagship:
mkdir -p ~/sentinel-ai/.factory/calibration
# Copy the existing heldout_code300.jsonl from the morning forge output
# into ~/sentinel-ai/.factory/calibration/heldout_code300.jsonl
# (or generate a fresh one from a different code corpus and update the
# alloy's calibrationCorpusFile field)
```

## Operating commands

```bash
# What's the line doing right now?
python -m factory_queue --root .factory --status

# Same thing, formatted as a one-screen dashboard (recommended)
python -m factory_queue --root .factory --status --pretty

# Pretty-print the intake station (replaces `ls .factory/line/intake/`)
python -m factory_queue --root .factory --list

# Pretty-print any station
python -m factory_queue --root .factory --list-station finished
python -m factory_queue --root .factory --list-station rework

# Show the last 20 events from the throughput log
python -m factory_queue --root .factory --tail

# Recover any stuck parts after a hard crash (before relaunching)
python -m factory_queue --root .factory --recover

# Promote a rework alloy back to intake (resets retry counter)
python -m factory_queue --root .factory --retry deepseek-v2-lite-chat-compacted.alloy.json

# Drop an alloy file into intake (atomic copy)
python -m factory_queue --root .factory --enqueue path/to/my-recipe.alloy.json
```

## Three modes of running the daemon

### Mode 1 — Forge only, no publish (safest first run)

The worker forges + assays each part and writes results to
`.factory/line/finished/`. NOTHING gets pushed to HuggingFace. You
manually inspect the finished artifacts and pick what to ship.

```bash
python -m factory_queue --root .factory
```

The daemon runs forever. SIGTERM (`Ctrl-C` from the terminal, or
`kill <pid>` from elsewhere) exits cleanly. The next startup recovers
any in-flight part automatically — no work lost.

### Mode 2 — Forge + auto-publish (you trust the gate)

Each finished part is immediately published to `huggingface.co/<org>/`
using your `HF_TOKEN`. The acceptance criteria in the alloy are NOT
checked yet (continuum's shipping flow does that) — `--publish` is
unconditional. Use only on alloys you fully trust.

```bash
python -m factory_queue --root .factory --publish --org continuum-ai
```

### Mode 3 — Process exactly N parts then exit (testing)

```bash
python -m factory_queue --root .factory --max-iters 1   # build one part, exit
python -m factory_queue --root .factory --once           # also one, then exit
```

Use this for the first run on a fresh box: process one alloy, eyeball
the result in `.factory/line/finished/`, then run the full daemon.

## Operating as the human foreman

You're the foreman until continuum's grid layer ships. The foreman's
job is exactly what you'd think:

- **Add work** to the queue: `cp my-recipe.alloy.json .factory/line/intake/`
- **Reorder work**: bump file mtime via `touch` (oldest mtime is processed first)
- **Pause the node**: `kill <worker_pid>` (the next startup recovers
  whatever was mid-build) or stop dropping new parts in `intake/`
- **Cancel a part mid-build**: `kill <worker_pid>`, `mv assembly/<name>
  rework/` by hand, then restart the daemon
- **Promote a finished artifact to publish**: read
  `.factory/line/finished/<name>.result.json`, eyeball the assayed
  scores against the alloy's `acceptanceCriteria`, then run
  `python scripts/publish_model.py <forged_dir>` if it looks good
- **Demote a finished artifact**: `mv finished/<name>.alloy.json
  rework/<name>.alloy.json` and write a `.error.json` sidecar saying
  why
- **Reset a stuck retry counter**: rename
  `intake/<alloy>.retry3.alloy.json` → `intake/<alloy>.alloy.json`
  to give it a fresh attempt

The throughput log (`.factory/line/throughput.jsonl`) is the audit
trail: every state transition lands there. Reading the tail tells you
exactly what the line did since the last restart.

## Filesystem at a glance

```
.factory/
├── line/
│   ├── intake/                          ← drop alloys here
│   ├── assembly/                        ← worker's current part (1 at a time)
│   ├── finished/                        ← built + assayed, awaits release
│   ├── rework/                          ← failed QA, needs human triage
│   ├── .heartbeat.json                  ← daemon state (atomic-write)
│   ├── .worker.pid                      ← lock (one daemon per line)
│   └── throughput.jsonl                 ← append-only audit log
├── work/                                ← forge output (one dir per alloy)
└── calibration/                         ← held-out corpora the alloys ref
    └── heldout_code300.jsonl
```

## What's the kick-off command for the loaded queue (12 parts)?

```bash
ssh bigmama
cd ~/sentinel-ai && git pull && source .venv/bin/activate

# Bootstrap the .factory dir if it doesn't exist
mkdir -p .factory/calibration
# If the corpus isn't already there, copy it from the morning's forge:
cp ~/.continuum/forge-output/qwen3-coder-30b-a3b-compacted-19b-256k/calibration/heldout_code300.jsonl \
   .factory/calibration/

# Seed the queue from the catalog (drops 12 alloys into intake/)
python scripts/seed_factory_queue.py --root .factory

# First run: forge ONE part, no publish, check it
python -m factory_queue --root .factory --once

# If it looks good in finished/, run the daemon for the rest
nohup python -m factory_queue --root .factory > .factory/line/daemon.log 2>&1 &
echo "daemon started, pid $!"

# Check status from anywhere on the box
python scripts/factory_queue.py --root .factory --status
python scripts/factory_queue.py --root .factory --tail
```

That's it. One node, one human foreman, twelve forges queued, no
continuum dependency. The grid layer ships when it ships.

## Storage lifecycle (factory_storage)

Source models are 50–260GB each, intermediate forge work dirs add
another 100–200GB, and the queue typically has 12 parts in flight.
Without lifecycle management the box fills up after 4-5 forges and
the daemon dies on disk-full mid-build.

The daemon **auto-cleans before each new part** if disk pressure
crosses `--cleanup-threshold` (default 85% used). Auto-cleanup is
conservative: it only removes ORPHAN work dirs (`work/<name>/` where
`<name>` matches no alloy in any station). Anything ambiguous stays
put until you intervene.

```bash
# What's on disk right now?
python -m factory_storage --root .factory --audit

# Just the disk pressure
python -m factory_storage --root .factory --pressure

# List orphan work dirs (unused, safe to delete)
python -m factory_storage --root .factory --orphans

# List stale work dirs (older than N days, may still be referenced)
python -m factory_storage --root .factory --stale --days 14

# Manual cleanup — show what WOULD be deleted, no changes
python -m factory_storage --root .factory --cleanup --dry-run --force

# Manual cleanup — actually delete orphans (bypass threshold)
python -m factory_storage --root .factory --cleanup --force
```

**Cold tier (the 7200rpm spinner):**

```bash
# One-time: format + mount the cold drive at /mnt/cold
sudo mkfs.ext4 /dev/sdX1 && sudo mount /dev/sdX1 /mnt/cold
sudo chown $USER:$USER /mnt/cold

# Tell the daemon to MOVE evictions to cold instead of deleting
python -m factory_queue --root .factory --cleanup-cold-root /mnt/cold

# Or for a one-shot manual cleanup pass via factory_storage:
python -m factory_storage --root .factory --cleanup --force --cold-root /mnt/cold
```

When `--cleanup-cold-root` is set, the daemon's auto-cleanup pass MOVES
orphan work dirs to the cold drive instead of deleting them. The
reference set semantics are unchanged; only the destination changes.
HF re-fetch is still the fallback if both copies are gone (the cold
drive is the warm cache for things you might re-forge soon, not a
permanent vault).

Recommended directory layout when cold tier is online:

```
/mnt/cold/
├── work-archive/        ← evicted forge work dirs (intermediate state,
│                          re-fetchable from HF if needed)
├── source-cache/        ← optional: HF cache mirror so re-downloads
│                          come from cold instead of the network
└── published-backup/    ← optional: backup of finished/ artifacts that
                           already shipped to HF (belt-and-suspenders)
```

The daemon today only writes `work-archive/`. Source cache and
published backup are foreman-managed (manual `cp -al` or `rsync`).

When the cold drive lands, evictions become "moved to cold storage"
instead of "deleted." The reference set is the same; only the
destination changes. HF re-fetch is still the fallback if both copies
are gone.

**The reference set — what auto-cleanup will NEVER touch:**

- Calibration corpora in `.factory/calibration/`
- Anything referenced by an alloy in `intake/` or `assembly/`
- Work dirs for finished alloys within the last 7 days
- Files with mtime in the last 24 hours

Every eviction is logged to `throughput.jsonl` as
`{outcome: "evicted", path, bytes, action}` so the audit trail is
preserved.

## What still needs the grid (continuum) to come online

These work without continuum but feel less polished:

- **Multi-node coordination** — running this daemon on TWO forge boxes
  in parallel needs continuum to know which box gets which alloy
- **Quality-gate enforcement** — `acceptanceCriteria` is in the alloy
  but nothing reads it automatically yet; you eyeball it
- **Automatic publish** — `--publish` is unconditional today; the
  conditional version (publish iff acceptance gate clears) is in
  continuum
- **VRAM-aware scheduling** — the daemon doesn't check free VRAM; if
  you queue a 24GB Mixtral 8x22B alloy on a node with another job
  using 12GB it'll OOM and land in rework/
- **Cross-node throughput dashboard** — for now, just `tail -f
  .factory/line/throughput.jsonl` on each box

All of these are continuum's work, not sentinel's. The hive node
exposes the right surface area; continuum will plug into it.
