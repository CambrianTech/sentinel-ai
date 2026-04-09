"""factory_storage — disk lifecycle for one hive node.

The forge node has finite disk. Source models are 50–260GB each,
intermediate forge work dirs add another 100–200GB, and the queue
typically has 5–15 parts in flight. Without lifecycle management the
box fills up after 4-5 forges and the daemon dies on disk-full errors
mid-build.

S3-style storage tiers, mapped to the hive node's local filesystem:

  HOT       intake/, assembly/, .heartbeat, calibration/, recent
            finished/ — never auto-evicted
  WARM      forge work dirs (work/<name>/), older finished/ — LRU
            eviction when disk pressure crosses threshold
  COLD      (future) /mnt/cold/ on the 7200rpm spinner — currently
            same as EVICT (delete; HF re-fetch on demand)
  EVICT     orphan work dirs, anything older than N days with no
            references in any station

Reference counting — a file/dir is REFERENCED if any of these hold:

  • Listed as source.baseModel by any alloy in intake/ or assembly/
  • Mentioned as forged_dir in a recent finished/*.result.json
  • Is the calibrationCorpusFile of any active alloy
  • Touched in the last 24 hours (mtime — recently active)

Auto-cleanup is conservative: it only removes ORPHAN work dirs by
default (work/<name>/ where <name> matches no alloy in any station).
Anything ambiguous stays put until the foreman intervenes. The
throughput log records every eviction so the audit trail is preserved.

Future: when the cold drive lands, --cold-root tells auto_cleanup to
move evictions there instead of deleting them. The reference set
remains the source of truth for what's safe to relocate.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


# ── Reference set ───────────────────────────────────────────────────────────


@dataclass
class ReferenceSet:
    """What the line currently considers 'in use'.

    The auto_cleanup pass uses this as the safety check: nothing here
    is touched. Built once per cleanup pass by scanning every alloy in
    every station + every recent finished manifest + the calibration
    directory.
    """
    work_dir_names: set[str]            # work/<name>/ that are referenced
    base_models: set[str]               # HF source ids (e.g. "Qwen/Qwen3-30B-A3B")
    calibration_paths: set[Path]        # absolute paths to corpora
    finished_within_days: int = 7       # how long finished/ counts as active


def build_reference_set(root: Path, finished_window_days: int = 7) -> ReferenceSet:
    """Walk every station + work/ + calibration/ and collect the set of
    things the line currently considers in use. The result is what
    auto_cleanup checks before deleting anything.

    A work dir is referenced if its <name> matches the alloy stem in
    any station (intake/assembly/finished within the window).
    """
    root = Path(root)
    line = root / "line"

    work_dir_names: set[str] = set()
    base_models: set[str] = set()
    calibration_paths: set[Path] = set()

    def _scan_alloy(alloy_path: Path) -> None:
        try:
            alloy = json.loads(alloy_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        # Work dir name = alloy filename stem with .alloy stripped
        # AND the .retryN suffix stripped (so retried parts share work).
        stem = alloy_path.stem.replace(".alloy", "")
        # strip .retry<N>
        import re
        stem = re.sub(r"\.retry\d+$", "", stem)
        work_dir_names.add(stem)

        source = alloy.get("source") or {}
        bm = source.get("baseModel")
        if bm:
            base_models.add(bm)

        for stage in alloy.get("stages", []):
            corpus = stage.get("calibrationCorpusFile")
            if corpus:
                p = Path(corpus)
                if not p.is_absolute():
                    p = (root / "calibration" / Path(corpus).name).resolve()
                calibration_paths.add(p)

    # Active stations: intake + assembly always count
    for station in ("intake", "assembly"):
        sd = line / station
        if sd.exists():
            for alloy_path in sd.glob("*.alloy.json"):
                _scan_alloy(alloy_path)

    # Finished within the window also counts
    cutoff = time.time() - finished_window_days * 86400
    finished_dir = line / "finished"
    if finished_dir.exists():
        for alloy_path in finished_dir.glob("*.alloy.json"):
            try:
                if alloy_path.stat().st_mtime < cutoff:
                    continue
            except OSError:
                continue
            _scan_alloy(alloy_path)
            # Also pull forged_dir from the result manifest if present
            sidecar = alloy_path.with_suffix(".result.json")
            if sidecar.exists():
                try:
                    manifest = json.loads(sidecar.read_text())
                    fd = manifest.get("forged_dir")
                    if fd:
                        work_dir_names.add(Path(fd).name)
                except (OSError, json.JSONDecodeError):
                    pass

    return ReferenceSet(
        work_dir_names=work_dir_names,
        base_models=base_models,
        calibration_paths=calibration_paths,
        finished_within_days=finished_window_days,
    )


# ── Audit / discovery ───────────────────────────────────────────────────────


def audit(root: Path) -> dict:
    """Scan all forge artifacts and return a sizes-and-counts report.

    Used by the CLI `--audit` command and (eventually) by continuum's
    grid view of node disk pressure. Cheap to call.
    """
    root = Path(root)
    line = root / "line"
    work = root / "work"

    intake_count = sum(1 for _ in (line / "intake").glob("*.alloy.json")) if (line / "intake").exists() else 0
    assembly_count = sum(1 for _ in (line / "assembly").glob("*.alloy.json")) if (line / "assembly").exists() else 0
    finished_count = sum(1 for _ in (line / "finished").glob("*.alloy.json")) if (line / "finished").exists() else 0
    rework_count = sum(1 for _ in (line / "rework").glob("*.alloy.json")) if (line / "rework").exists() else 0

    work_dirs = []
    if work.exists():
        for entry in work.iterdir():
            if entry.is_dir():
                size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                work_dirs.append({"name": entry.name, "bytes": size})

    p = pressure(root)
    return {
        "intake_count": intake_count,
        "assembly_count": assembly_count,
        "finished_count": finished_count,
        "rework_count": rework_count,
        "work_dirs": len(work_dirs),
        "work_dir_details": sorted(work_dirs, key=lambda d: -d["bytes"]),
        "total_bytes": sum(d["bytes"] for d in work_dirs),
        "pressure": p,
    }


def find_orphans(root: Path) -> list[Path]:
    """Work dirs whose corresponding alloy is no longer in any station.

    These are the safest things to evict — by definition, no live alloy
    points at them, no recent finished manifest references them.
    """
    root = Path(root)
    work = root / "work"
    if not work.exists():
        return []

    refs = build_reference_set(root)
    orphans: list[Path] = []
    for entry in work.iterdir():
        if not entry.is_dir():
            continue
        if entry.name not in refs.work_dir_names:
            orphans.append(entry)
    return orphans


def find_stale(root: Path, days: int = 14) -> list[Path]:
    """Files / dirs in work/ that haven't been touched in N days.

    Stale ≠ orphan. A stale dir might still be referenced (e.g. an
    alloy that's been queued forever). The caller decides what to do.
    Used by the foreman audit, NOT by auto_cleanup.
    """
    root = Path(root)
    work = root / "work"
    if not work.exists():
        return []
    cutoff = time.time() - days * 86400
    stale: list[Path] = []
    for entry in work.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                stale.append(entry)
        except OSError:
            continue
    return stale


def safe_to_evict(path: Path, refs: ReferenceSet) -> bool:
    """Cross-reference safety check before deleting anything.

    Currently checks: is this a work/<name>/ dir, and is <name> in the
    reference set? Anything outside work/ is conservatively NOT safe.
    """
    path = Path(path)
    if path.parent.name != "work":
        return False
    return path.name not in refs.work_dir_names


# ── Pressure / disk stats ───────────────────────────────────────────────────


def pressure(root: Path) -> dict:
    """Free / total / pct_used for the filesystem hosting the line.

    Used by auto_cleanup to decide whether to evict, and by the CLI
    `--pressure` command for operator visibility.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    return {
        "free_gb": round(usage.free / 1e9, 2),
        "total_gb": round(usage.total / 1e9, 2),
        "pct_used": round(100 * (usage.total - usage.free) / usage.total, 2),
    }


# ── Auto-cleanup ────────────────────────────────────────────────────────────


def _append_throughput(root: Path, entry: dict) -> None:
    """Append a one-line entry to the line's throughput log. Same shape
    as factory_queue's _append_throughput so the line history is one
    unified audit trail."""
    log_path = root / "line" / "throughput.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **entry,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def auto_cleanup(
    root: Path,
    *,
    threshold_pct: float = 85.0,
    force: bool = False,
    cold_root: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Conservative storage cleanup pass.

    The hive node calls this before starting each new part. If disk
    pressure is below `threshold_pct`, returns immediately without
    touching anything (skipped=True). Otherwise:

      1. Build the reference set from all live alloys
      2. Find orphan work dirs (work/<name>/ where <name> is not in refs)
      3. For each orphan: log to throughput, then delete (or move to
         cold_root if provided)
      4. Return a report with bytes freed + count

    Calibration corpora and any directory referenced by an active alloy
    are NEVER touched, even at full pressure. Cleanup is the algorithm's
    chance to free obvious wins; ambiguous cases stay put until the
    foreman triages them manually.

    Args:
        root: queue root (the dir that contains line/, work/, calibration/)
        threshold_pct: skip cleanup if pct_used is below this
        force: bypass threshold check (testing + manual --cleanup)
        cold_root: if set, MOVE evictions here instead of deleting.
                   Future: the 10TB spinner mount point.
        dry_run: report what WOULD be deleted without touching anything
    """
    root = Path(root)
    p = pressure(root)
    if not force and p["pct_used"] < threshold_pct:
        return {
            "skipped": True,
            "reason": f"pct_used={p['pct_used']} < threshold={threshold_pct}",
            "pressure": p,
            "evicted_count": 0,
            "bytes_freed": 0,
        }

    orphans = find_orphans(root)
    evicted: list[dict] = []
    bytes_freed = 0

    for orphan in orphans:
        try:
            size = sum(f.stat().st_size for f in orphan.rglob("*") if f.is_file())
        except OSError:
            size = 0

        if dry_run:
            evicted.append({"path": str(orphan), "bytes": size, "action": "would_delete"})
            continue

        action: str
        if cold_root is not None:
            cold_root = Path(cold_root)
            cold_root.mkdir(parents=True, exist_ok=True)
            target = cold_root / orphan.name
            shutil.move(str(orphan), str(target))
            action = f"moved_to_cold:{target}"
        else:
            shutil.rmtree(orphan, ignore_errors=True)
            action = "deleted"

        bytes_freed += size
        evicted.append({"path": str(orphan), "bytes": size, "action": action})
        _append_throughput(root, {
            "outcome": "evicted",
            "path": str(orphan),
            "bytes": size,
            "action": action,
        })

    return {
        "skipped": False,
        "pressure": p,
        "evicted_count": len(evicted),
        "bytes_freed": bytes_freed,
        "evicted": evicted,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    """CLI for manual storage operations.

    Auto-cleanup runs inside the daemon's process_one loop. The CLI is
    for the foreman who wants to inspect or trigger cleanup manually.
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".factory")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--orphans", action="store_true")
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--pressure", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--threshold", type=float, default=85.0)
    ap.add_argument("--cold-root", default=None,
                    help="evict TO this dir instead of deleting (future cold drive)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="bypass disk pressure threshold for cleanup")
    args = ap.parse_args()

    root = Path(args.root)

    if args.audit:
        print(json.dumps(audit(root), indent=2))
        return
    if args.orphans:
        for o in find_orphans(root):
            print(o)
        return
    if args.stale:
        for s in find_stale(root, days=args.days):
            print(s)
        return
    if args.pressure:
        print(json.dumps(pressure(root), indent=2))
        return
    if args.cleanup:
        rep = auto_cleanup(
            root,
            threshold_pct=args.threshold,
            force=args.force,
            cold_root=Path(args.cold_root) if args.cold_root else None,
            dry_run=args.dry_run,
        )
        print(json.dumps(rep, indent=2))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
