"""TDD spec for factory_storage — disk lifecycle for one hive node.

The forge node has finite disk (2-4TB SSD today, +8-10TB cold drive
soon). Source models are 50-260GB each, intermediate forge work dirs
add another 100-200GB, and the queue has 12 parts. Without lifecycle
management the box fills up after 4-5 forges and the daemon dies on
disk-full errors mid-build.

This module implements the algorithmic side of S3-style storage tiers:

  HOT       intake/, assembly/, .heartbeat, calibration/, recent
            finished/ — never evicted
  WARM      forge work dirs, older finished/ — LRU eviction when disk
            pressure crosses threshold
  COLD      (future) /mnt/cold/ on the 7200rpm spinner — currently:
            same as EVICT (delete; HF re-fetch on demand)
  EVICT     orphan work dirs, anything older than N days with no
            references in any station

Reference counting: a file/dir is REFERENCED if any of these hold:
  • Listed as source.baseModel by any alloy in intake/ or assembly/
  • Mentioned as forged_dir in a finished/ result manifest <7 days old
  • Path is the calibrationCorpusFile of any active alloy
  • Touched in the last 24h (mtime — recently active)

Public surface:
  audit(root)           → AuditReport (sizes by tier, eviction candidates)
  find_orphans(root)    → list[Path] (work dirs whose alloy is gone)
  find_stale(root,days) → list[Path] (touched > N days ago, unreferenced)
  pressure(root)        → DiskPressure (free_gb, total_gb, pct_used)
  safe_to_evict(p, ref) → bool (cross-reference safety check)
  auto_cleanup(root,    → CleanupReport (what was deleted, freed bytes)
               threshold_pct)

Daemon integration: process_one() calls auto_cleanup() if pressure
crosses the configured threshold (default 85% used) BEFORE starting
the next part. The cleanup is logged to throughput.jsonl as one
'evicted' entry per artifact removed.

Future cold-tier: when --cold-root is passed, evictions move to the
cold drive instead of being deleted. Today --cold-root is a stub.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _write_alloy(path: Path, name: str, base_model: str = "Test/Base") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": name,
        "source": {"baseModel": base_model, "architecture": "test"},
        "stages": [],
    }))
    return path


def _write_work_dir(work_root: Path, name: str, size_kb: int = 4) -> Path:
    d = work_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(b"x" * (size_kb * 1024))
    return d


# ── audit ───────────────────────────────────────────────────────────────────


def test_audit_returns_report_with_sizes(tmp_path):
    from factory_queue import FactoryQueue
    from factory_storage import audit
    q = FactoryQueue(tmp_path)
    _write_alloy(q.intake_dir / "a.alloy.json", "a")
    _write_work_dir(tmp_path / "work", "a", size_kb=10)

    report = audit(tmp_path)
    assert report["intake_count"] == 1
    assert report["work_dirs"] >= 1
    assert report["total_bytes"] > 0


# ── find_orphans: work dirs whose alloy is gone ─────────────────────────────


def test_find_orphans_returns_work_dirs_with_no_corresponding_alloy(tmp_path):
    """work/foo/ exists but no alloy named 'foo' is in any station — orphan."""
    from factory_queue import FactoryQueue
    from factory_storage import find_orphans
    q = FactoryQueue(tmp_path)

    # Live: work/live/ AND intake/_seed_live.alloy.json
    _write_alloy(q.intake_dir / "_seed_live.alloy.json", "live")
    _write_work_dir(tmp_path / "work", "_seed_live")

    # Orphan: work/abandoned/ but no alloy anywhere
    _write_work_dir(tmp_path / "work", "abandoned")

    orphans = find_orphans(tmp_path)
    orphan_names = {p.name for p in orphans}
    assert "abandoned" in orphan_names
    assert "_seed_live" not in orphan_names


def test_find_orphans_keeps_work_dirs_referenced_by_finished_manifest(tmp_path):
    """A work dir referenced by a recent finished/ manifest is NOT orphaned."""
    from factory_queue import FactoryQueue
    from factory_storage import find_orphans
    q = FactoryQueue(tmp_path)

    work = _write_work_dir(tmp_path / "work", "_seed_done")
    # Move alloy through to finished/ with manifest pointing at work dir
    alloy = _write_alloy(q.intake_dir / "_seed_done.alloy.json", "done")
    a2 = q.pop_oldest_intake()
    q.mark_finished(a2, {"forged_dir": str(work)})

    orphans = find_orphans(tmp_path)
    assert work not in orphans


# ── find_stale: old, unreferenced work dirs ─────────────────────────────────


def test_find_stale_returns_files_older_than_threshold(tmp_path):
    from factory_queue import FactoryQueue
    from factory_storage import find_stale
    FactoryQueue(tmp_path)  # creates the dir tree

    old = _write_work_dir(tmp_path / "work", "old")
    # Backdate
    eight_days = time.time() - 8 * 86400
    import os
    os.utime(old, (eight_days, eight_days))
    for child in old.iterdir():
        os.utime(child, (eight_days, eight_days))

    fresh = _write_work_dir(tmp_path / "work", "fresh")

    stale = find_stale(tmp_path, days=7)
    stale_names = {p.name for p in stale}
    assert "old" in stale_names
    assert "fresh" not in stale_names


# ── safe_to_evict: cross-reference safety ───────────────────────────────────


def test_safe_to_evict_refuses_dir_referenced_by_intake(tmp_path):
    from factory_queue import FactoryQueue
    from factory_storage import safe_to_evict, build_reference_set
    q = FactoryQueue(tmp_path)
    _write_alloy(q.intake_dir / "_seed_a.alloy.json", "a")
    work = _write_work_dir(tmp_path / "work", "_seed_a")

    refs = build_reference_set(tmp_path)
    assert safe_to_evict(work, refs) is False


def test_safe_to_evict_allows_orphan_dir(tmp_path):
    from factory_queue import FactoryQueue
    from factory_storage import safe_to_evict, build_reference_set
    FactoryQueue(tmp_path)
    work = _write_work_dir(tmp_path / "work", "abandoned")

    refs = build_reference_set(tmp_path)
    assert safe_to_evict(work, refs) is True


# ── pressure: free disk on the line root ────────────────────────────────────


def test_pressure_returns_disk_stats(tmp_path):
    from factory_storage import pressure
    p = pressure(tmp_path)
    assert "free_gb" in p
    assert "total_gb" in p
    assert "pct_used" in p
    assert 0.0 <= p["pct_used"] <= 100.0


# ── auto_cleanup ────────────────────────────────────────────────────────────


def test_auto_cleanup_removes_orphans_only(tmp_path):
    from factory_queue import FactoryQueue
    from factory_storage import auto_cleanup
    q = FactoryQueue(tmp_path)
    _write_alloy(q.intake_dir / "_seed_keep.alloy.json", "keep")
    _write_work_dir(tmp_path / "work", "_seed_keep")          # referenced
    _write_work_dir(tmp_path / "work", "abandoned")           # orphan

    # Force cleanup regardless of disk pressure for testing
    report = auto_cleanup(tmp_path, force=True)
    assert (tmp_path / "work" / "abandoned").exists() is False
    assert (tmp_path / "work" / "_seed_keep").exists() is True
    assert report["evicted_count"] >= 1
    assert report["bytes_freed"] > 0


def test_auto_cleanup_skips_when_pressure_below_threshold(tmp_path):
    """If disk is comfortable, cleanup is a no-op (report.skipped=True)."""
    from factory_queue import FactoryQueue
    from factory_storage import auto_cleanup
    q = FactoryQueue(tmp_path)
    _write_work_dir(tmp_path / "work", "abandoned")

    # threshold so high it'll never fire
    report = auto_cleanup(tmp_path, threshold_pct=99.9, force=False)
    assert report["skipped"] is True
    # Orphan still there because cleanup didn't run
    assert (tmp_path / "work" / "abandoned").exists() is True


def test_auto_cleanup_logs_evictions_to_throughput(tmp_path):
    from factory_queue import FactoryQueue
    from factory_storage import auto_cleanup
    q = FactoryQueue(tmp_path)
    _write_work_dir(tmp_path / "work", "abandoned")

    auto_cleanup(tmp_path, force=True)
    log = q.throughput_log_path.read_text().splitlines()
    evictions = [json.loads(line) for line in log if "evicted" in line]
    assert len(evictions) >= 1
    assert evictions[0]["outcome"] == "evicted"
    assert "abandoned" in evictions[0]["path"]


def test_auto_cleanup_keeps_calibration_corpus(tmp_path):
    """Calibration corpora are HOT — never evicted by auto_cleanup."""
    from factory_storage import auto_cleanup
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    corpus = cal_dir / "heldout_code300.jsonl"
    corpus.write_text('{"code":"def f(): pass"}\n')
    # Backdate
    import os
    eight_days = time.time() - 8 * 86400
    os.utime(corpus, (eight_days, eight_days))

    auto_cleanup(tmp_path, force=True)
    assert corpus.exists(), "calibration corpus must NEVER be auto-evicted"


# ── Daemon integration: process_one calls auto_cleanup ──────────────────────


def test_auto_cleanup_moves_orphans_to_cold_root_when_set(tmp_path):
    """When cold_root is provided, evictions MOVE to cold instead of
    being deleted. The 7200rpm spinner integration."""
    from factory_queue import FactoryQueue
    from factory_storage import auto_cleanup
    q = FactoryQueue(tmp_path)
    orphan = _write_work_dir(tmp_path / "work", "abandoned")
    cold = tmp_path / "mnt" / "cold"

    report = auto_cleanup(tmp_path, force=True, cold_root=cold)
    # Original deleted, copy on cold tier
    assert not orphan.exists()
    assert (cold / "abandoned").exists()
    assert report["evicted_count"] == 1
    # Throughput log records the action with the cold-tier path
    log = q.throughput_log_path.read_text().splitlines()
    cold_entries = [json.loads(line) for line in log if "moved_to_cold" in line]
    assert len(cold_entries) == 1
    assert "moved_to_cold" in cold_entries[0]["action"]
    assert str(cold / "abandoned") in cold_entries[0]["action"]


def test_worker_passes_cold_root_to_cleanup_fn(tmp_path):
    """FactoryWorker.cleanup_cold_root threads through to cleanup_fn."""
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    _write_alloy(q.intake_dir / "_seed_a.alloy.json", "a")

    cleanup_kwargs: list[dict] = []
    def fake_executor(alloy_path, output_dir=None, dry_run=False):
        out = Path(output_dir or (Path(alloy_path).parent / "out"))
        out.mkdir(parents=True, exist_ok=True)
        return out

    def fake_cleanup(root, **kwargs):
        cleanup_kwargs.append(kwargs)
        return {"skipped": False, "evicted_count": 0, "bytes_freed": 0}

    cold = tmp_path / "mnt" / "cold"
    w = FactoryWorker(
        q, executor=fake_executor,
        work_root=tmp_path / "work",
        cleanup_fn=fake_cleanup,
        cleanup_cold_root=cold,
    )
    w.process_one()
    assert len(cleanup_kwargs) == 1
    assert cleanup_kwargs[0]["cold_root"] == cold


def test_worker_process_one_runs_cleanup_when_pressure_high(tmp_path):
    """The daemon's process_one should auto_cleanup BEFORE starting a
    new part if disk pressure crosses the threshold. Verified via a
    spy that records calls."""
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    _write_alloy(q.intake_dir / "_seed_a.alloy.json", "a")
    _write_work_dir(tmp_path / "work", "abandoned")

    cleanup_calls = []

    def fake_executor(alloy_path, output_dir=None, dry_run=False):
        out = Path(output_dir or (Path(alloy_path).parent / "out"))
        out.mkdir(parents=True, exist_ok=True)
        return out

    def fake_cleanup(root, **kwargs):
        cleanup_calls.append(root)
        return {"skipped": False, "evicted_count": 0, "bytes_freed": 0}

    w = FactoryWorker(
        q, executor=fake_executor,
        work_root=tmp_path / "work",
        cleanup_fn=fake_cleanup,
    )
    w.process_one()
    assert len(cleanup_calls) == 1
