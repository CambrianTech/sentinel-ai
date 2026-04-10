"""TDD spec: factory_queue daemon — one hive node's local concerns.

Continuum's grid is the foreman. It decides which node builds what,
when to pause, what fits in VRAM, what to prioritize. THIS module is
ONE node's local executor — a dumb worker that:

  1. Runs forever as a daemon (no --max-iters exit)
  2. Recovers crashed assembly/ parts on startup (no work lost)
  3. Writes a heartbeat the grid can poll
  4. Holds a PID lock (one worker per line)
  5. Tracks retry counts so a permanently broken part eventually gives up
  6. Logs throughput (every state transition) for the grid to read
  7. Reports status on demand

Anything beyond this — foreman, priority, VRAM check, pause policy,
node selection — lives in continuum's grid layer, not here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


SYNTHETIC_ALLOY = {
    "name": "test-alloy",
    "source": {"architecture": "qwen3_moe", "baseModel": "Qwen/Test-Base"},
    "stages": [],
}


def _write_alloy(path: Path, alloy: dict | None = None) -> Path:
    path.write_text(json.dumps(alloy or SYNTHETIC_ALLOY, indent=2))
    return path


class _FakeExecutor:
    def __init__(self, raise_on=None):
        self.calls = []
        self.raise_on = raise_on or set()

    def __call__(self, alloy_path, output_dir=None, dry_run=False):
        self.calls.append(str(alloy_path))
        stem = Path(alloy_path).stem.replace(".alloy", "").split(".retry")[0]
        if stem in self.raise_on:
            raise RuntimeError(f"forge boom for {stem}")
        out = Path(output_dir or (Path(alloy_path).parent / "out"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(b"fake")
        return out


# ── Heartbeat ───────────────────────────────────────────────────────────────


def test_heartbeat_path_under_line_dir(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    assert q.heartbeat_path == tmp_path / "line" / ".heartbeat.json"


def test_write_heartbeat_records_state(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    q.write_heartbeat(state="building", current_part="a.alloy.json")
    data = json.loads(q.heartbeat_path.read_text())
    assert data["pid"] == os.getpid()
    assert data["state"] == "building"
    assert data["current_part"] == "a.alloy.json"
    assert "last_beat_at" in data


def test_read_heartbeat_returns_none_when_absent(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    assert q.read_heartbeat() is None


# ── PID lock ────────────────────────────────────────────────────────────────


def test_acquire_pid_lock_writes_self_pid(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    assert q.acquire_pid_lock() is True
    assert q.pid_path.read_text().strip() == str(os.getpid())
    q.release_pid_lock()


def test_acquire_pid_lock_refuses_when_live_pid_present(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    q.pid_path.write_text(str(os.getpid()))  # we're alive
    assert q.acquire_pid_lock() is False


def test_acquire_pid_lock_cleans_stale_pid(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    q.pid_path.write_text("99999999")  # very-likely-dead
    assert q.acquire_pid_lock() is True


# ── Crash recovery ──────────────────────────────────────────────────────────


def test_recover_assembly_moves_stuck_parts_to_intake(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    q.enqueue(_write_alloy(tmp_path / "a.alloy.json"))
    stuck = q.pop_oldest_intake()
    assert stuck.parent == q.assembly_dir

    recovered = q.recover_assembly()
    assert len(recovered) == 1
    intake_files = list(q.intake_dir.glob("*.alloy.json"))
    assert len(intake_files) == 1
    assert "retry1" in intake_files[0].name


def test_recover_assembly_moves_to_rework_after_max_retries(tmp_path):
    from factory_queue import FactoryQueue, MAX_RETRIES
    q = FactoryQueue(tmp_path)
    name = f"a.retry{MAX_RETRIES}.alloy.json"
    (q.assembly_dir / name).write_text(json.dumps(SYNTHETIC_ALLOY))

    recovered = q.recover_assembly()
    assert len(recovered) == 1
    assert list(q.intake_dir.glob("*.alloy.json")) == []
    assert len(list(q.rework_dir.glob("*.alloy.json"))) == 1


def test_retry_count_helper():
    from factory_queue import _retry_count
    assert _retry_count("a.alloy.json") == 0
    assert _retry_count("a.retry1.alloy.json") == 1
    assert _retry_count("a.retry3.alloy.json") == 3


# ── Throughput log ──────────────────────────────────────────────────────────


def test_mark_finished_appends_throughput_entry(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    q.enqueue(_write_alloy(tmp_path / "a.alloy.json"))
    running = q.pop_oldest_intake()
    q.mark_finished(running, {"forged_dir": "/tmp/x"})

    entry = json.loads(q.throughput_log_path.read_text().splitlines()[0])
    assert entry["outcome"] == "finished"
    assert entry["alloy"].endswith(".alloy.json")


def test_mark_rework_appends_throughput_entry(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    q.enqueue(_write_alloy(tmp_path / "a.alloy.json"))
    running = q.pop_oldest_intake()
    q.mark_rework(running, error="boom", traceback_text="tb")

    entry = json.loads(q.throughput_log_path.read_text().splitlines()[0])
    assert entry["outcome"] == "rework"
    assert entry["error"] == "boom"


# ── run_forever (the daemon loop) ───────────────────────────────────────────


def test_worker_run_forever_processes_intake_then_idles(tmp_path):
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    for i in range(3):
        q.enqueue(_write_alloy(tmp_path / f"a{i}.alloy.json", {**SYNTHETIC_ALLOY, "name": f"a{i}"}))

    sleeps = []
    def fake_sleep(s):
        sleeps.append(s)
        if sleeps.count(5.0) >= 2:
            raise KeyboardInterrupt

    w = FactoryWorker(q, executor=_FakeExecutor(), work_root=tmp_path / "work")
    w.run_forever(idle_sleep_seconds=5.0, sleep_fn=fake_sleep)
    assert q.stats() == {"intake": 0, "assembly": 0, "finished": 3, "rework": 0}
    assert sleeps.count(5.0) >= 2  # idled after draining


def test_worker_run_forever_recovers_on_startup(tmp_path):
    """A part stuck in assembly/ before the daemon started must be
    recovered to intake/ on startup, then processed."""
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    stuck = q.assembly_dir / "stuck.alloy.json"
    stuck.write_text(json.dumps({**SYNTHETIC_ALLOY, "name": "stuck"}))

    sleeps = []
    def fake_sleep(s):
        sleeps.append(s)
        if sleeps.count(5.0) >= 2:
            raise KeyboardInterrupt

    w = FactoryWorker(q, executor=_FakeExecutor(), work_root=tmp_path / "work")
    w.run_forever(idle_sleep_seconds=5.0, sleep_fn=fake_sleep)
    # The stuck part was recovered with a retry marker, then forged successfully
    assert q.stats()["finished"] == 1


# ── status ──────────────────────────────────────────────────────────────────


def test_status_with_no_heartbeat_reports_offline(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    s = q.status()
    assert s["state"] == "offline"
    assert "stats" in s


def test_status_reflects_heartbeat_state(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    q.enqueue(_write_alloy(tmp_path / "a.alloy.json"))
    q.write_heartbeat(state="building", current_part="a.alloy.json")
    s = q.status()
    assert s["state"] == "building"
    assert s["current_part"] == "a.alloy.json"
    assert s["stats"]["intake"] == 1
