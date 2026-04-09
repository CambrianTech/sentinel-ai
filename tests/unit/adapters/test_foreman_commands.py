"""TDD spec for foreman convenience commands.

The hive node daemon is the assembly line. The foreman (you, or
continuum's grid layer when it ships) needs convenience commands to
inspect, promote, and triage parts WITHOUT typing `mv` and `cp` by
hand. These are all small disk operations on the line/ directory; the
daemon doesn't need to be running.

Commands added in this round:

  list_parts(station)        — return alloy summaries from one station
  retry_rework(name)         — move rework/<name> → intake/, reset
                                .retryN counter so the part gets a
                                fresh attempt
  enqueue_path(path)         — alias for q.enqueue with normalization

CLI surface:
  --list                     pretty-print intake station
  --list-station <name>      pretty-print any station (intake|assembly|finished|rework)
  --retry <name>             promote a rework part back to intake
  --enqueue <path>           drop an alloy file into intake (atomic)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _alloy(name: str, **extras) -> dict:
    return {
        "name": name,
        "source": {"baseModel": f"Test/{name}", "architecture": "qwen3_moe"},
        "stages": [],
        **extras,
    }


def _write(path: Path, name: str, **extras) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_alloy(name, **extras)))
    return path


# ── list_parts ──────────────────────────────────────────────────────────────


def test_list_parts_returns_summaries_for_intake(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    q.enqueue(_write(tmp_path / "a.alloy.json", "a"))
    q.enqueue(_write(tmp_path / "b.alloy.json", "b"))

    parts = q.list_parts("intake")
    assert len(parts) == 2
    names = {p["name"] for p in parts}
    assert names == {"a", "b"}
    # Each entry carries enough fields for a pretty print
    assert all("filename" in p and "name" in p and "base_model" in p for p in parts)


def test_list_parts_handles_empty_station(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    assert q.list_parts("rework") == []


def test_list_parts_invalid_station_raises(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    with pytest.raises(ValueError):
        q.list_parts("nonexistent_station")


def test_list_parts_includes_retry_marker_when_present(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    # Synthesize a retried part directly
    p = q.intake_dir / "a.retry2.alloy.json"
    p.write_text(json.dumps(_alloy("a")))
    parts = q.list_parts("intake")
    assert parts[0]["retries"] == 2


# ── retry_rework ────────────────────────────────────────────────────────────


def test_retry_rework_promotes_alloy_back_to_intake(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    # Synthesize a part in rework/ with a sidecar
    name = "a.alloy.json"
    rework_path = q.rework_dir / name
    rework_path.write_text(json.dumps(_alloy("a")))
    sidecar = rework_path.with_suffix(".error.json")
    sidecar.write_text('{"error": "boom"}')

    promoted = q.retry_rework(name)
    assert promoted is not None
    assert promoted.parent == q.intake_dir
    # Sidecar moved with it (preserving the error trail) but alloy is back
    assert not rework_path.exists()


def test_retry_rework_resets_retry_counter(tmp_path):
    """A part that was rework'd at retry3 gets re-queued at retry0 — the
    foreman is explicitly giving it another chance."""
    from factory_queue import FactoryQueue, _retry_count
    q = FactoryQueue(tmp_path)
    name = "a.retry3.alloy.json"
    (q.rework_dir / name).write_text(json.dumps(_alloy("a")))

    promoted = q.retry_rework(name)
    assert _retry_count(promoted.name) == 0


def test_retry_rework_missing_returns_none(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    assert q.retry_rework("nonexistent.alloy.json") is None


def test_retry_rework_logs_to_throughput(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    (q.rework_dir / "a.alloy.json").write_text(json.dumps(_alloy("a")))

    q.retry_rework("a.alloy.json")
    log = q.throughput_log_path.read_text().splitlines()
    entries = [json.loads(line) for line in log if "promoted_from_rework" in line]
    assert len(entries) == 1
    assert entries[0]["outcome"] == "promoted_from_rework"


# ── enqueue (already exists; just round-trip with normalization) ────────────


def test_enqueue_accepts_string_path(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    src = _write(tmp_path / "a.alloy.json", "a")
    enqueued = q.enqueue(str(src))  # accept str, not just Path
    assert enqueued.exists()


def test_enqueue_raises_on_missing_file(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    with pytest.raises(FileNotFoundError):
        q.enqueue(tmp_path / "nonexistent.alloy.json")
