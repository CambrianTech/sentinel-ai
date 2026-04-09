"""TDD spec for the BigMama factory queue + worker loop.

The factory pipeline turns "we have alloys + a forge + an eval registry +
a publisher" into "BigMama cranks models 24/7." The architecture is the
simplest possible disk-backed queue:

    .factory/line/intake/    ← drop alloy files here (cp, mv, generator)
    .factory/line/assembly/    ← worker moves alloy here while processing
    .factory/line/finished/       ← success: alloy + result manifest
    .factory/line/rework/     ← failure: alloy + traceback

The worker loop:
    1. Poll pending/ for the oldest .alloy.json
    2. atomically move it to running/
    3. Call execute_alloy(path) — runs prune/train/eval per the alloy
    4. Call publish(output_dir) — pushes to HF
    5. On success: move alloy to done/ + write result manifest
    6. On failure: move alloy to failed/ + write error log + traceback
    7. Loop

No DB, no service, no network coordination. The filesystem IS the queue.
Multi-worker support comes free if we ever need it via O_EXCL atomic
moves (single 5090 = single worker today).

Public API the test exercises:

    FactoryQueue(root) — disk-backed queue at <root>/queue/{pending,running,done,failed}
        .enqueue(alloy_path)               — copy alloy file into pending/
        .pop_oldest_intake() → Path|None  — atomic move pending→running
        .mark_finished(running_path, manifest) — move running→done with result
        .mark_rework(running_path, error)  — move running→failed with traceback
        .stats() → dict                    — counts of each bucket

    FactoryWorker(queue, executor, publisher)
        .process_one() → bool              — process exactly one alloy if any
        .run_loop(max_iters=None)          — poll loop, exits when pending empty + max_iters reached

Both executor and publisher are injected as callables so tests can pass
fakes (no GPU/HF in unit tests). Production wiring: executor =
alloy_executor.execute_alloy, publisher = publish_model.publish.

Written test-first per TDD discipline.
"""

from __future__ import annotations

import json
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


# ── FactoryQueue ────────────────────────────────────────────────────────────


def test_factory_queue_creates_directory_layout(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    for sub in ("intake", "assembly", "finished", "rework"):
        assert (tmp_path / "line" / sub).is_dir()


def test_factory_queue_enqueue_copies_alloy_into_pending(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    src = _write_alloy(tmp_path / "src.alloy.json")
    enqueued = q.enqueue(src)
    assert enqueued.parent == tmp_path / "line" / "intake"
    assert enqueued.exists()
    # Original is preserved (the queue is a copy, not a move)
    assert src.exists()


def test_pop_oldest_intake_returns_oldest_and_moves_to_assembly(tmp_path):
    import time
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    a = _write_alloy(tmp_path / "a.alloy.json", {**SYNTHETIC_ALLOY, "name": "a"})
    b = _write_alloy(tmp_path / "b.alloy.json", {**SYNTHETIC_ALLOY, "name": "b"})
    q.enqueue(a)
    time.sleep(0.01)
    q.enqueue(b)

    first = q.pop_oldest_intake()
    assert first is not None
    assert first.parent == tmp_path / "line" / "assembly"
    assert "a" in first.name

    second = q.pop_oldest_intake()
    assert second is not None
    assert "b" in second.name

    assert q.pop_oldest_intake() is None  # drained


def test_mark_finished_moves_assembly_to_finished_and_writes_manifest(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    a = _write_alloy(tmp_path / "a.alloy.json")
    q.enqueue(a)
    running = q.pop_oldest_intake()

    manifest = {"hf_repo": "continuum-ai/test-alloy", "alloyHash": "sha256:abc"}
    done_path = q.mark_finished(running, manifest)
    assert done_path.parent == tmp_path / "line" / "finished"
    assert done_path.exists()

    # Manifest sidecar lives next to the alloy
    sidecar = done_path.with_suffix(".result.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["hf_repo"] == "continuum-ai/test-alloy"


def test_mark_rework_moves_assembly_to_rework_and_writes_traceback(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    a = _write_alloy(tmp_path / "a.alloy.json")
    q.enqueue(a)
    running = q.pop_oldest_intake()

    failed_path = q.mark_rework(running, error="boom", traceback_text="Traceback...")
    assert failed_path.parent == tmp_path / "line" / "rework"
    sidecar = failed_path.with_suffix(".error.json")
    assert sidecar.exists()
    err = json.loads(sidecar.read_text())
    assert err["error"] == "boom"
    assert err["traceback"] == "Traceback..."


def test_stats_counts_stations(tmp_path):
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    a = _write_alloy(tmp_path / "a.alloy.json", {**SYNTHETIC_ALLOY, "name": "a"})
    b = _write_alloy(tmp_path / "b.alloy.json", {**SYNTHETIC_ALLOY, "name": "b"})
    q.enqueue(a); q.enqueue(b)
    stats = q.stats()
    assert stats == {"intake": 2, "assembly": 0, "finished": 0, "rework": 0}

    r = q.pop_oldest_intake()
    assert q.stats()["assembly"] == 1
    q.mark_finished(r, {"ok": True})
    assert q.stats() == {"intake": 1, "assembly": 0, "finished": 1, "rework": 0}


# ── FactoryWorker ───────────────────────────────────────────────────────────


class _FakeExecutor:
    """Records calls; returns a fake output_dir on success."""
    def __init__(self, raise_on=None):
        self.calls = []
        self.raise_on = raise_on or set()

    def __call__(self, alloy_path, output_dir=None, dry_run=False):
        self.calls.append(str(alloy_path))
        if Path(alloy_path).stem in self.raise_on:
            raise RuntimeError(f"forge boom for {alloy_path}")
        out = Path(output_dir or (Path(alloy_path).parent / "out"))
        out.mkdir(parents=True, exist_ok=True)
        # Simulate the executor having dropped a forged-model dir
        (out / "model.safetensors").write_bytes(b"fake")
        return out


class _FakePublisher:
    """Records calls; returns a fake HF repo URL."""
    def __init__(self, raise_on_publish=False):
        self.calls = []
        self.raise_on_publish = raise_on_publish

    def __call__(self, output_dir, org="continuum-ai", **kwargs):
        self.calls.append(str(output_dir))
        if self.raise_on_publish:
            raise RuntimeError("hf push boom")
        return f"https://huggingface.co/{org}/{Path(output_dir).parent.name}"


def test_worker_process_one_runs_executor_then_publisher_then_marks_finished(tmp_path):
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    a = _write_alloy(tmp_path / "alloy_a.alloy.json")
    q.enqueue(a)

    executor = _FakeExecutor()
    publisher = _FakePublisher()
    w = FactoryWorker(q, executor=executor, publisher=publisher, work_root=tmp_path / "work")

    processed = w.process_one()
    assert processed is True
    assert len(executor.calls) == 1
    assert len(publisher.calls) == 1
    assert q.stats() == {"intake": 0, "assembly": 0, "finished": 1, "rework": 0}


def test_worker_process_one_returns_false_when_intake_empty(tmp_path):
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    w = FactoryWorker(q, executor=_FakeExecutor(), publisher=_FakePublisher(), work_root=tmp_path / "work")
    assert w.process_one() is False


def test_worker_marks_rework_on_executor_exception(tmp_path):
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    a = _write_alloy(tmp_path / "alloy_a.alloy.json")
    q.enqueue(a)

    executor = _FakeExecutor(raise_on={"alloy_a.alloy"})
    publisher = _FakePublisher()
    w = FactoryWorker(q, executor=executor, publisher=publisher, work_root=tmp_path / "work")

    processed = w.process_one()
    assert processed is True  # processed (and failed)
    assert q.stats() == {"intake": 0, "assembly": 0, "finished": 0, "rework": 1}
    assert publisher.calls == []  # publisher never invoked on forge failure


def test_worker_marks_rework_on_publisher_exception(tmp_path):
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    a = _write_alloy(tmp_path / "alloy_a.alloy.json")
    q.enqueue(a)

    executor = _FakeExecutor()
    publisher = _FakePublisher(raise_on_publish=True)
    w = FactoryWorker(q, executor=executor, publisher=publisher, work_root=tmp_path / "work")

    w.process_one()
    assert q.stats()["rework"] == 1
    assert q.stats()["finished"] == 0


def test_worker_run_loop_drains_intake(tmp_path):
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    for i in range(3):
        q.enqueue(_write_alloy(tmp_path / f"a{i}.alloy.json", {**SYNTHETIC_ALLOY, "name": f"a{i}"}))

    w = FactoryWorker(
        q, executor=_FakeExecutor(), publisher=_FakePublisher(),
        work_root=tmp_path / "work",
    )
    processed_count = w.run_loop(max_iters=10)
    assert processed_count == 3
    assert q.stats() == {"intake": 0, "assembly": 0, "finished": 3, "rework": 0}


def test_worker_run_loop_max_iters_respected(tmp_path):
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    for i in range(5):
        q.enqueue(_write_alloy(tmp_path / f"a{i}.alloy.json", {**SYNTHETIC_ALLOY, "name": f"a{i}"}))

    w = FactoryWorker(
        q, executor=_FakeExecutor(), publisher=_FakePublisher(),
        work_root=tmp_path / "work",
    )
    processed_count = w.run_loop(max_iters=2)
    assert processed_count == 2
    assert q.stats()["intake"] == 3
    assert q.stats()["finished"] == 2
