"""TDD spec: Tier 2 modelHash recording in finished/ result manifest.

When the daemon marks a part finished, it should ALSO compute the
modelHash of the forged_dir using the canonical alloy_hashing convention
and stash it in the result manifest. This gives continuum (the shipping
department) the chain-of-custody data it needs to:

  1. Verify the on-disk artifact matches what the publish stage will push
  2. Compare against the previous run's modelHash if the same alloy is
     re-forged later (deterministic forge verification)
  3. Stamp the modelHash into the published model card automatically

Without this, continuum would have to rehash forged_dir itself before
publishing — duplicate work and a window where the artifact could
silently change.

Forge-time recording is the simple half. The full Tier 2 reproducibility
gate (assert produced hash matches declared hash) lives in continuum's
shipping flow because that's where the comparison policy is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


SYNTHETIC_ALLOY = {
    "name": "test-tier2",
    "source": {"baseModel": "Test/Base", "architecture": "qwen3_moe"},
    "stages": [],
}


def _write_alloy(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SYNTHETIC_ALLOY))
    return path


class _FakeExecutorWithSafetensors:
    """Drops fake safetensors shards into the output dir so the
    modelHash recorder has something to walk."""

    def __init__(self):
        self.calls = []

    def __call__(self, alloy_path, output_dir=None, dry_run=False):
        self.calls.append(str(alloy_path))
        out = Path(output_dir or (Path(alloy_path).parent / "out"))
        out.mkdir(parents=True, exist_ok=True)
        # Two synthetic shards with deterministic content for hash testing
        (out / "model-00001-of-00002.safetensors").write_bytes(b"shard one bytes")
        (out / "model-00002-of-00002.safetensors").write_bytes(b"shard two bytes")
        return out


def test_alloy_hashing_helpers_importable():
    from alloy_hashing import compose_model_hash, hash_local_safetensors_dir
    assert callable(compose_model_hash)
    assert callable(hash_local_safetensors_dir)


def test_finished_manifest_records_modelhash(tmp_path):
    """When the daemon successfully forges a part, the result manifest
    must carry a modelHash field computed from forged_dir."""
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    _write_alloy(q.intake_dir / "test-tier2.alloy.json")

    w = FactoryWorker(
        q,
        executor=_FakeExecutorWithSafetensors(),
        work_root=tmp_path / "work",
    )
    w.process_one()

    finished = list(q.finished_dir.glob("*.alloy.json"))
    assert len(finished) == 1
    sidecar = finished[0].with_suffix(".result.json")
    assert sidecar.exists()
    manifest = json.loads(sidecar.read_text())

    # The Tier 2 contribution: modelHash is recorded
    assert "modelHash" in manifest
    assert manifest["modelHash"].startswith("sha256:")
    # And the per-shard hashes are preserved for chain of custody
    assert "fileHashes" in manifest
    assert len(manifest["fileHashes"]) == 2
    # File hashes should be sorted by filename for canonical reproducibility
    names = [f["filename"] for f in manifest["fileHashes"]]
    assert names == sorted(names)


def test_recorded_modelhash_is_deterministic(tmp_path):
    """Forging the SAME alloy twice (with the same fake executor that
    produces the same bytes) MUST produce the same modelHash. This is
    the determinism the Tier 2 gate relies on."""
    from factory_queue import FactoryQueue, FactoryWorker
    import shutil

    # First forge
    q1 = FactoryQueue(tmp_path / "run1")
    _write_alloy(q1.intake_dir / "test.alloy.json")
    w1 = FactoryWorker(
        q1,
        executor=_FakeExecutorWithSafetensors(),
        work_root=tmp_path / "run1" / "work",
    )
    w1.process_one()
    m1 = json.loads(
        next(q1.finished_dir.glob("*.alloy.json")).with_suffix(".result.json").read_text()
    )

    # Second forge — fresh queue, same alloy, same fake executor
    q2 = FactoryQueue(tmp_path / "run2")
    _write_alloy(q2.intake_dir / "test.alloy.json")
    w2 = FactoryWorker(
        q2,
        executor=_FakeExecutorWithSafetensors(),
        work_root=tmp_path / "run2" / "work",
    )
    w2.process_one()
    m2 = json.loads(
        next(q2.finished_dir.glob("*.alloy.json")).with_suffix(".result.json").read_text()
    )

    assert m1["modelHash"] == m2["modelHash"], (
        "modelHash changed between two forges of the same bytes — "
        "the canonical hash convention is non-deterministic"
    )


def test_finished_with_no_safetensors_records_empty_filehashes(tmp_path):
    """If forged_dir has no .safetensors (e.g. a synthetic test or a
    failed-but-finished artifact), the manifest still gets fields —
    just empty. No crash, no missing field."""
    from factory_queue import FactoryQueue, FactoryWorker
    q = FactoryQueue(tmp_path)
    _write_alloy(q.intake_dir / "test.alloy.json")

    def fake_executor(alloy_path, output_dir=None, dry_run=False):
        out = Path(output_dir or (Path(alloy_path).parent / "out"))
        out.mkdir(parents=True, exist_ok=True)
        # Note: no safetensors files
        return out

    w = FactoryWorker(q, executor=fake_executor, work_root=tmp_path / "work")
    w.process_one()

    sidecar = next(q.finished_dir.glob("*.alloy.json")).with_suffix(".result.json")
    manifest = json.loads(sidecar.read_text())
    assert "modelHash" in manifest
    assert manifest.get("fileHashes") == []
