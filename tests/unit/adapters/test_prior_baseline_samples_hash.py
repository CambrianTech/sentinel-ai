"""TDD spec for the priorMetricBaselines samplesHash schema field.

Roadmap step 8 from docs/PLUGIN-SPRINT.md: the §4.1.3.4 falsifiability
anchors (priorMetricBaselines[]) publish their samples files but the
alloy schema today has no samplesHash field on the evaluation block,
so the negative-baseline JSONLs are NOT cryptographically pinned. This
is a brand-integrity gap — anyone with HF write access could swap the
negative-baseline JSONL and the published Δ wouldn't be byte-verifiable.

The two affected cells:
    qwen3-coder-30b-a3b-compacted-19b-256k priorMetricBaselines[router-gate-l2-norm]
    olmoe-1b-7b-compacted-5b              priorMetricBaselines[broad-corpus]

Both surfaced as xfails on tests/reproducibility/test_published_alloys_sample_hashes.py
in earlier commits. This step closes the gap by:

  1. Adding samplesHash to the schema (forge-alloy side, lands as a
     parallel commit on the forge-alloy repo + a follow-up to the
     existing forge_alloy.types module). For the local cache + the
     sentinel-ai test layer, the migration script populates the field
     directly from HF's per-file content sha256.
  2. Migrating the cached alloys to populate samplesHash on every
     priorMetricBaselines cell that has a samplesPath.
  3. Flipping the 2 xfails to PASS by computing samplesHash via
     hashlib.sha256 on the published JSONL bytes (Tier 3 already
     downloads them) and asserting equality.

Written test-first per TDD/TDValidation discipline.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CACHE_DIR = REPO_ROOT / "tests" / "reproducibility" / "_cache"
SAMPLES_CACHE = CACHE_DIR / "samples"


# ── Migration tool ──────────────────────────────────────────────────────────


def test_prior_baseline_migration_script_exists():
    """A migration tool walks every cached alloy and populates
    integrity-style samplesHash on every priorMetricBaselines cell that
    has a samplesPath but no samplesHash. Idempotent."""
    migration = REPO_ROOT / "scripts" / "migrate_prior_baseline_samples_hash.py"
    assert migration.exists(), (
        "scripts/migrate_prior_baseline_samples_hash.py must exist as the "
        "one-shot tool for populating priorMetricBaselines[].evaluation.samplesHash."
    )


# ── Every priorMetricBaseline cell with samplesPath has samplesHash ─────────


def _cached_alloys():
    out = []
    for cf in sorted(CACHE_DIR.glob("continuum-ai_*.json")):
        try:
            data = json.loads(cf.read_text())
        except json.JSONDecodeError:
            continue
        out.append((cf.name, data))
    return out


def test_every_prior_baseline_with_samples_has_hash():
    """For every priorMetricBaseline cell that publishes samplesPath,
    the cell MUST also carry samplesHash. Loud failure here means the
    falsifiability anchor is unpinned and the methodology paper's
    §4.1.3.4 claim is byte-unverifiable."""
    failures = []
    for name, alloy in _cached_alloys():
        for pmb in alloy.get("priorMetricBaselines", []):
            ev = pmb.get("evaluation") or {}
            sp = ev.get("samplesPath")
            sh = ev.get("samplesHash")
            if sp and not sh:
                failures.append(f"{name}: {pmb.get('id', '?')} has samplesPath but no samplesHash")
    assert not failures, (
        f"{len(failures)} prior-baseline cells are unpinned:\n  "
        + "\n  ".join(failures)
        + "\n\nRun the migration: python scripts/migrate_prior_baseline_samples_hash.py --confirm"
    )


def test_prior_baseline_hash_matches_published_bytes():
    """For every priorMetricBaseline cell with samplesHash, the hash MUST
    equal sha256 of the bytes of the published samples file. The
    samples file is downloaded under tests/reproducibility/_cache/samples/
    by Tier 3; this test reuses that cache (or fetches if missing)."""
    failures = []
    for name, alloy in _cached_alloys():
        repo_stem = name.split("__", 1)[0]
        repo = repo_stem.replace("_", "/", 1)
        for pmb in alloy.get("priorMetricBaselines", []):
            ev = pmb.get("evaluation") or {}
            sp = ev.get("samplesPath")
            sh = ev.get("samplesHash")
            if not (sp and sh):
                continue
            expected = sh.replace("sha256:", "").lower()
            local = SAMPLES_CACHE / f"{repo.replace('/', '_')}__{sp.replace('/', '_')}"
            if not local.exists():
                # Fetch on demand — same path Tier 3 uses
                local.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/{repo}/resolve/main/{sp}"
                with urllib.request.urlopen(url, timeout=60) as resp:
                    local.write_bytes(resp.read())
            actual = hashlib.sha256(local.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(
                    f"{name} prior-baseline {pmb.get('id', '?')}: "
                    f"recorded sha256:{expected[:16]}... vs actual sha256:{actual[:16]}..."
                )
    assert not failures, "\n".join(failures)


# ── Tier 3 xfails should now be passing ─────────────────────────────────────


def test_tier3_unpinned_xfails_are_resolved():
    """The Tier 3 reproducibility test for prior-baseline hash matching
    had 2 xfails before Step 8 (the unpinned negative-baseline cells).
    After Step 8 they MUST flip to PASS. This is the architectural
    gate that proves the schema gap is closed end-to-end.

    We don't run the Tier 3 test here directly (it's slow + lives in
    a sibling module). Instead, we verify the cached alloys carry
    samplesHash on EVERY prior-baseline cell with a samplesPath.
    The Tier 3 test running in the full suite will then exercise the
    same hash check via its own assertion path.
    """
    pinned_count = 0
    for _name, alloy in _cached_alloys():
        for pmb in alloy.get("priorMetricBaselines", []):
            ev = pmb.get("evaluation") or {}
            if ev.get("samplesPath") and ev.get("samplesHash"):
                pinned_count += 1
    assert pinned_count >= 2, (
        f"expected at least 2 prior-baseline cells to be pinned post-Step-8 "
        f"(qwen3-coder-30b-a3b router-gate-l2-norm + olmoe broad-corpus); "
        f"found only {pinned_count} pinned cells"
    )
