"""Tier 3 reproducibility test — verify the per-problem JSONL samples
published alongside each continuum-ai/* artifact match the sha256 hashes
recorded in the alloy.

This is the cheapest possible falsifiability check on the morning's MoE
breakthrough and every other shipped artifact: no GPU, no model load, no
inference, no torch import. Just download → sha256 → compare.

Why it works: the publish pipeline (alloy_to_card.py / publish_model.py)
records sha256 hashes of the per-problem JSONL evaluation outputs in the
alloy's results.benchmarks[].resultHash + .baseResultHash fields. Anyone
with the alloy can independently verify those bytes against the producer's
HuggingFace upload — meaning the producer cannot silently swap the published
samples without breaking this test. That's the cryptographic chain-of-
custody promise for the eval results, paralleling integrity.modelHash for
the model weights.

Tiers — quick reference:
    Tier 1: dispatch resolution           tests/reproducibility/test_published_alloys_dispatch.py
    Tier 2: byte-equivalent re-forge      (NOT YET BUILT, requires 5090)
    Tier 3: sample-hash verification      THIS FILE  (runs on Mac, no GPU)
    Tier 4: re-score samples → pass@1     (TODO, runs on Mac, requires `evalplus`)

Tier 4 (eval re-scoring) is the natural follow-up that lights up after
Tier 3 — once we trust the sample bytes, re-running the eval scorer
against them produces the published pass@1 numbers without ever invoking
the model. That validates the published BENCHMARK SCORES, not just the
sample bytes.

Brand-integrity gaps this test surfaces:
    - The alloy's priorMetricBaselines[] cell publishes the negative-
      baseline samples file (e.g. student_samples_router_l2_baseline.jsonl)
      but does NOT pin its sha256 hash. Until alloy_to_card.py adds the
      hash, the falsifiability anchor for the §4.1.3.4 negative baseline
      can be silently swapped. Test marks this as a known gap (xfail) so
      adding the hash auto-flips it to pass.
    - The §4.1.3.4.1 calibration corpus is referenced by path in the alloy
      but is NOT uploaded to the HF repo. Same fix-layer concern.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
SAMPLES_CACHE = CACHE_DIR / "samples"


class SampleHashCase(NamedTuple):
    """One (repo, samples-file, expected-sha256, role) tuple to verify."""
    repo: str
    samples_path: str         # path inside the HF repo, e.g. "eval/humaneval/student_samples.jsonl"
    expected_sha256: str      # the alloy's claimed hash, with or without "sha256:" prefix
    role: str                 # human label: "student", "base", "negative-baseline-router-l2", ...
    benchmark_names: tuple[str, ...] = ()  # which benchmark(s) this samples file scores

    @property
    def normalized_hash(self) -> str:
        """Strip optional 'sha256:' prefix and lowercase."""
        h = self.expected_sha256
        if h.startswith("sha256:"):
            h = h[len("sha256:"):]
        return h.lower()


def _http_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e


def _fetch_samples(repo: str, samples_path: str) -> bytes:
    """Fetch and cache a samples JSONL by repo + in-repo path."""
    SAMPLES_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = SAMPLES_CACHE / f"{repo.replace('/', '_')}__{samples_path.replace('/', '_')}"
    if not cache_path.exists():
        url = f"https://huggingface.co/{repo}/resolve/main/{samples_path}"
        cache_path.write_bytes(_http_bytes(url))
    return cache_path.read_bytes()


def _load_alloy(filename: str) -> dict:
    """Load a cached alloy by its cache filename."""
    p = CACHE_DIR / filename
    if not p.exists():
        raise FileNotFoundError(
            f"Alloy not in cache: {p}. Run the dispatch test first to populate the cache."
        )
    return json.loads(p.read_text())


def _build_cases() -> list[SampleHashCase]:
    """Walk every cached alloy, extract every (samplesPath, resultHash) pair
    from results.benchmarks[]. The dispatch test populates _cache/ on first
    run; this list is built dynamically so adding a new published alloy via
    the dispatch test catalog automatically extends Tier 3 coverage too."""
    cases: list[SampleHashCase] = []
    if not CACHE_DIR.exists():
        return cases
    for cache_file in sorted(CACHE_DIR.glob("continuum-ai_*.json")):
        try:
            alloy = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            continue
        # Repo id reverse-engineered from the cache filename:
        # "continuum-ai_<slug>__<filename>.json" → "continuum-ai/<slug>"
        stem = cache_file.name.split("__", 1)[0]
        repo = stem.replace("_", "/", 1)
        results = alloy.get("results") or {}
        # Group benchmarks by samples file so we test each samples file once,
        # not once per benchmark scored against it.
        student_groups: dict[tuple[str, str], list[str]] = {}
        base_groups: dict[tuple[str, str], list[str]] = {}
        for b in results.get("benchmarks", []):
            sp = b.get("samplesPath")
            rh = b.get("resultHash")
            bsp = b.get("baseSamplesPath")
            brh = b.get("baseResultHash")
            if sp and rh:
                student_groups.setdefault((sp, rh), []).append(b.get("name", "?"))
            if bsp and brh:
                base_groups.setdefault((bsp, brh), []).append(b.get("name", "?"))
        for (sp, rh), bench_names in student_groups.items():
            cases.append(SampleHashCase(
                repo=repo, samples_path=sp, expected_sha256=rh,
                role="student", benchmark_names=tuple(bench_names),
            ))
        for (sp, rh), bench_names in base_groups.items():
            cases.append(SampleHashCase(
                repo=repo, samples_path=sp, expected_sha256=rh,
                role="base", benchmark_names=tuple(bench_names),
            ))
    return cases


def _build_negative_baseline_cases() -> list[SampleHashCase]:
    """Same idea but walks priorMetricBaselines[] instead of results.benchmarks.
    These cells publish the falsifiability anchor samples (e.g. the
    §4.1.3.4 negative-baseline router-gate-L2 cell) — they have a samplesPath
    but the alloys today do NOT pin a samplesHash on them. This test
    parametrizes over them anyway and xfails them with a clear message
    until the alloy schema gains a samplesHash field for prior baselines."""
    cases: list[SampleHashCase] = []
    if not CACHE_DIR.exists():
        return cases
    for cache_file in sorted(CACHE_DIR.glob("continuum-ai_*.json")):
        try:
            alloy = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            continue
        stem = cache_file.name.split("__", 1)[0]
        repo = stem.replace("_", "/", 1)
        for pmb in alloy.get("priorMetricBaselines", []):
            ev = pmb.get("evaluation") or {}
            sp = ev.get("samplesPath")
            sh = ev.get("samplesHash") or ev.get("resultHash")
            if not sp:
                continue
            cases.append(SampleHashCase(
                repo=repo,
                samples_path=sp,
                expected_sha256=sh or "",   # empty → xfail-with-message
                role=f"prior-baseline:{pmb.get('id', '?')}",
            ))
    return cases


# Build the case lists at import time so pytest can parametrize over them.
_FORWARD_CASES = _build_cases()
_PRIOR_BASELINE_CASES = _build_negative_baseline_cases()


# Sanity test — must collect SOMETHING, otherwise the dispatch test cache
# wasn't populated yet and Tier 3 has nothing to verify.
def test_cache_has_alloys():
    """The dispatch test must run first to populate _cache/. If this fails,
    run: python -m pytest tests/reproducibility/test_published_alloys_dispatch.py"""
    assert CACHE_DIR.exists() and any(CACHE_DIR.glob("continuum-ai_*.json")), (
        "No cached alloys found. Run the dispatch test first to populate "
        f"{CACHE_DIR}, or check that the dispatch test catalog is non-empty."
    )


def test_cases_were_extracted():
    """Sanity: at least one forward sample-hash case must have been extracted
    from the cached alloys. If this fails, the alloys probably don't have
    resultHash fields populated — check the publish pipeline."""
    assert len(_FORWARD_CASES) > 0, (
        "Zero forward sample-hash cases extracted from cached alloys. "
        "Either no alloys are cached yet, or none of them have benchmarks "
        "with both samplesPath and resultHash populated. Check that "
        "alloy_to_card.py is writing both fields when an artifact has "
        "uploaded eval samples."
    )


# ── Forward verification: every (alloy, benchmark, sample file) ─────────────


@pytest.mark.parametrize(
    "case",
    _FORWARD_CASES,
    ids=[f"{c.repo.split('/')[-1]}__{c.role}__{Path(c.samples_path).name}" for c in _FORWARD_CASES],
)
def test_published_samples_match_alloy_hash(case: SampleHashCase):
    """The bytes uploaded to HF must hash to exactly what the alloy claims.

    This is the cryptographic chain-of-custody check for the eval results.
    If the producer (or anyone with write access to the HF repo) silently
    edited the per-problem JSONL after publish, this test catches it.
    """
    data = _fetch_samples(case.repo, case.samples_path)
    actual = hashlib.sha256(data).hexdigest()
    expected = case.normalized_hash
    n_lines = data.count(b"\n")
    assert actual == expected, (
        f"\n  Sample file hash mismatch — chain of custody broken!"
        f"\n  Repo:        {case.repo}"
        f"\n  File:        {case.samples_path}"
        f"\n  Role:        {case.role}"
        f"\n  Benchmarks:  {', '.join(case.benchmark_names) or '?'}"
        f"\n  Expected:    sha256:{expected}  (from alloy results.benchmarks[].resultHash)"
        f"\n  Actual:      sha256:{actual}    ({len(data)} bytes, {n_lines} lines)"
        f"\n  This means the JSONL on HuggingFace differs from what the alloy"
        f"\n  was published with. Either the alloy is wrong (re-pin) or the"
        f"\n  samples file was modified post-publish (investigate)."
    )


# ── Prior-baseline verification (negative-baseline anchors) ─────────────────


@pytest.mark.parametrize(
    "case",
    _PRIOR_BASELINE_CASES,
    ids=[f"{c.repo.split('/')[-1]}__{c.role}" for c in _PRIOR_BASELINE_CASES] or ["empty"],
)
def test_prior_baseline_samples_pinned_and_match(case: SampleHashCase):
    """Negative-baseline cells (priorMetricBaselines[]) must pin their
    samples-hash for falsifiability. Today they don't — this test xfails
    with a clear message until alloy_to_card.py adds samplesHash to the
    schema and the publish pipeline pins it.

    Once the gap is closed, the test auto-flips to passing.
    """
    if not _PRIOR_BASELINE_CASES:
        pytest.skip("No priorMetricBaselines in any cached alloy")
    if not case.expected_sha256:
        pytest.xfail(
            f"\n  {case.repo} priorMetricBaselines[{case.role}] publishes\n"
            f"  samplesPath={case.samples_path!r} but does NOT pin a samplesHash.\n"
            f"  This is a §4.1.3.4 falsifiability gap — anyone with HF write\n"
            f"  access could silently swap the negative-baseline JSONL and\n"
            f"  this test would not catch it.\n"
            f"  Fix: add samplesHash to the priorMetricBaselines[].evaluation\n"
            f"  schema in forge_alloy/types.py + alloy_to_card.py + publish_model.py.\n"
            f"  Once added, this xfail auto-flips to xpass."
        )
    # If a hash is present (future state), verify it
    data = _fetch_samples(case.repo, case.samples_path)
    actual = hashlib.sha256(data).hexdigest()
    assert actual == case.normalized_hash, (
        f"Prior-baseline sample hash mismatch for {case.repo} {case.role}: "
        f"expected sha256:{case.normalized_hash}, got sha256:{actual}"
    )
