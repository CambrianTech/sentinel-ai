"""Tier 1 reproducibility test — every published continuum-ai/* alloy must
resolve to a non-empty adapter chain via scripts/adapters.

This test is the SPEC the family-adapter sprint exists to satisfy. Day-one
expectation is that several published alloys fail this test, because their
architectures or stage types don't have adapters yet. Each red → green flip
is one commit's worth of plugin-sprint work.

Tier 1 = pure dispatch resolution. No model load, no torch import, no GPU.
This test is safe to run on any machine. The Tier 2 test (actual byte-
identical re-forge against the published modelHash) lives separately and
runs only on hardware with the right GPU profile.

To run this test:
    cd sentinel-ai
    python -m pytest tests/reproducibility/test_published_alloys_dispatch.py -v

To run a single alloy:
    python -m pytest tests/reproducibility/test_published_alloys_dispatch.py \\
        -v -k "qwen3_5_9b_general_forged"

Network: this test fetches alloys from HuggingFace on first run and caches
them under tests/reproducibility/_cache/. Re-runs use the cache. To force
a re-fetch (e.g. after a published alloy is updated), delete the cache dir.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

import pytest

# Add scripts/ to import path so the test imports the adapter package the
# same way alloy_executor does.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from adapters import resolve_adapter_chain, DispatchError, registered_architectures
from adapters.dispatch import describe_chain


CACHE_DIR = Path(__file__).resolve().parent / "_cache"
HF_API_BASE = "https://huggingface.co/api/models"
HF_RESOLVE_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"


# ── Catalog: every published continuum-ai/* artifact this test must cover ───
#
# Each entry is (repo_id, alloy_filename, status). filename=None means "ask
# the HF API to find the alloy file" — used when the filename varies across
# artifacts. status is one of:
#
#   "active"        — must pass; failure is a real adapter gap to close
#   "no-alloy-file" — repo has no .alloy.json (publish-pipeline gap; brand
#                     integrity issue tracked separately, not a dispatch
#                     failure). Skipped with a clear message.
#   "deferred"      — adapter for this architecture is on the queue but not
#                     yet built per the "qwen3.5 first" instruction. xfail,
#                     so adding the adapter automatically flips it to passing.
#
# This list grows as new artifacts are published. The test parametrizes over
# the entire list — adding a new entry automatically creates a new test case.

PUBLISHED_ALLOYS: list[tuple[str, str | None, str]] = [
    # ── Qwen3.5 dense catalog (FIRST priority — fully green is the gate) ─────
    ("continuum-ai/qwen3.5-0.8b-general-forged",            None, "active"),
    ("continuum-ai/qwen3.5-2b-general-forged",              None, "active"),
    ("continuum-ai/qwen3.5-4b-general-forged",              None, "active"),
    ("continuum-ai/qwen3.5-4b-code-forged",                 None, "active"),
    ("continuum-ai/qwen3.5-4b-code-128k-forged",            None, "active"),
    ("continuum-ai/qwen3.5-9b-general-forged",              None, "active"),

    # Variants of parent forges that the publish pipeline failed to write
    # an alloy.json for. These are real artifacts on HF but they have no
    # provenance file. Tracked as a separate brand-integrity gap — the fix
    # is in scripts/publish_model.py / scripts/alloy_to_card.py, not in the
    # adapter dispatch layer. Skipped here with a clear message.
    ("continuum-ai/qwen3.5-4b-code-forged-defragged",       None, "no-alloy-file"),
    ("continuum-ai/qwen3.5-4b-code-forged-GGUF",            None, "no-alloy-file"),
    ("continuum-ai/qwen3.5-27b-code-forged",                None, "no-alloy-file"),
    ("continuum-ai/qwen3.5-27b-code-forged-defragged",      None, "no-alloy-file"),
    ("continuum-ai/qwen3.5-27b-code-forged-mlx-4bit",       None, "no-alloy-file"),

    # ── MoE §4.1.3.4 anchor artifacts ────────────────────────────────────────
    # qwen3_moe adapter landed — this one must dispatch cleanly.
    ("continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k", None, "active"),
    # olmoe adapter landed — second §4.1.3.4 cross-architecture anchor.
    ("continuum-ai/olmoe-1b-7b-compacted-5b",               None, "active"),
    # Dense compensated v2-7B — the §4.1.3.3 anchor (qwen2 architecture).
    ("continuum-ai/qwen2.5-coder-7b-compacted",             None, "deferred"),
]


def _safe_id(repo: str) -> str:
    """Turn a repo id into a pytest-friendly test id."""
    return repo.replace("/", "_").replace("-", "_").replace(".", "_")


def _http_json(url: str) -> dict:
    """Tiny urllib JSON fetcher — avoids requiring `requests` for the test."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e


def _http_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e


def _discover_alloy_filename(repo: str) -> str:
    """Ask the HF API which file in the repo is the alloy JSON."""
    meta = _http_json(f"{HF_API_BASE}/{repo}")
    candidates = [
        s["rfilename"]
        for s in meta.get("siblings", [])
        if s["rfilename"].endswith(".json")
        and ("alloy" in s["rfilename"].lower() or s["rfilename"] == "forge-alloy.json")
    ]
    if not candidates:
        raise RuntimeError(
            f"No alloy JSON found in {repo}. Files: "
            f"{[s['rfilename'] for s in meta.get('siblings', [])]}"
        )
    # Prefer a filename that contains the repo's slug (most specific) over
    # the generic 'forge-alloy.json' fallback used by some legacy artifacts.
    slug = repo.split("/")[-1]
    specific = [c for c in candidates if slug in c]
    return specific[0] if specific else candidates[0]


def _fetch_alloy(repo: str, filename: str | None) -> dict:
    """Fetch and cache an alloy. Cache key is repo+filename so updates land."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = _discover_alloy_filename(repo)
    cache_path = CACHE_DIR / f"{repo.replace('/', '_')}__{filename.replace('/', '_')}"
    if not cache_path.exists():
        url = HF_RESOLVE_BASE.format(repo=repo, filename=filename)
        cache_path.write_bytes(_http_bytes(url))
    return json.loads(cache_path.read_text())


# ── Catalog inventory test — confirms the test fixture matches reality ──────


def test_catalog_is_non_empty():
    """Sanity: the test catalog must contain at least the 11 Qwen3.5 entries
    plus the 3 §4.1.3.4-era anchors."""
    assert len(PUBLISHED_ALLOYS) >= 14, (
        f"PUBLISHED_ALLOYS has {len(PUBLISHED_ALLOYS)} entries — expected at least 14 "
        f"(11 Qwen3.5 dense catalog + 2 MoE anchors + 1 dense compensated)"
    )


def test_qwen3_dense_adapter_is_registered():
    """Sanity: the qwen3_5 architecture must resolve. If this fails the
    adapter package didn't import its concrete adapters."""
    assert "qwen3_5" in registered_architectures(), (
        f"qwen3_5 not registered. Registered architectures: {registered_architectures()}. "
        f"Check that scripts/adapters/__init__.py imports the qwen3_dense module."
    )


# ── The actual reproducibility-Tier-1 dispatch tests ────────────────────────


@pytest.mark.parametrize(
    "repo,filename,status",
    PUBLISHED_ALLOYS,
    ids=[_safe_id(r) for r, _, _ in PUBLISHED_ALLOYS],
)
def test_published_alloy_resolves_to_adapter_chain(repo: str, filename: str | None, status: str):
    """Every published alloy must produce a non-empty adapter chain.

    Day-one expectation: many of these fail because their family adapter
    or stage handlers don't exist yet. Each green flip is one commit of
    plugin-sprint work.
    """
    if status == "no-alloy-file":
        pytest.skip(
            f"{repo} has no .alloy.json published in its HF repo. This is a "
            f"publish-pipeline gap (alloy_to_card.py / publish_model.py should "
            f"emit one for every artifact, including downstream variants). "
            f"Tracked as a brand-integrity gap separate from the dispatch layer."
        )
    if status == "deferred":
        pytest.xfail(
            f"{repo} architecture is not yet supported by an adapter. Per the "
            f"'qwen3.5 first' instruction, the deferred-family adapters land "
            f"AFTER the Qwen3.5 dense catalog is fully green. Adding the adapter "
            f"will automatically flip this xfail to xpass."
        )

    alloy = _fetch_alloy(repo, filename)
    name = alloy.get("name", repo)
    architecture = (alloy.get("source") or {}).get("architecture")

    try:
        chain = resolve_adapter_chain(alloy)
    except DispatchError as e:
        pytest.fail(
            f"\n  Alloy {name} (architecture={architecture!r}) failed to dispatch:\n"
            f"  {e}\n"
            f"  This is the gap the family-adapter sprint must close. Add the missing\n"
            f"  adapter or stage handler, then re-run."
        )

    assert chain, f"Alloy {name} resolved but produced an empty chain"

    # For stages that REQUIRE family-specific overrides (REQUIRES_FAMILY_OVERRIDE
    # on the base class), the chain element must reference a method that the
    # concrete family adapter has actually overridden — not the base-class
    # NotImplementedError stub. Output / bookend stages (quant, eval, publish,
    # package, deploy, deliver) are family-agnostic by default and are exempt.
    from adapters.base import FamilyAdapter as _BaseFamilyAdapter

    family = chain[0].family_adapter
    family_class = type(family)
    for call in chain:
        if call.method_name not in _BaseFamilyAdapter.REQUIRES_FAMILY_OVERRIDE:
            continue
        bound = getattr(family_class, call.method_name, None)
        on_base = getattr(_BaseFamilyAdapter, call.method_name, None)
        if bound is None or bound is on_base:
            pytest.fail(
                f"\n  Alloy {name} stage[{call.stage_index}] type={call.stage_type!r} "
                f"resolved to {family_class.__name__}.{call.method_name} BUT that method "
                f"is still inherited from FamilyAdapter (raises NotImplementedError).\n"
                f"  Override {call.method_name}() on {family_class.__name__} to handle this stage.\n"
                f"  Resolved chain so far:\n{describe_chain(chain)}"
            )
