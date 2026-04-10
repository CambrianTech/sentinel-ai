"""Pinned-dataset HumanEval / HumanEval+ pass@1 scorer.

Wraps evalplus's official `python -m evalplus.evaluate` so it runs cleanly
on macOS. evalplus's `reliability_guard` calls `resource.setrlimit(RLIMIT_AS, ...)`
which fails on macOS with "current limit exceeds maximum limit", and
because evalplus uses 'spawn' multiprocessing on macOS by default, a
runtime monkey-patch in the parent process doesn't reach the child workers
that run candidates. Result on stock macOS: every solution is killed at
startup and pass@1 is uniformly 0.000 — false-negative on every JSONL.

The fix is two-part and has to land in a CLEAN child process (so any
already-loaded evalplus modules from the parent test runner don't leak in
with the wrong reliability_guard binding):

  1. Spawn a fresh `python` subprocess.
  2. Inject a tiny preamble via `-c` that sets multiprocessing start method
     to 'fork', monkey-patches `reliability_guard` to a no-op on both
     `evalplus.eval` and `evalplus.eval.utils`, then invokes evalplus's
     CLI main() with the right argv.

Forked children inherit the parent's monkey-patch; the resource.setrlimit
call never happens in the worker; per-problem solutions execute normally;
evalplus reports the canonical pass@1 — same as Linux, same as the publish
pipeline.

Earlier history (preserved for posterity): an earlier version of this
module implemented a hand-rolled scorer that exec'd the dataset's `test`
field inline. It matched evalplus on most JSONLs to ±0.05 pp but disagreed
on one problem in the OLMoE broad-corpus JSONL because evalplus's official
scorer applies `_special_oracle` and `contract` handling that the hand-
rolled path didn't replicate. The right answer was to fix the wrapping,
not write a parallel scorer. This module is the fix.

Usage:
    from _humaneval_scorer import score_jsonl
    result = score_jsonl("path/to/samples.jsonl")
    print(result["humaneval"]["pass_at_1"], result["humaneval_plus"]["pass_at_1"])
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


# Subprocess preamble. Runs in a fresh Python interpreter that has not yet
# imported any evalplus module — so the patch lands BEFORE evalplus's
# multiprocessing pool snapshots the function reference, and the forked
# workers inherit the no-op binding cleanly.
_PREAMBLE = """
import multiprocessing as _mp
_mp.set_start_method("fork", force=True)
import evalplus.eval.utils as _u
import evalplus.eval as _e
_noop = lambda *a, **k: None
_u.reliability_guard = _noop
_e.reliability_guard = _noop
import sys as _sys
_sys.argv = {argv!r}
from evalplus.evaluate import main as _main
try:
    _main()
except SystemExit:
    pass
"""


def _evalplus_dataset_hash() -> str:
    """Return the sha256 of the pinned HumanEval+ dataset evalplus loads.
    Imported here so the test code can capture it for the reproducibility
    report and so changes to the dataset (which shouldn't happen — it's
    pinned) are visible immediately."""
    from evalplus.data import get_human_eval_plus_hash
    return get_human_eval_plus_hash()


def score_jsonl(samples_path: str | Path) -> dict[str, Any]:
    """Score a JSONL of {"task_id", "solution"} samples against the pinned
    HumanEval+ dataset.

    Returns:
        {
            "humaneval":      {"pass_at_1": float, "passed": int, "total": int},
            "humaneval_plus": {"pass_at_1": float, "passed": int, "total": int},
            "dataset_hash":   "<sha256 of the pinned dataset>",
            "raw_output":     "<full evalplus stdout for diagnostics>",
        }

    pass_at_1 values are 0..1 fractions matching evalplus's CLI output
    convention. Multiply by 100 for the percentage form the alloys publish.
    """
    samples_path = Path(samples_path).resolve()
    if not samples_path.exists():
        raise FileNotFoundError(f"samples file not found: {samples_path}")

    # Always delete any stale evalplus output file from a previous run so
    # we never accidentally read a cached false-negative. evalplus writes
    # the file alongside the input as <stem>_eval_results.json.
    details_path = samples_path.with_name(samples_path.stem + "_eval_results.json")
    if details_path.exists():
        details_path.unlink()

    argv = [
        "evalplus.evaluate",
        "--dataset", "humaneval",
        "--samples", str(samples_path),
        "--parallel", str(min(os.cpu_count() or 4, 8)),
    ]
    preamble = _PREAMBLE.format(argv=argv)

    proc = subprocess.run(
        [sys.executable, "-c", preamble],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"evalplus subprocess failed (rc={proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    output = proc.stdout

    # Parse the canonical evalplus output:
    #   humaneval (base tests)
    #   pass@1:	0.622
    #   humaneval+ (base + extra tests)
    #   pass@1:	0.537
    base_match = re.search(
        r"humaneval \(base tests\)\s*\n?\s*pass@1:\s*(\d+\.\d+)",
        output,
    )
    plus_match = re.search(
        r"humaneval\+ \(base \+ extra tests\)\s*\n?\s*pass@1:\s*(\d+\.\d+)",
        output,
    )
    if not base_match or not plus_match:
        raise RuntimeError(
            f"Could not parse pass@1 from evalplus output. Raw output:\n{output}"
        )
    base_pass1 = float(base_match.group(1))
    plus_pass1 = float(plus_match.group(1))

    # Read the per-problem details to extract exact pass/total counts.
    base_passed = base_total = plus_passed = plus_total = None
    if details_path.exists():
        details = json.loads(details_path.read_text())
        eval_results = details.get("eval", {})
        # Each task maps to a list of attempts (one per sample). For pass@1
        # the convention is "task passes if ANY attempt passes." We follow
        # evalplus's own convention for the count.
        base_passed = sum(
            1 for attempts in eval_results.values()
            if any(a.get("base_status") == "pass" for a in attempts)
        )
        plus_passed = sum(
            1 for attempts in eval_results.values()
            if any(a.get("plus_status") == "pass" for a in attempts)
        )
        base_total = plus_total = len(eval_results)

    return {
        "humaneval": {
            "pass_at_1": base_pass1,
            "passed": base_passed,
            "total": base_total,
        },
        "humaneval_plus": {
            "pass_at_1": plus_pass1,
            "passed": plus_passed,
            "total": plus_total,
        },
        "dataset_hash": _evalplus_dataset_hash(),
        "raw_output": output,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("samples", help="Path to a HumanEval JSONL")
    args = p.parse_args()
    result = score_jsonl(args.samples)
    he = result["humaneval"]
    hep = result["humaneval_plus"]
    print(f"HumanEval base    pass@1: {he['pass_at_1']:.4f}  ({he['passed']}/{he['total']})")
    print(f"HumanEval+ strict pass@1: {hep['pass_at_1']:.4f}  ({hep['passed']}/{hep['total']})")
    print(f"Dataset hash:             {result['dataset_hash']}")
