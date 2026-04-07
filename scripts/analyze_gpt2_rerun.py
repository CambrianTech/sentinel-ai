#!/usr/bin/env python3
"""
analyze_gpt2_rerun.py — Compare a fresh gpt2-medium 10-cycle V1 controller run
against the original §4.1 EXPERIENTIAL-PLASTICITY transfer function fit.

Three outcomes per Kash's cycle-9 prediction framework (continuum kash-feedback.md):

  A. Cycle 9 collapses again the same way (~-400% recovery)
     → Real architectural exhaustion. §4 stands. Bug wasn't the cause.
     → Reinforces the paper, transfer function constants may shift slightly.

  B. Cycle 9 collapses but at different magnitude or different cycle
     → Real exhaustion AND bug was distorting where it manifested.
     → Most interesting scientifically. Clean before/after of how a metric bug
       deformed multi-cycle behavior. Becomes its own section in the harness paper.

  C. Cycle 9 doesn't collapse at all — curve continues smoothly
     → The cycle-9 collapse WAS the bug. No underlying exhaustion in this regime.
     → Transfer function gets longer, smoother, more confident data.
     → Strongest paper outcome.

Usage:
    python scripts/analyze_gpt2_rerun.py /tmp/gpt2-rerun.log
"""

import sys
import re
import json
import math
from pathlib import Path

# Original §4.1 cycle-by-cycle recovery ratios from EXPERIENTIAL-PLASTICITY paper
# These were measured on gpt2-medium V1 controller, BEFORE the LoRA-on-pruned-hooks fix
# and BEFORE the L2-norm importance metric was identified as broken.
ORIGINAL_RECOVERY_RATIOS = {
    1: 1.178,
    2: 0.952,
    3: 0.858,
    4: 0.789,
    5: 0.731,
    6: 0.689,
    7: 0.642,
    8: 0.598,
    9: -4.331,  # the catastrophic collapse — is it real or is it the bug?
    10: 2.409,
}


def parse_cycle_results(log_path: Path) -> list[dict]:
    """Extract per-cycle metrics from the experiment log."""
    text = log_path.read_text()
    cycles = []

    # Match patterns from experiment_self_directed.py output
    cycle_pattern = re.compile(
        r"Cycle (\d+):.*?"
        r"Pruned (\d+) heads\. Post-prune ppl: ([\d.]+) \(was ([\d.]+)\).*?"
        r"(?:eval_loss=([\d.]+), ppl=([\d.]+).*?){0,}"
        r"Cycle complete.*?recovery_ratio=([-\d.]+)",
        re.DOTALL,
    )

    # Simpler line-based parsing as fallback
    cycle_num = 0
    pre_ppl = post_ppl = final_ppl = recovery = None

    for line in text.split("\n"):
        m = re.match(r".*Cycle (\d+):", line)
        if m:
            if cycle_num and pre_ppl and post_ppl:
                cycles.append({
                    "cycle": cycle_num,
                    "pre_prune_ppl": pre_ppl,
                    "post_prune_ppl": post_ppl,
                    "final_ppl": final_ppl,
                    "recovery_ratio": recovery,
                })
            cycle_num = int(m.group(1))
            pre_ppl = post_ppl = final_ppl = recovery = None

        m = re.search(r"Post-prune ppl: ([\d.]+) \(was ([\d.]+)\)", line)
        if m:
            post_ppl = float(m.group(1))
            pre_ppl = float(m.group(2))

        m = re.search(r"eval_loss=[\d.]+, ppl=([\d.]+)", line)
        if m:
            final_ppl = float(m.group(1))

        m = re.search(r"recovery_ratio=([-\d.]+)", line)
        if m:
            recovery = float(m.group(1))

    if cycle_num and pre_ppl and post_ppl:
        cycles.append({
            "cycle": cycle_num,
            "pre_prune_ppl": pre_ppl,
            "post_prune_ppl": post_ppl,
            "final_ppl": final_ppl,
            "recovery_ratio": recovery,
        })

    return cycles


def classify_outcome(new_cycles: list[dict]) -> str:
    """Classify the rerun result against Kash's three predictions."""
    cycle9 = next((c for c in new_cycles if c["cycle"] == 9), None)

    if cycle9 is None:
        return "INCOMPLETE: did not reach cycle 9"

    new_recovery_9 = cycle9.get("recovery_ratio")
    original_recovery_9 = ORIGINAL_RECOVERY_RATIOS[9]  # -4.331

    if new_recovery_9 is None:
        return "INCOMPLETE: cycle 9 has no recovery_ratio"

    # Outcome A: cycle 9 collapses similarly (within 50% of original magnitude)
    if new_recovery_9 < -2.0:
        return (
            "OUTCOME A: cycle 9 collapses again (real architectural exhaustion). "
            f"Original={original_recovery_9}, New={new_recovery_9}. "
            "§4 stands. The transfer function captured a real plasticity mode. "
            "Re-fit constants but the shape is real."
        )

    # Outcome B: cycle 9 collapses but differently (magnitude or location)
    if new_recovery_9 < 0:
        return (
            "OUTCOME B: cycle 9 still goes negative, but differently. "
            f"Original={original_recovery_9}, New={new_recovery_9}. "
            "Real exhaustion AND bug was distorting magnitude. "
            "Re-fit transfer function on clean data. Paper gets a new section: "
            "'How a metric bug compounds across multi-cycle plasticity.'"
        )

    # Outcome C: cycle 9 doesn't collapse at all
    return (
        "OUTCOME C: cycle 9 does NOT collapse — curve continues smoothly. "
        f"Original={original_recovery_9}, New={new_recovery_9}. "
        "The cycle-9 collapse WAS the bug. Transfer function gets a longer, "
        "smoother, more confident fit. Strongest possible paper outcome."
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_gpt2_rerun.py <log_path>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        sys.exit(1)

    cycles = parse_cycle_results(log_path)

    print(f"\n{'='*70}")
    print(f"  gpt2-medium 10-cycle V1 re-run analysis")
    print(f"  Source: {log_path}")
    print(f"  Cycles parsed: {len(cycles)}")
    print(f"{'='*70}\n")

    print(f"{'Cycle':>6} {'Pre PPL':>12} {'Post-prune':>12} {'Final PPL':>12} {'Recovery':>12}  {'Original':>12}")
    print("-" * 80)
    for c in cycles:
        n = c["cycle"]
        pre = c.get("pre_prune_ppl") or float("nan")
        post = c.get("post_prune_ppl") or float("nan")
        final = c.get("final_ppl") or float("nan")
        rec = c.get("recovery_ratio") or float("nan")
        orig = ORIGINAL_RECOVERY_RATIOS.get(n, float("nan"))
        print(f"{n:>6} {pre:>12.2f} {post:>12.2f} {final:>12.4f} {rec:>12.4f}  {orig:>12.4f}")

    print()
    print(classify_outcome(cycles))
    print()


if __name__ == "__main__":
    main()
