"""
Layer 6: No silent regression across cycles.

The invariant: if a forge cycle's eval is more than 10% worse than the previous
cycle's eval AND more than 1.0 PPL worse in absolute terms, the harness halts
and dumps state. No silent advance through a regression.

This is the structural fix that makes the LoRA-on-pruned-hooks bug class
impossible to ship silently. The original bug compounded across cycles —
this invariant would have caught it at cycle 2.

Run: pytest tests/defrag_validation/test_layer6_no_silent_regression.py -v
Speed: <1 second (pure Python, no models)
"""

import pytest


# Reproduce the invariant logic in isolation so we can test it without
# spinning up the alloy executor. The actual alloy executor uses the same
# constants — see scripts/alloy_executor.py around the "Layer 6 invariant"
# comment block.

REGRESSION_THRESHOLD_RATIO = 1.10
REGRESSION_THRESHOLD_ABS = 1.0


def detect_regression(cycle_ppls: list[float]) -> tuple[bool, int | None]:
    """Replicates the alloy executor's regression check.

    Returns (regressed, halting_cycle).
    """
    for i in range(1, len(cycle_ppls)):
        prev = cycle_ppls[i - 1]
        cur = cycle_ppls[i]
        regressed_ratio = cur > prev * REGRESSION_THRESHOLD_RATIO
        regressed_abs = cur - prev > REGRESSION_THRESHOLD_ABS
        if regressed_ratio and regressed_abs:
            return True, i
    return False, None


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSmoothRecovery:
    """Healthy multi-cycle runs should not trigger the invariant."""

    def test_monotonic_improvement_passes(self):
        """Each cycle is better than the last — no halt."""
        ppls = [50.0, 30.0, 20.0, 15.0, 12.0]
        regressed, cycle = detect_regression(ppls)
        assert not regressed
        assert cycle is None

    def test_plateau_passes(self):
        """Cycles plateau but don't regress — no halt."""
        ppls = [10.0, 10.05, 10.02, 10.1, 10.08]
        regressed, _ = detect_regression(ppls)
        assert not regressed

    def test_tiny_noise_passes(self):
        """Small absolute regression below the floor doesn't trigger."""
        # 9% relative jump, 0.9 absolute — both below threshold
        ppls = [10.0, 10.9]
        regressed, _ = detect_regression(ppls)
        assert not regressed


class TestRegressionDetection:
    """Catastrophic and moderate regressions must halt."""

    def test_catastrophic_collapse_halts(self):
        """The original LoRA-on-pruned-hooks bug pattern: 62 → 7 → 501."""
        ppls = [62.0, 7.0, 501.0]
        regressed, cycle = detect_regression(ppls)
        assert regressed
        assert cycle == 2  # halts at cycle 2 (the regression)

    def test_moderate_regression_halts(self):
        """20% worse with sufficient absolute delta — halts."""
        ppls = [10.0, 12.5]
        regressed, cycle = detect_regression(ppls)
        assert regressed
        assert cycle == 1

    def test_compounding_drift_halts_at_first_violation(self):
        """A run where each cycle is slightly worse — halts as soon as the
        cumulative drift crosses the per-cycle threshold."""
        # Each cycle is +15% worse — first cycle pair triggers
        ppls = [10.0, 11.5, 13.225, 15.21]
        regressed, cycle = detect_regression(ppls)
        assert regressed
        assert cycle == 1  # caught at the first violation, not later


class TestThresholdEdges:
    """The exact threshold values matter — test both sides."""

    def test_just_below_ratio_threshold_passes(self):
        """9.9% regression — just below 10%."""
        ppls = [100.0, 109.9]
        regressed, _ = detect_regression(ppls)
        assert not regressed

    def test_just_above_ratio_threshold_halts(self):
        """10.1% regression — just above 10%, AND >1.0 absolute."""
        ppls = [100.0, 110.1]
        regressed, _ = detect_regression(ppls)
        assert regressed

    def test_high_ratio_low_absolute_passes(self):
        """50% relative jump but only 0.5 PPL absolute — below abs floor."""
        ppls = [1.0, 1.5]
        regressed, _ = detect_regression(ppls)
        assert not regressed  # the ABS floor protects tiny baselines from noise

    def test_low_ratio_high_absolute_passes(self):
        """5% relative jump but 5 PPL absolute — below ratio floor."""
        ppls = [100.0, 105.0]
        regressed, _ = detect_regression(ppls)
        assert not regressed  # the RATIO floor protects against absolute-only triggers


class TestHistoricalBugReproduction:
    """Exact reproductions of bugs the harness was built to prevent."""

    def test_lora_on_pruned_hooks_bug_caught(self):
        """The 9B forge that produced 62 → 7 → 501 — the harness halts at cycle 2,
        before the catastrophic final eval would have happened."""
        # Cycle 0 = baseline, cycle 1 = post-train (apparent improvement),
        # cycle 2 = where the bug starts to leak as hooks accumulate
        # In the actual bug, post-train PPL was 7 but final eval was 501
        # The harness sees the 501 as a cycle eval and halts
        ppls = [62.15, 7.0, 501.0]
        regressed, cycle = detect_regression(ppls)
        assert regressed
        assert cycle == 2

    def test_v6_qwen35_9b_bug_caught(self):
        """Our actual v6 forge result: baseline 62, intermediate cycles improved,
        final eval 501-ish. Harness must catch this before publish."""
        # Simulating: baseline → cycle 1 looks fine → cycle 2 the bug bites
        ppls = [62.15, 8.0, 305.83]
        regressed, cycle = detect_regression(ppls)
        assert regressed
        assert cycle == 2

    def test_smooth_8_cycles_passes(self):
        """The original gpt2-medium §4.1 cycles 1-8 (the smooth part of the
        transfer function fit) — these PPLs should NOT trigger the invariant."""
        # Approximate PPL trajectory implied by the §4.1 recovery ratios
        # (smooth exponential decay until cycle 8)
        ppls = [50.0, 38.0, 30.0, 25.0, 22.0, 20.0, 18.5, 17.5, 17.0]
        regressed, _ = detect_regression(ppls)
        assert not regressed
