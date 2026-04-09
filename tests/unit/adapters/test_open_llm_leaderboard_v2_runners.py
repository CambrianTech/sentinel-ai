"""TDD spec for the Open LLM Leaderboard v2 runner pack.

The HuggingFace Open LLM Leaderboard v2 (the most-watched general-purpose
LLM leaderboard) scores models against six benchmarks via lm-eval-harness:

    leaderboard_ifeval        — instruction following (verifiable constraints)
    leaderboard_bbh           — Big-Bench Hard (multi-task reasoning)
    leaderboard_math_hard     — MATH level 5 (hardest tier)
    leaderboard_gpqa          — Graduate-level Physics Q&A (Diamond subset)
    leaderboard_mmlu_pro      — MMLU-Pro (TIGER-Lab harder MMLU successor)
    leaderboard_musr          — Multistep Soft Reasoning

All six route through lm-eval-harness, so the right architecture is ONE
base class (LmEvalHarnessRunner) that takes a harness task name + metric
extractor, plus six thin subclasses that just declare the task name and
metric key.

The score() method takes a path to a results JSON file (the output
lm-eval-harness writes when invoked end-to-end via `lm_eval --tasks ... --output_path ...`)
and parses it into a ScoreResult. The evaluate() method runs harness
end-to-end against a model directory (codegen-equivalent for lm-eval).

Lazy import of lm_eval — Tier 1 dispatch path on a Mac without harness
installed must keep working. The lazy import fires inside evaluate(),
not score() (score reads JSON, no harness needed for re-scoring).

Test strategy:
  1. Base class exists and is importable
  2. Each runner is registered under the documented name
  3. Each runner declares the correct harness task name + metric key
  4. Each runner's score() correctly parses a synthetic results JSON
     in the lm-eval-harness output format
  5. The 6 sota_stubs that get graduated (IFEval, MMLU-Pro, GPQA-Diamond)
     are NOT in the stub list anymore — they have real bodies

Written test-first per TDD/TDValidation discipline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Base class ──────────────────────────────────────────────────────────────


def test_lm_eval_harness_base_importable():
    from eval_runners.lm_eval_harness_base import LmEvalHarnessRunner
    from eval_runners.base import BenchmarkRunner
    assert issubclass(LmEvalHarnessRunner, BenchmarkRunner)


def test_lm_eval_harness_base_has_required_class_attrs():
    """LmEvalHarnessRunner subclasses must declare task_name and metric_key
    as class attributes; the base raises if a subclass forgets."""
    from eval_runners.lm_eval_harness_base import LmEvalHarnessRunner

    class _NoTaskName(LmEvalHarnessRunner):
        name = "no_task_name"
        metric_key = "acc,none"
        # task_name omitted

    with pytest.raises((NotImplementedError, ValueError)):
        _NoTaskName().score("/nonexistent")


# ── Runner registrations ────────────────────────────────────────────────────


OPEN_LLM_V2_RUNNERS = [
    ("ifeval",      "ifeval",      "leaderboard_ifeval",     "inst_level_strict_acc,none"),
    ("bbh",         "bbh",         "leaderboard_bbh",        "acc_norm,none"),
    ("math_hard",   "math_hard",   "leaderboard_math_hard",  "exact_match,none"),
    ("gpqa",        "gpqa",        "leaderboard_gpqa",       "acc_norm,none"),
    ("mmlu_pro",    "mmlu_pro",    "leaderboard_mmlu_pro",   "acc,none"),
    ("musr",        "musr",        "leaderboard_musr",       "acc_norm,none"),
]


@pytest.mark.parametrize("module_stem,registered_name,task_name,metric_key", OPEN_LLM_V2_RUNNERS)
def test_runner_module_imports_and_has_class(module_stem, registered_name, task_name, metric_key):
    mod = __import__(f"eval_runners.{module_stem}", fromlist=[module_stem])
    # Find the BenchmarkRunner subclass in the module
    from eval_runners.base import BenchmarkRunner
    runner_classes = [
        v for v in vars(mod).values()
        if isinstance(v, type) and issubclass(v, BenchmarkRunner) and v is not BenchmarkRunner
    ]
    assert runner_classes, f"no BenchmarkRunner subclass in eval_runners/{module_stem}.py"
    cls = next(c for c in runner_classes if getattr(c, "name", None) == registered_name)
    assert cls.task_name == task_name, (
        f"{cls.__name__}.task_name should be {task_name!r}, got {cls.task_name!r}"
    )
    assert cls.metric_key == metric_key, (
        f"{cls.__name__}.metric_key should be {metric_key!r}, got {cls.metric_key!r}"
    )


@pytest.mark.parametrize("module_stem,registered_name,task_name,metric_key", OPEN_LLM_V2_RUNNERS)
def test_runner_is_registered_in_global_registry(module_stem, registered_name, task_name, metric_key):
    from eval_runners import resolve_runner
    runner = resolve_runner(registered_name)
    assert runner.name == registered_name
    assert runner.task_name == task_name


# ── score() parses lm-eval-harness results JSON correctly ───────────────────


@pytest.mark.parametrize("module_stem,registered_name,task_name,metric_key", OPEN_LLM_V2_RUNNERS)
def test_runner_score_parses_synthetic_results_json(
    tmp_path, module_stem, registered_name, task_name, metric_key
):
    """Build a synthetic lm-eval-harness results JSON in the canonical
    output format and verify the runner extracts the right metric."""
    from eval_runners import resolve_runner

    # lm-eval-harness output format: results JSON with results[task_name][metric_key]
    fake_score = 0.4271  # arbitrary 0..1 fraction
    results = {
        "results": {
            task_name: {
                metric_key: fake_score,
                "alias": task_name,
            }
        },
        "configs": {task_name: {}},
        "n-samples": {task_name: {"original": 100, "effective": 100}},
        "config": {
            "model": "synthetic",
            "model_args": "pretrained=synthetic",
            "batch_size": 1,
        },
    }
    results_path = tmp_path / f"results_{registered_name}.json"
    results_path.write_text(json.dumps(results))

    runner = resolve_runner(registered_name)
    result = runner.score(results_path)
    assert result.benchmark_name == registered_name
    assert abs(result.pass_at_1 - fake_score) < 1e-9, (
        f"{registered_name} extracted {result.pass_at_1}, expected {fake_score}"
    )
    assert result.samples_path == str(results_path)
    assert result.extras.get("task_name") == task_name
    assert result.extras.get("metric_key") == metric_key


def test_runner_score_raises_on_missing_task_in_results():
    """If the results JSON doesn't contain the runner's task_name, score()
    must raise ValueError naming the missing task — never silently substitute."""
    from eval_runners import resolve_runner
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        bad = Path(tmpdir) / "bad_results.json"
        bad.write_text(json.dumps({"results": {"some_other_task": {"acc,none": 0.5}}}))

        runner = resolve_runner("ifeval")
        with pytest.raises((ValueError, KeyError)) as exc_info:
            runner.score(bad)
        assert "leaderboard_ifeval" in str(exc_info.value)


def test_runner_score_raises_on_missing_file():
    from eval_runners import resolve_runner
    runner = resolve_runner("mmlu_pro")
    with pytest.raises(FileNotFoundError):
        runner.score("/nonexistent/path/results.json")


# ── Stubs got graduated ─────────────────────────────────────────────────────


def test_graduated_runners_no_longer_in_sota_stubs():
    """IFEval, MMLU-Pro, and GPQA Diamond used to live in sota_stubs.
    After graduation they MUST not be in REGISTRATIONS — otherwise we'd
    have two classes registered under the same name and the registry
    would raise on import."""
    from eval_runners import sota_stubs
    graduated_names = {"ifeval", "mmlu_pro", "gpqa", "gpqa_diamond"}
    stub_names = {getattr(c, "name", "") for c in sota_stubs.REGISTRATIONS}
    overlap = graduated_names & stub_names
    assert not overlap, (
        f"these runners graduated but are still registered as stubs: {overlap}. "
        f"Remove them from sota_stubs.REGISTRATIONS."
    )
